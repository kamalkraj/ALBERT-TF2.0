# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(
            self,
            initial_learning_rate,
            decay_schedule_fn,
            warmup_steps,
            power=1.0,
            name=None):
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or 'WarmUp') as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = (
                self.initial_learning_rate *
                tf.math.pow(warmup_percent_done, self.power))
            return tf.cond(global_step_float < warmup_steps_float,
                           lambda: warmup_learning_rate,
                           lambda: self.decay_schedule_fn(step),
                           name=name)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_schedule_fn': self.decay_schedule_fn,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
            'name': self.name
        }


def create_optimizer(init_lr, num_train_steps, num_warmup_steps):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        end_learning_rate=0.0)
    if num_warmup_steps:
        learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
                                  decay_schedule_fn=learning_rate_fn,
                                  warmup_steps=num_warmup_steps)
    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=['layer_norm', 'bias'])
    return optimizer


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.

    Just adding the square of the weights to the loss function is *not* the
    correct way of using L2 regularization/weight decay with Adam, since that will
    interact with the m and v parameters in strange ways.

    Instead we want ot decay the weights in a manner that doesn't interact with
    the m/v parameters. This is equivalent to adding the square of the weights to
    the loss with plain (non-momentum) SGD.
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 weight_decay_rate=0.0,
                 include_in_weight_decay=None,
                 exclude_from_weight_decay=None,
                 name='AdamWeightDecay',
                 **kwargs):
        super(AdamWeightDecay, self).__init__(
            learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {'WarmUp': WarmUp}
        return super(AdamWeightDecay, cls).from_config(
            config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                    apply_state)
        apply_state['weight_decay_rate'] = tf.constant(
            self.weight_decay_rate, name='adam_weight_decay_rate')

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var *
                apply_state['weight_decay_rate'],
                use_locking=self._use_locking)
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None):
        grads, tvars = list(zip(*grads_and_vars))
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars))

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients['lr_t'], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(
            var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(
                grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(
            var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_sparse(
                grad, var, indices, **kwargs)

    def get_config(self):
        config = super(AdamWeightDecay, self).get_config()
        config.update({
            'weight_decay_rate': self.weight_decay_rate,
        })
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


class LAMB(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the LAMB (Layer-wise Adaptive Moments)
    optimizer as TF2 tf.keras.optimizers.

    See paper [Large Batch Optimization for Deep Learning: Training BERT
    in 76 minutes](https://arxiv.org/abs/1904.00962).
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 weight_decay_rate=0.0,
                 exclude_from_weight_decay=None,
                 exclude_from_layer_adaptation=None,
                 name='LAMB',
                 **kwargs):
        """
        learning_rate: A `Tensor` or a floating point value.
          The learning rate.
        beta_1: A `float` value or a constant `float` tensor.
          The exponential decay rate for the 1st moment estimates.
        beta_2: A `float` value or a constant `float` tensor.
          The exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability.
        weight_decay_rate: weight decay rate.
        exclude_from_weight_decay: comma separated name patterns of variables
          excluded from weight decay. Variables whose name contain a substring
          matching the pattern will be excluded.
        exclude_from_layer_adaptation: comma separated name patterns of
          variables excluded from layer adaptation. Variables whose name
          contain a substring matching the pattern will be excluded.
        name: Optional name for the operations created when applying
          gradients. Defaults to "LAMB".
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
          `lr`, `decay`}. `clipnorm` is clip gradients by norm; `clipvalue`
          is clip gradients by value, `decay` is included for backward
          compatibility to allow time inverse decay of learning rate. `lr`
          is included for backward compatibility, recommended to use
          `learning_rate` instead.
        """
        super(LAMB, self).__init__(name, **kwargs)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters.
        self._set_hyper('weight_decay_rate', weight_decay_rate)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

        # This is learning rate decay for using keras learning rate schedule.
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if
        # the arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {'WarmUp': WarmUp}
        return super(LAMB, cls).from_config(
            config, custom_objects=custom_objects)

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(LAMB, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        weight_decay_rate = tf.identity(
            self._get_hyper('weight_decay_rate', var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        apply_state[(var_device, var_dtype)].update(
            dict(
                weight_decay_rate=weight_decay_rate,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = m * coefficients['beta_1_t'] + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = v * coefficients['beta_2_t'] + v_scaled_g_values
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1. - coefficients['beta_1_power'])
        v_t_hat = v_t / (1. - coefficients['beta_2_power'])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients['epsilon'])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients['weight_decay_rate'] * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        var_update = var - ratio * coefficients['lr_t'] * update
        return var.assign(var_update, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = m.assign(
            m * coefficients['beta_1_t'], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = v.assign(
            v * coefficients['beta_2_t'], use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_t_hat = m_t / (1. - coefficients['beta_1_power'])
        v_t_hat = v_t / (1. - coefficients['beta_2_power'])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients['epsilon'])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients['weight_decay_rate'] * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        var_update = var.assign_sub(
            ratio * coefficients['lr_t'] * update,
            use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t])

    def get_config(self):
        config = super(LAMB, self).get_config()
        config.update({
            'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
            'weight_decay_rate':
            self._serialize_hyperparameter('weight_decay_rate'),
            'decay':
            self._serialize_hyperparameter('decay'),
            'beta_1':
            self._serialize_hyperparameter('beta_1'),
            'beta_2':
            self._serialize_hyperparameter('beta_2'),
            'epsilon':
            self.epsilon,
        })
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for
        `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match('^(.*):\\d+$', param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
