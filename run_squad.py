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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0 in tf2.0."""

from __future__ import absolute_import, division, print_function

import functools
import json
import os

import tensorflow as tf
from absl import app, flags, logging

import input_pipeline
import squad_lib
import tokenization
from albert import AlbertConfig, AlbertModel
from model_training_utils import run_customized_training_loop
from optimization import LAMB, AdamWeightDecay, WarmUp

flags.DEFINE_enum(
    'mode', 'train_and_predict',
    ['train_and_predict', 'train', 'predict', 'export_only'],
    'One of {"train_and_predict", "train", "predict", "export_only"}. '
    '`train_and_predict`: both train and predict to a json file. '
    '`train`: only trains the model. '
    '`predict`: predict answers from the squad json file. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`.')

flags.DEFINE_string('train_data_path', '',
                    'Training data path with train tfrecords.')

flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')

# Model training specific flags.
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
# Predict processing related.
flags.DEFINE_string('predict_file', None,
                    'Prediction data path with train tfrecords.')
flags.DEFINE_integer('predict_batch_size', 8,
                     'Total batch size for predicting.')

flags.DEFINE_integer(
    'n_best_size', 20,
    'The total number of n-best predictions to generate in the '
    'nbest_predictions.json output file.')

flags.DEFINE_integer(
    'max_answer_length', 30,
    'The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another.')

flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")


flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string(
    "model_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_enum(
    "strategy_type", "one", ["one", "mirror"],
    "Training strategy for single or multi gpu training")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ALBERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("weight_decay", 0.01, "weight_decay")

flags.DEFINE_float("adam_epsilon", 1e-6, "adam_epsilon")

flags.DEFINE_integer("num_train_epochs", 3,
                     "Total number of training epochs to perform.")

flags.DEFINE_bool("enable_xla", False, "enables XLA")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_enum("optimizer","AdamW",["LAMB","AdamW"],"Optimizer for training LAMB/AdamW")

flags.DEFINE_integer("seed", 42, "random_seed")

FLAGS = flags.FLAGS


class ALBertSquadLogitsLayer(tf.keras.layers.Layer):
    """Returns a layer that computes custom logits for BERT squad model."""

    def __init__(self, initializer=None, float_type=tf.float32, **kwargs):
        super(ALBertSquadLogitsLayer, self).__init__(**kwargs)
        self.initializer = initializer
        self.float_type = float_type

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.final_dense = tf.keras.layers.Dense(
            units=2, kernel_initializer=self.initializer, name='final_dense')
        super(ALBertSquadLogitsLayer, self).build(unused_input_shapes)

    def call(self, inputs):
        """Implements call() for the layer."""
        sequence_output = inputs

        input_shape = sequence_output.shape.as_list()
        sequence_length = input_shape[1]
        num_hidden_units = input_shape[2]

        final_hidden_input = tf.keras.backend.reshape(sequence_output,
                                                      [-1, num_hidden_units])
        logits = self.final_dense(final_hidden_input)
        logits = tf.keras.backend.reshape(logits, [-1, sequence_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])
        unstacked_logits = tf.unstack(logits, axis=0)
        if self.float_type == tf.float16:
            unstacked_logits = tf.cast(unstacked_logits, tf.float32)
        return unstacked_logits[0], unstacked_logits[1]


def set_config_v2(enable_xla=False):
    """Config eager context according to flag values using TF 2.0 API."""
    if enable_xla:
        tf.config.optimizer.set_jit(True)
        # Disable PinToHostOptimizer in grappler when enabling XLA because it
        # causes OOM and performance regression.
        tf.config.optimizer.set_experimental_options(
            {'pin_to_host_optimization': False}
        )


def squad_loss_fn(start_positions,
                  end_positions,
                  start_logits,
                  end_logits,
                  loss_factor=1.0):
    """Returns sparse categorical crossentropy for start/end logits."""
    start_loss = tf.keras.backend.sparse_categorical_crossentropy(
        start_positions, start_logits, from_logits=True)
    end_loss = tf.keras.backend.sparse_categorical_crossentropy(
        end_positions, end_logits, from_logits=True)

    total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
    total_loss *= loss_factor
    return total_loss


def get_loss_fn(loss_factor=1.0):
    """Gets a loss function for squad task."""

    def _loss_fn(labels, model_outputs):
        start_positions = labels['start_positions']
        end_positions = labels['end_positions']
        _, start_logits, end_logits = model_outputs
        return squad_loss_fn(
            start_positions,
            end_positions,
            start_logits,
            end_logits,
            loss_factor=loss_factor)

    return _loss_fn


def get_raw_results(predictions):
    """Converts multi-replica predictions to RawResult."""
    for unique_ids, start_logits, end_logits in zip(predictions['unique_ids'],
                                                    predictions['start_logits'],
                                                    predictions['end_logits']):
        for values in zip(unique_ids.numpy(), start_logits.numpy(),
                          end_logits.numpy()):
            yield squad_lib.RawResult(
                unique_id=values[0],
                start_logits=values[1].tolist(),
                end_logits=values[2].tolist())


def predict_squad_customized(strategy, input_meta_data, albert_config,
                             predict_tfrecord_path, num_steps):
    """Make predictions using a Bert-based squad model."""
    predict_dataset = input_pipeline.create_squad_dataset(
        predict_tfrecord_path,
        input_meta_data['max_seq_length'],
        FLAGS.predict_batch_size,
        is_training=False)
    predict_iterator = iter(
        strategy.experimental_distribute_dataset(predict_dataset))

    with strategy.scope():
        # add comments for #None,0.1,1,0
        squad_model = get_model(albert_config, input_meta_data['max_seq_length'],
                                None, 0.1, 1, 0)

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    logging.info('Restoring checkpoints from %s', checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=squad_model)
    checkpoint.restore(checkpoint_path).expect_partial()

    @tf.function
    def predict_step(iterator):
        """Predicts on distributed devices."""

        def _replicated_step(inputs):
            """Replicated prediction calculation."""
            x, _ = inputs
            unique_ids, start_logits, end_logits = squad_model(
                x, training=False)
            return dict(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits)

        outputs = strategy.experimental_run_v2(
            _replicated_step, args=(next(iterator),))
        return tf.nest.map_structure(strategy.experimental_local_results, outputs)

    all_results = []
    for _ in range(num_steps):
        predictions = predict_step(predict_iterator)
        for result in get_raw_results(predictions):
            all_results.append(result)
        if len(all_results) % 100 == 0:
            logging.info('Made predictions for %d records.', len(all_results))
    return all_results


def get_model(albert_config, max_seq_length, init_checkpoint, learning_rate,
              num_train_steps, num_warmup_steps):
    """Returns keras fuctional model"""
    float_type = tf.float32
    # hidden_dropout_prob = 0.9 # as per original code relased
    unique_ids = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, name='unique_ids')
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    albert_layer = AlbertModel(config=albert_config, float_type=float_type)

    _, sequence_output = albert_layer(
        input_word_ids, input_mask, input_type_ids)

    albert_model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                  outputs=[sequence_output])

    if init_checkpoint != None:
        albert_model.load_weights(init_checkpoint)

    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=albert_config.initializer_range)

    squad_logits_layer = ALBertSquadLogitsLayer(
        initializer=initializer, float_type=float_type, name='squad_logits')

    start_logits, end_logits = squad_logits_layer(sequence_output)

    squad_model = tf.keras.Model(
        inputs={
            'unique_ids': unique_ids,
            'input_ids': input_word_ids,
            'input_mask': input_mask,
            'segment_ids': input_type_ids,
        },
        outputs=[unique_ids, start_logits, end_logits],
        name='squad_model')

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
                                                                     decay_steps=num_train_steps, end_learning_rate=0.0)
    if num_warmup_steps:
        learning_rate_fn = WarmUp(initial_learning_rate=learning_rate,
                                  decay_schedule_fn=learning_rate_fn,
                                  warmup_steps=num_warmup_steps)

    if FLAGS.optimizer == "LAMB":
        optimizer_fn = LAMB
    else:
        optimizer_fn = AdamWeightDecay

    optimizer = optimizer_fn(
        learning_rate=learning_rate_fn,
        weight_decay_rate=FLAGS.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=FLAGS.adam_epsilon,
        exclude_from_weight_decay=['layer_norm', 'bias'])

    squad_model.optimizer = optimizer

    return squad_model


def train_squad(strategy,
                input_meta_data,
                custom_callbacks=None,
                run_eagerly=False):
    """Run bert squad training."""
    if strategy:
        logging.info('Training using customized training loop with distribution'
                     ' strategy.')
    # Enables XLA in Session Config. Should not be set for TPU.
    if FLAGS.enable_xla:
        set_config_v2(FLAGS.enable_xla)

    num_train_examples = input_meta_data['train_data_size']
    max_seq_length = input_meta_data['max_seq_length']
    num_train_steps = None
    num_warmup_steps = None
    steps_per_epoch = int(num_train_examples / FLAGS.train_batch_size)
    num_train_steps = int(
        num_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    with strategy.scope():
        albert_config = AlbertConfig.from_json_file(FLAGS.albert_config_file)
        model = get_model(albert_config, input_meta_data['max_seq_length'],
                          FLAGS.init_checkpoint, FLAGS.learning_rate,
                          num_train_steps, num_warmup_steps)

    train_input_fn = functools.partial(
        input_pipeline.create_squad_dataset,
        FLAGS.train_data_path,
        max_seq_length,
        FLAGS.train_batch_size,
        is_training=True)

    # The original BERT model does not scale the loss by
    # 1/num_replicas_in_sync. It could be an accident. So, in order to use
    # the same hyper parameter, we do the same thing here by keeping each
    # replica loss as it is.
    loss_fn = get_loss_fn(loss_factor=1.0 /strategy.num_replicas_in_sync)

    trained_model = run_customized_training_loop(
        strategy=strategy,
        model=model,
        loss_fn=loss_fn,
        model_dir=FLAGS.model_dir,
        train_input_fn=train_input_fn,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_epoch,
        epochs=FLAGS.num_train_epochs,
        run_eagerly=run_eagerly,
        custom_callbacks=custom_callbacks)
    trained_model.save(f"SQuAD/model.h5")


def predict_squad(strategy, input_meta_data):
    """Makes predictions for a squad dataset."""
    albert_config = AlbertConfig.from_json_file(FLAGS.albert_config_file)
    doc_stride = input_meta_data['doc_stride']
    max_query_length = input_meta_data['max_query_length']

    eval_examples = squad_lib.read_squad_examples(
        input_file=FLAGS.predict_file,
        is_training=False)

    tokenizer = tokenization.FullTokenizer(vocab_file=None,
                                           spm_model_file=FLAGS.spm_model_file, do_lower_case=FLAGS.do_lower_case)

    eval_writer = squad_lib.FeatureWriter(
        filename=os.path.join(FLAGS.model_dir, 'eval.tf_record'),
        is_training=False)
    eval_features = []

    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
    dataset_size = squad_lib.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=input_meta_data['max_seq_length'],
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=eval_writer.process_feature)
    eval_writer.close()

    logging.info('***** Running predictions *****')
    logging.info('  Num orig examples = %d', len(eval_examples))
    logging.info('  Num split examples = %d', len(eval_features))
    logging.info('  Batch size = %d', FLAGS.predict_batch_size)

    num_steps = int(dataset_size / FLAGS.predict_batch_size)
    all_results = predict_squad_customized(strategy, input_meta_data, albert_config,
                                           eval_writer.filename, num_steps)

    output_prediction_file = os.path.join(FLAGS.model_dir, 'predictions.json')
    output_nbest_file = os.path.join(FLAGS.model_dir, 'nbest_predictions.json')
    output_null_log_odds_file = os.path.join(FLAGS.model_dir, 'null_odds.json')

    squad_lib.write_predictions(
        eval_examples,
        eval_features,
        all_results,
        FLAGS.n_best_size,
        FLAGS.max_answer_length,
        FLAGS.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file)


def export_squad(model_export_path, input_meta_data):
    """Exports a trained model as a `SavedModel` for inference.

    Args:
      model_export_path: a string specifying the path to the SavedModel directory.
      input_meta_data: dictionary containing meta data about input and model.

    Raises:
      Export path is not specified, got an empty string or None.
    """
    pass


def main(_):
    # Users should always run this script under TF 2.x
    assert tf.version.VERSION.startswith('2.')

    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))

    strategy = None
    if FLAGS.strategy_type == 'mirror':
        strategy = tf.distribute.MirroredStrategy()
    elif FLAGS.strategy_type == 'one':
        strategy = tf.distribute.OneDeviceStrategy()
    else:
        raise ValueError('The distribution strategy type is not supported: %s' %
                         FLAGS.strategy_type)
    if FLAGS.mode in ('train', 'train_and_predict'):
        train_squad(strategy, input_meta_data)
    if FLAGS.mode in ('predict', 'train_and_predict'):
        predict_squad(strategy, input_meta_data)


if __name__ == '__main__':
    flags.mark_flag_as_required('albert_config_file')
    flags.mark_flag_as_required('model_dir')
    app.run(main)
