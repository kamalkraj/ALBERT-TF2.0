# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python2, python3
"""BERT finetuning runner with sentence piece tokenization."""

from __future__ import absolute_import, division, print_function

import collections
import csv
import functools
import json
import os

import numpy as np
import six
import tensorflow as tf
from absl import app, flags, logging
from six.moves import zip

import classifier_data_lib
import tokenization
from albert import AlbertConfig, AlbertModel
from input_pipeline import create_classifier_dataset
from model_training_utils import run_customized_training_loop
from optimization import LAMB, AdamWeightDecay, WarmUp

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "train_data_path", None,
    "train_data path for tfrecords for the task.")

flags.DEFINE_string(
    "eval_data_path", None,
    "eval_data path for tfrecords for the task.")

flags.DEFINE_string(
    "predict_data_path", None,
    "predict_data path for tfrecords for the task.")

flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_enum(
    "strategy_type", "one", ["one", "mirror"],
    "Training strategy for single or multi gpu training")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ALBERT model).")

flags.DEFINE_string("input_meta_data_path",None,"input_meta_data_path")


flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_float("classifier_dropout",0.1,"classification layer dropout")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False ,"Whether to run prediction on the test set")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("weight_decay", 0.01, "weight_decay")

flags.DEFINE_float("adam_epsilon", 1e-6, "adam_epsilon")

flags.DEFINE_integer("num_train_epochs", 3,
                   "Total number of training epochs to perform.")

flags.DEFINE_bool("enable_xla",False, "enables XLA")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_enum("optimizer","AdamW",["LAMB","AdamW"],"Optimizer for training LAMB/AdamW")

flags.DEFINE_bool("custom_training_loop",False,"Use Cutsom training loop instead of model.fit")

flags.DEFINE_integer("seed", 42, "random_seed")

def set_config_v2(enable_xla=False):
  """Config eager context according to flag values using TF 2.0 API."""
  if enable_xla:
    tf.config.optimizer.set_jit(True)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it
    # causes OOM and performance regression.
    tf.config.optimizer.set_experimental_options(
        {'pin_to_host_optimization': False}
    )

def get_loss_fn(num_classes, loss_factor=1.0):
  """Gets the classification loss function."""

  def classification_loss_fn(labels, logits):
    """Classification loss."""
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    loss *= loss_factor
    return loss

  return classification_loss_fn

def get_loss_fn_v2(loss_factor=1.0):
    """Gets the loss function for STS."""

    def sts_loss_fn(labels, logits):
        """STS loss"""
        logits = tf.squeeze(logits, [-1])
        per_example_loss = tf.square(logits - labels)
        loss = tf.reduce_mean(per_example_loss)
        loss *= loss_factor
        return loss
    
    return sts_loss_fn

def get_model(albert_config, max_seq_length, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,loss_multiplier):
    """Returns keras fuctional model"""
    float_type = tf.float32
    hidden_dropout_prob = FLAGS.classifier_dropout # as per original code relased
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    albert_layer = AlbertModel(config=albert_config, float_type=float_type)

    pooled_output, _ = albert_layer(input_word_ids, input_mask, input_type_ids)

    albert_model = tf.keras.Model(inputs=[input_word_ids,input_mask,input_type_ids],
                                  outputs=[pooled_output])

    albert_model.load_weights(init_checkpoint)

    initializer = tf.keras.initializers.TruncatedNormal(stddev=albert_config.initializer_range)

    output = tf.keras.layers.Dropout(rate=hidden_dropout_prob)(pooled_output)

    output = tf.keras.layers.Dense(
        num_labels,
        kernel_initializer=initializer,
        name='output',
        dtype=float_type)(
            output)
    model = tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs=output)

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
                                                decay_steps=num_train_steps,end_learning_rate=0.0)
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
    
    if FLAGS.task_name.lower() == 'sts':
        loss_fct = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizer,loss=loss_fct,metrics=['mse'])
    else:
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer,loss=loss_fct,metrics=['accuracy'])

    return model



def main(_):
  logging.set_verbosity(logging.INFO)

  if FLAGS.enable_xla:
	  set_config_v2(FLAGS.enable_xla)

  strategy = None
  if FLAGS.strategy_type == "one":
	  strategy = tf.distribute.OneDeviceStrategy("GPU:0")
  elif FLAGS.strategy_type == "mirror":
	  strategy = tf.distribute.MirroredStrategy()
  else:
	  raise ValueError('The distribution strategy type is not supported: %s' %
                     FLAGS.strategy_type)

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  num_labels = input_meta_data["num_labels"]
  FLAGS.max_seq_length = input_meta_data["max_seq_length"]
  processor_type = input_meta_data['processor_type']

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  albert_config = AlbertConfig.from_json_file(FLAGS.albert_config_file)

  if FLAGS.max_seq_length > albert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the ALBERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, albert_config.max_position_embeddings))

  tf.io.gfile.makedirs(FLAGS.output_dir)

  num_train_steps = None
  num_warmup_steps = None
  steps_per_epoch = None
  if FLAGS.do_train:
    len_train_examples = input_meta_data['train_data_size']
    steps_per_epoch = int(len_train_examples / FLAGS.train_batch_size)
    num_train_steps = int(
        len_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  loss_multiplier = 1.0 / strategy.num_replicas_in_sync

  with strategy.scope():
	  model = get_model(
		  albert_config=albert_config,
		  max_seq_length=FLAGS.max_seq_length,
		  num_labels=num_labels,
		  init_checkpoint=FLAGS.init_checkpoint,
		  learning_rate=FLAGS.learning_rate,
		  num_train_steps=num_train_steps,
		  num_warmup_steps=num_warmup_steps,
          loss_multiplier=loss_multiplier)
  model.summary()

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len_train_examples)
    logging.info("  Batch size = %d", FLAGS.train_batch_size)
    logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = functools.partial(
      create_classifier_dataset,
      FLAGS.train_data_path,
      seq_length=FLAGS.max_seq_length,
      batch_size=FLAGS.train_batch_size,
      drop_remainder=False)

    eval_input_fn = functools.partial(
      create_classifier_dataset,
      FLAGS.eval_data_path,
      seq_length=FLAGS.max_seq_length,
      batch_size=FLAGS.eval_batch_size,
      is_training=False,
      drop_remainder=False)

    with strategy.scope():

        summary_dir = os.path.join(FLAGS.output_dir, 'summaries')
        summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
        checkpoint_path = os.path.join(FLAGS.output_dir, 'checkpoint')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)
        custom_callbacks = [summary_callback, checkpoint_callback]

        def metric_fn():
            if FLAGS.task_name.lower() == "sts":
                return tf.keras.metrics.MeanSquaredError(dtype=tf.float32)
            else:
                return tf.keras.metrics.SparseCategoricalAccuracy(dtype=tf.float32)

        if FLAGS.custom_training_loop:
            if FLAGS.task_name.lower() == "sts":
                loss_fn = get_loss_fn_v2(loss_factor=loss_multiplier)
            else:
                loss_fn = get_loss_fn(num_labels,loss_factor=loss_multiplier)
            model = run_customized_training_loop(strategy = strategy,
                    model = model,
                    loss_fn = loss_fn,
                    model_dir = checkpoint_path,
                    train_input_fn = train_input_fn,
                    steps_per_epoch = steps_per_epoch,
                    epochs=FLAGS.num_train_epochs,
                    eval_input_fn = eval_input_fn,
                    eval_steps = int(input_meta_data['eval_data_size']/FLAGS.eval_batch_size),
                    metric_fn = metric_fn,
                    custom_callbacks = custom_callbacks)
        else:
            training_dataset = train_input_fn()
            evaluation_dataset = eval_input_fn()
            model.fit(x=training_dataset,validation_data=evaluation_dataset,epochs=FLAGS.num_train_epochs,callbacks=custom_callbacks)


  if FLAGS.do_eval:

    len_eval_examples = input_meta_data['eval_data_size']

    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len_eval_examples)
    logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    evaluation_dataset = eval_input_fn()
    with strategy.scope():
        loss,accuracy = model.evaluate(evaluation_dataset)

    print(f"loss : {loss} , Accuracy : {accuracy}")

  if FLAGS.do_predict:

    logging.info("***** Running prediction*****")
    flags.mark_flag_as_required("input_data_dir")
    flags.mark_flag_as_required("predict_data_path")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=None,spm_model_file=FLAGS.spm_model_file, do_lower_case=FLAGS.do_lower_case)

    processors = {
    "cola": classifier_data_lib.ColaProcessor,
    "sts": classifier_data_lib.StsbProcessor,
    "sst": classifier_data_lib.Sst2Processor,
    "mnli": classifier_data_lib.MnliProcessor,
    "qnli": classifier_data_lib.QnliProcessor,
    "qqp": classifier_data_lib.QqpProcessor,
    "rte": classifier_data_lib.RteProcessor,
    "mrpc": classifier_data_lib.MrpcProcessor,
    "wnli": classifier_data_lib.WnliProcessor,
    "xnli": classifier_data_lib.XnliProcessor,
    }
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    predict_examples = processor.get_test_examples(FLAGS.input_data_dir)

    label_list = processor.get_labels()
    label_map = {i:label for i,label in enumerate(label_list)}

    classifier_data_lib.file_based_convert_examples_to_features(predict_examples,
                                        label_list, input_meta_data['max_seq_length'],
                                        tokenizer, FLAGS.predict_data_path)

    predict_input_fn = functools.partial(
    create_classifier_dataset,
    FLAGS.predict_data_path,
    seq_length=input_meta_data['max_seq_length'],
    batch_size=FLAGS.eval_batch_size,
    is_training=False,
    drop_remainder=False)
    prediction_dataset = predict_input_fn()

    with strategy.scope():
        logits = model.predict(prediction_dataset)
        if FLAGS.task_name.lower() == "sts":
            predictions = logits
            probabilities = logits
        else:
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            probabilities = tf.nn.softmax(logits, axis=-1)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    output_submit_file = os.path.join(FLAGS.output_dir, "submit_results.tsv")
    with tf.io.gfile.GFile(output_predict_file, "w") as pred_writer,\
        tf.io.gfile.GFile(output_submit_file, "w") as sub_writer:
        logging.info("***** Predict results *****")
        for (example, probability, prediction) in zip(predict_examples, probabilities, predictions):
            output_line = "\t".join(
                str(class_probability.numpy())
                for class_probability in probability) + "\n"
            pred_writer.write(output_line)

            actual_label = label_map[int(prediction)]
            sub_writer.write(
                six.ensure_str(example.guid) + "\t" + actual_label + "\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("train_data_path")
  flags.mark_flag_as_required("eval_data_path")
  flags.mark_flag_as_required("input_meta_data_path")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("spm_model_file")
  flags.mark_flag_as_required("albert_config_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
