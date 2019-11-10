"""ALBERT models that are compatible with TF 2.0."""

from __future__ import absolute_import, division, print_function

import copy

import tensorflow as tf

from albert import AlbertConfig, AlbertModel
from utils import tf_utils

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions.
  Args:
      sequence_tensor: Sequence output of `BertModel` layer of shape
        (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
        hidden units of `BertModel` layer.
      positions: Positions ids of tokens in sequence to mask for pretraining of
        with dimension (batch_size, max_predictions_per_seq) where
        `max_predictions_per_seq` is maximum number of tokens to mask out and
        predict per each sequence.
  Returns:
      Masked out sequence tensor of shape (batch_size * max_predictions_per_seq,
      num_hidden).
  """
  sequence_shape = tf_utils.get_shape_list(
      sequence_tensor, name='sequence_output_tensor')
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.keras.backend.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.keras.backend.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.keras.backend.reshape(
      sequence_tensor, [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

  return output_tensor

class ALBertPretrainLayer(tf.keras.layers.Layer):
  """Wrapper layer for pre-training a ALBERT model.
  This layer wraps an existing `albert_layer` which is a Keras Layer.
  It outputs `sequence_output` from TransformerBlock sub-layer and
  `sentence_output` which are suitable for feeding into a ALBertPretrainLoss
  layer. This layer can be used along with an unsupervised input to
  pre-train the embeddings for `albert_layer`.
  """

  def __init__(self,
               config,
               albert_layer,
               initializer=None,
               float_type=tf.float32,
               **kwargs):
    super(ALBertPretrainLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)
    self.float_type = float_type

    self.embedding_table = albert_layer.embedding_lookup.embeddings
    self.num_next_sentence_label = 2
    if initializer:
      self.initializer = initializer
    else:
      self.initializer = tf.keras.initializers.TruncatedNormal(
          stddev=self.config.initializer_range)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.output_bias = self.add_weight(
        shape=[self.config.vocab_size],
        name='predictions/output_bias',
        initializer=tf.keras.initializers.Zeros())
    self.lm_dense = tf.keras.layers.Dense(
        self.config.embedding_size,
        activation=tf_utils.get_activation(self.config.hidden_act),
        kernel_initializer=self.initializer,
        name='predictions/transform/dense')
    self.lm_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='predictions/transform/LayerNorm')

    # Next sentence binary classification dense layer including bias to match
    # TF1.x BERT variable shapes.
    with tf.name_scope('seq_relationship'):
      self.next_seq_weights = self.add_weight(
          shape=[self.num_next_sentence_label, self.config.hidden_size],
          name='output_weights',
          initializer=self.initializer)
      self.next_seq_bias = self.add_weight(
          shape=[self.num_next_sentence_label],
          name='output_bias',
          initializer=tf.keras.initializers.Zeros())
    super(ALBertPretrainLayer, self).build(unused_input_shapes)

  def __call__(self,
               pooled_output,
               sequence_output=None,
               masked_lm_positions=None,
               **kwargs):
    inputs = tf_utils.pack_inputs(
        [pooled_output, sequence_output, masked_lm_positions])
    return super(ALBertPretrainLayer, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    pooled_output = unpacked_inputs[0]
    sequence_output = unpacked_inputs[1]
    masked_lm_positions = unpacked_inputs[2]

    mask_lm_input_tensor = gather_indexes(sequence_output, masked_lm_positions)
    lm_output = self.lm_dense(mask_lm_input_tensor)
    lm_output = self.lm_layer_norm(lm_output)
    lm_output = tf.matmul(lm_output, self.embedding_table, transpose_b=True)
    lm_output = tf.nn.bias_add(lm_output, self.output_bias)
    lm_output = tf.nn.log_softmax(lm_output, axis=-1)

    logits = tf.matmul(pooled_output, self.next_seq_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, self.next_seq_bias)
    sentence_output = tf.nn.log_softmax(logits, axis=-1)
    return (lm_output, sentence_output)


class ALBertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def __init__(self, bert_config, **kwargs):
    super(ALBertPretrainLossAndMetricLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(bert_config)

  def __call__(self,
               lm_output,
               sentence_output=None,
               lm_label_ids=None,
               lm_label_weights=None,
               sentence_labels=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([
        lm_output, sentence_output, lm_label_ids, lm_label_weights,
        sentence_labels
    ])
    return super(ALBertPretrainLossAndMetricLayer, self).__call__(
        inputs, **kwargs)

  def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                   lm_per_example_loss, sentence_output, sentence_labels,
                   sentence_per_example_loss):
    """Adds metrics."""
    masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        lm_labels, lm_output)
    masked_lm_accuracy = tf.reduce_mean(masked_lm_accuracy * lm_label_weights)
    self.add_metric(
        masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')

    lm_example_loss = tf.reshape(lm_per_example_loss, [-1])
    lm_example_loss = tf.reduce_mean(lm_example_loss * lm_label_weights)
    self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')

    sentence_order_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        sentence_labels, sentence_output)
    self.add_metric(
        sentence_order_accuracy,
        name='sentence_order_accuracy',
        aggregation='mean')

    sentence_order_mean_loss = tf.reduce_mean(sentence_per_example_loss)
    self.add_metric(
        sentence_order_mean_loss, name='sentence_order_mean_loss', aggregation='mean')

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    lm_output = unpacked_inputs[0]
    sentence_output = unpacked_inputs[1]
    lm_label_ids = unpacked_inputs[2]
    lm_label_ids = tf.keras.backend.reshape(lm_label_ids, [-1])
    lm_label_ids_one_hot = tf.keras.backend.one_hot(lm_label_ids,
                                                    self.config.vocab_size)
    lm_label_weights = tf.keras.backend.cast(unpacked_inputs[3], tf.float32)
    lm_label_weights = tf.keras.backend.reshape(lm_label_weights, [-1])
    lm_per_example_loss = -tf.keras.backend.sum(
        lm_output * lm_label_ids_one_hot, axis=[-1])
    numerator = tf.keras.backend.sum(lm_label_weights * lm_per_example_loss)
    denominator = tf.keras.backend.sum(lm_label_weights) + 1e-5
    mask_label_loss = numerator / denominator

    sentence_labels = unpacked_inputs[4]
    sentence_labels = tf.keras.backend.reshape(sentence_labels, [-1])
    sentence_label_one_hot = tf.keras.backend.one_hot(sentence_labels, 2)
    per_example_loss_sentence = -tf.keras.backend.sum(
        sentence_label_one_hot * sentence_output, axis=-1)
    sentence_loss = tf.keras.backend.mean(per_example_loss_sentence)
    loss = mask_label_loss + sentence_loss
    # TODO(hongkuny): Avoids the hack and switches add_loss.
    final_loss = tf.fill(
        tf.keras.backend.shape(per_example_loss_sentence), loss)

    self._add_metrics(lm_output, lm_label_ids, lm_label_weights,
                      lm_per_example_loss, sentence_output, sentence_labels,
                      per_example_loss_sentence)
    return final_loss


def pretrain_model(albert_config,
                   seq_length,
                   max_predictions_per_seq,
                   initializer=None):
  """Returns model to be used for pre-training.
  Args:
      albert_config: Configuration that defines the core ALBERT model.
      seq_length: Maximum sequence length of the training data.
      max_predictions_per_seq: Maximum number of tokens in sequence to mask out
        and use for pretraining.
      initializer: Initializer for weights in BertPretrainLayer.
  Returns:
      Pretraining model as well as core BERT submodel from which to save
      weights after pretraining.
  """

  input_word_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
  input_mask = tf.keras.layers.Input(
      shape=(seq_length,), name='input_mask', dtype=tf.int32)
  input_type_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_type_ids', dtype=tf.int32)
  masked_lm_positions = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,),
      name='masked_lm_positions',
      dtype=tf.int32)
  masked_lm_weights = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,),
      name='masked_lm_weights',
      dtype=tf.int32)
  next_sentence_labels = tf.keras.layers.Input(
      shape=(1,), name='next_sentence_labels', dtype=tf.int32)
  masked_lm_ids = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)

  
  float_type = tf.float32
  albert_encoder = "albert_model"
  albert_layer = AlbertModel(config=albert_config, float_type=float_type, name=albert_encoder)
  pooled_output, sequence_output = albert_layer(input_word_ids, input_mask,
                                                  input_type_ids)
  albert_submodel = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output])
  
  pooled_output = albert_submodel.outputs[0]
  sequence_output = albert_submodel.outputs[1]
  
  pretrain_layer = ALBertPretrainLayer(
      albert_config,
      albert_submodel.get_layer(albert_encoder),
      initializer=initializer,
      name='cls')
  lm_output, sentence_output = pretrain_layer(pooled_output, sequence_output,
                                              masked_lm_positions)

  pretrain_loss_layer = ALBertPretrainLossAndMetricLayer(albert_config)
  output_loss = pretrain_loss_layer(lm_output, sentence_output, masked_lm_ids,
                                    masked_lm_weights, next_sentence_labels)

  return tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids,
          'masked_lm_positions': masked_lm_positions,
          'masked_lm_ids': masked_lm_ids,
          'masked_lm_weights': masked_lm_weights,
          'next_sentence_labels': next_sentence_labels,
      },
      outputs=output_loss),albert_submodel