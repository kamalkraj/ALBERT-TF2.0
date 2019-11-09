from __future__ import absolute_import, division, print_function

import os
import re

import tensorflow as tf
from absl import app, flags

from albert import AlbertConfig, AlbertModel

from albert_model import pretrain_model

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tf_hub_path", None,
    "tf_hub_path for download models")

flags.DEFINE_enum("model_type","albert_encoder",["albert_encoder","albert"],
                  "Select model type for weight conversion.\n"
                  "albert_enoder for finetuning tasks.\n"
                  "albert for MLM & SOP FineTuning on domain specific data.")

flags.DEFINE_integer("version",2,"tf hub model version to convert 1 or 2.")

flags.DEFINE_enum("model","base",["base", "large", "xlarge", "xxlarge"],"model for converison")

weight_map = {
    "bert/embeddings/word_embeddings": "albert_model/word_embeddings/embeddings:0",
    "bert/embeddings/token_type_embeddings": "albert_model/embedding_postprocessor/type_embeddings:0",
    "bert/embeddings/position_embeddings": "albert_model/embedding_postprocessor/position_embeddings:0",
    "bert/embeddings/LayerNorm/beta": "albert_model/embedding_postprocessor/layer_norm/beta:0",
    "bert/embeddings/LayerNorm/gamma": "albert_model/embedding_postprocessor/layer_norm/gamma:0",
    "bert/encoder/embedding_hidden_mapping_in/kernel": "albert_model/embedding_postprocessor/embedding_hidden_mapping_in/kernel:0",
    "bert/encoder/embedding_hidden_mapping_in/bias": "albert_model/embedding_postprocessor/embedding_hidden_mapping_in/bias:0",
    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel": "albert_model/encoder/shared_layer/self_attention/query/kernel:0",
    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias": "albert_model/encoder/shared_layer/self_attention/query/bias:0",
    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel": "albert_model/encoder/shared_layer/self_attention/key/kernel:0",
    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias": "albert_model/encoder/shared_layer/self_attention/key/bias:0",
    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel": "albert_model/encoder/shared_layer/self_attention/value/kernel:0",
    "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias": "albert_model/encoder/shared_layer/self_attention/value/bias:0",
    "bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel": "albert_model/encoder/shared_layer/self_attention_output/kernel:0",
    "bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias": "albert_model/encoder/shared_layer/self_attention_output/bias:0",
    "bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta": "albert_model/encoder/shared_layer/self_attention_layer_norm/beta:0",
    "bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma": "albert_model/encoder/shared_layer/self_attention_layer_norm/gamma:0",
    "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel": "albert_model/encoder/shared_layer/intermediate/kernel:0",
    "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias": "albert_model/encoder/shared_layer/intermediate/bias:0",
    "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel": "albert_model/encoder/shared_layer/output/kernel:0",
    "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias": "albert_model/encoder/shared_layer/output/bias:0",
    "bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta": "albert_model/encoder/shared_layer/output_layer_norm/beta:0",
    "bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma": "albert_model/encoder/shared_layer/output_layer_norm/gamma:0",
    "bert/pooler/dense/kernel": "albert_model/pooler_transform/kernel:0",
    "bert/pooler/dense/bias": "albert_model/pooler_transform/bias:0",
    "cls/predictions/transform/dense/kernel": "cls/predictions/transform/dense/kernel:0",
    "cls/predictions/transform/dense/bias": "cls/predictions/transform/dense/bias:0",
    "cls/predictions/transform/LayerNorm/beta": "cls/predictions/transform/LayerNorm/beta:0",
    "cls/predictions/transform/LayerNorm/gamma": "cls/predictions/transform/LayerNorm/gamma:0",
    "cls/predictions/output_bias": "cls/predictions/output_bias:0",
    'cls/seq_relationship/output_weights': 'cls/seq_relationship/output_weights:0',
    'cls/seq_relationship/output_bias': 'cls/seq_relationship/output_bias:0'
}



weight_map = {v: k for k, v in weight_map.items()}


def main(_):

    tfhub_model_path = FLAGS.tf_hub_path
    max_seq_length = 512
    float_type = tf.float32
    

    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    if FLAGS.version == 2:
        albert_config = AlbertConfig.from_json_file(
            os.path.join(tfhub_model_path, "assets", "albert_config.json"))
    else:
        albert_config = AlbertConfig.from_json_file(
            os.path.join("model_configs", FLAGS.model, "config.json"))

    tags = []

    stock_values = {}

    with tf.Graph().as_default():
        sm = tf.compat.v2.saved_model.load(tfhub_model_path, tags=tags)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            stock_values = {v.name.split(":")[0]: v.read_value()
                            for v in sm.variables}
            stock_values = sess.run(stock_values)

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []

    if FLAGS.model_type == "albert_encoder":
        albert_layer = AlbertModel(config=albert_config, float_type=float_type)

        pooled_output, sequence_output = albert_layer(input_word_ids, input_mask,
                                                  input_type_ids)
        albert_model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids],
        outputs=[pooled_output, sequence_output])
        albert_params = albert_model.weights
        param_values = tf.keras.backend.batch_get_value(albert_model.weights)
    else:
        albert_full_model,_ = pretrain_model(albert_config,max_seq_length,max_predictions_per_seq=20)
        albert_layer = albert_full_model.get_layer("albert_model")
        albert_params = albert_full_model.weights
        param_values = tf.keras.backend.batch_get_value(albert_full_model.weights)

    for ndx, (param_value, param) in enumerate(zip(param_values, albert_params)):
        stock_name = weight_map[param.name]

        if stock_name in stock_values:
            ckpt_value = stock_values[stock_name]

            if param_value.shape != ckpt_value.shape:
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(
                param.name, stock_name, tfhub_model_path))
            skip_count += 1
    tf.keras.backend.batch_set_value(weight_value_tuples)

    print("Done loading {} ALBERT weights from: {} into {} (prefix:{}). "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), tfhub_model_path, albert_layer, "albert", skip_count, len(skipped_weight_value_tuples)))
    print("Unused weights from saved model:",
          "\n\t" + "\n\t".join(sorted(set(stock_values.keys()).difference(loaded_weights))))

    if FLAGS.model_type == "albert_encoder":
        albert_model.save_weights(f"{tfhub_model_path}/tf2_model.h5")
    else:
        albert_full_model.save_weights(f"{tfhub_model_path}/tf2_model_full.h5")

if __name__ == "__main__":
    flags.mark_flag_as_required("tf_hub_path")
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("version")
    flags.mark_flag_as_required("model_type")
    app.run(main)
