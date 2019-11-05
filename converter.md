## TensorFLow-Hub to TF 2.0 weights converison


### TF-HUB weights

#### Download
|                                   Verison 1                                   |                                   Version 2                                   |
|:-----------------------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
|   [base](https://storage.googleapis.com/tfhub-modules/google/albert_base/1.tar.gz)  |   [base](https://storage.googleapis.com/tfhub-modules/google/albert_base/2.tar.gz)  |
|  [large](https://storage.googleapis.com/tfhub-modules/google/albert_large/1.tar.gz)  |  [large](https://storage.googleapis.com/tfhub-modules/google/albert_large/2.tar.gz)  |
|  [xlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xlarge/1.tar.gz) |  [xlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xlarge/2.tar.gz) |
| [xxlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xxlarge/1.tar.gz) | [xxlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xxlarge/2.tar.gz) |


### Converison Example

```bash
export MODEL_DIR=base
wget https://storage.googleapis.com/tfhub-modules/google/albert_base/2.tar.gz
mkdir ${MODEL_DIR}
tar -xvzf 2.tar.gz --directory=${MODEL_DIR}
# Converting weights to TF 2.0
python converter.py --tf_hub_path=${MODEL_DIR}/
# Copy albert_config.json to config.json
cp ${MODEL_DIR}/assets/albert_config.json ${MODEL_DIR}/config.json
# Rename assets to vocab
mv ${MODEL_DIR}/assets/ ${MODEL_DIR}/vocab
# Delete unwanted files
rm -rf ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/variables/ ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/tfhub_module.pb
```