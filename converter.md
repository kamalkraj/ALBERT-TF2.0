## TensorFLow-Hub to TF 2.0 weights converison


### TF-HUB weights

#### Download
|                                   Verison 1                                   |                                   Version 2                                   |
|:-----------------------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
|   [base](https://storage.googleapis.com/tfhub-modules/google/albert_base/1.tar.gz)  |   [base](https://storage.googleapis.com/tfhub-modules/google/albert_base/2.tar.gz)  |
|  [large](https://storage.googleapis.com/tfhub-modules/google/albert_large/1.tar.gz)  |  [large](https://storage.googleapis.com/tfhub-modules/google/albert_large/2.tar.gz)  |
|  [xlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xlarge/1.tar.gz) |  [xlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xlarge/2.tar.gz) |
| [xxlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xxlarge/1.tar.gz) | [xxlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xxlarge/2.tar.gz) |


### Converison flags
```
  --model: <base|large|xlarge|xxlarge>: model for converison
    (default: 'base')
  --model_type: <albert_encoder|albert>: Select model type for weight conversion.
    albert_enoder for finetuning tasks.
    albert for MLM & SOP FineTuning on domain specific data.
    (default: 'albert_encoder')
  --tf_hub_path: tf_hub_path for download models
  --version: tf hub model version to convert 1 or 2.
    (default: '2')
    (an integer)
```

### Converison Example

```bash
export MODEL_DIR=base
wget https://storage.googleapis.com/tfhub-modules/google/albert_base/2.tar.gz
mkdir ${MODEL_DIR}
tar -xvzf 2.tar.gz --directory=${MODEL_DIR}
# Converting weights to TF 2.0
python converter.py --tf_hub_path=${MODEL_DIR}/ --model_type=albert_encoder --version=2 --model=base
# Copy albert_config.json to config.json
cp ${MODEL_DIR}/assets/albert_config.json ${MODEL_DIR}/config.json
# Rename assets to vocab
mv ${MODEL_DIR}/assets/ ${MODEL_DIR}/vocab
# Delete unwanted files
rm -rf ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/variables/ ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/tfhub_module.pb
```

### Note 

In the released TF-HUB weights sentence order prediction weights are not included. FineTuning the model on domain specific data has to learn those weights from random initialization.