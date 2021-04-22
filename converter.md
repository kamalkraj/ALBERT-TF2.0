## TensorFLow-Hub to TF 2.0 weights converison


### TF-HUB weights

#### Download
| Verison 1                                                                              | Version 2                                                                              | TAR                                                                              |
|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [base](https://storage.googleapis.com/tfhub-modules/google/albert_base/1.tar.gz)       | [base](https://storage.googleapis.com/tfhub-modules/google/albert_base/2.tar.gz)       | [base](https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz)       |
| [large](https://storage.googleapis.com/tfhub-modules/google/albert_large/1.tar.gz)     | [large](https://storage.googleapis.com/tfhub-modules/google/albert_large/2.tar.gz)     | [large](https://storage.googleapis.com/albert_models/albert_large_v2.tar.gz)     |
| [xlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xlarge/1.tar.gz)   | [xlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xlarge/2.tar.gz)   | [xlarge](https://storage.googleapis.com/albert_models/albert_xlarge_v2.tar.gz)   |
| [xxlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xxlarge/1.tar.gz) | [xxlarge](https://storage.googleapis.com/tfhub-modules/google/albert_xxlarge/2.tar.gz) | [xxlarge](https://storage.googleapis.com/albert_models/albert_xxlarge_v2.tar.gz) |



### Converison flags
```
  --model: <base|large|xlarge|xxlarge>: model for converison
    (default: 'base')
  --model_type: <albert_encoder|albert>: Select model type for weight conversion.
    albert_enoder for finetuning tasks.
    albert for MLM & SOP FineTuning on domain specific data.
    (default: 'albert_encoder')
  --version: tf hub model version to convert 1 or 2.
    (default: '2')
    (an integer)
  --weights_path: weights_path for download models
  --weights_type: <tf_hub|tar>: model weight type tf_hub/tar
    (default: 'tf_hub')
```

### Converison Example from TF HUB

```bash
export MODEL_DIR=base
wget https://storage.googleapis.com/tfhub-modules/google/albert_base/2.tar.gz
mkdir ${MODEL_DIR}
tar -xvzf 2.tar.gz --directory=${MODEL_DIR}
# Converting weights to TF 2
python converter.py --weights_path=${MODEL_DIR}/ --weights_type=tf_hub --model_type=albert_encoder --version=2 --model=base
# Copy albert_config.json to config.json
cp ${MODEL_DIR}/assets/albert_config.json ${MODEL_DIR}/config.json
# Rename assets to vocab
mv ${MODEL_DIR}/assets/ ${MODEL_DIR}/vocab
# Delete unwanted files
rm -rf ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/variables/ ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/tfhub_module.pb
```

### Converison Example from TAR

```bash
export MODEL_DIR=base
wget https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz
mkdir ${MODEL_DIR}
tar -xvzf albert_base_v2.tar.gz --directory=${MODEL_DIR}
# Converting weights to TF 2
python converter.py --weights_path=${MODEL_DIR}/ --weights_type=tar --model_type=albert_encoder --version=2 --model=base
# Copy albert_config.json to config.json
cp ${MODEL_DIR}/albert_base/albert_config.json ${MODEL_DIR}/config.json
# move vocab file to vocab
mkdir ${MODEL_DIR}/vocab
mv ${MODEL_DIR}/albert_base/30k-clean.* ${MODEL_DIR}/vocab
# Delete unwanted files
rm -rf ${MODEL_DIR}/albert_base
```

### Note 

In the released TF-HUB weights sentence order prediction weights are not included. FineTuning the model on domain specific data has to learn those weights from random initialization.

For full weights conversion including sentence order prediction weights use converter from tar files.