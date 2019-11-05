# ALBERT-TF2.0
ALBERT model Fine Tuning using TF2.0 [WIP][90%]

`warning` 🐞🐞🐞

## Requirements
- python3
- pip install -r requirements.txt

## Download ALBERT TF 2.0 weights

#### Verison 1

- [base](https://drive.google.com/open?id=1WDz1193fEo8vROpi-hWn3hveMmddLjpy)
- [large](https://drive.google.com/open?id=1j4ePHivAXHNqqNucZOocwlkyneQyUROl)
- [xlarge](https://drive.google.com/open?id=10o7l7c7Y5UlkSQmFca0_iaRsGIPmJ5Ya)
- [xxlarge](https://drive.google.com/open?id=1gl5lOiAHq29C_sG6GoXLeZJHKDD2Gfju)

#### Version 2

- [base](https://drive.google.com/open?id=1FkrvdQnJR9za9Pv8cuiEXd1EI2hxx31a)
- [large](https://drive.google.com/open?id=1xADTTjwTogFmnhNU3EPJ86slykoSL4L7)
- [xlarge](https://drive.google.com/open?id=1GsAU_RqO8Pl7oPecj0opjA-4ktI8-4oX)
- [xxlarge](https://drive.google.com/open?id=1JtQcGKtt0QZThXS1jz2v5x72TrYYjg8N)

unzip the model inside repo.

Above weights does not contain the final layer in original model. Now can only be used for fine tuning downstream tasks.


## Download glue data
Download using the below cmd

```bash
python download_glue_data.py --data_dir glue_data --tasks all
```

## Fine-tuning
To prepare the fine-tuning data for final model training, use the
[`create_finetuning_data.py`](./create_finetuning_data.py) script.  Resulting
datasets in `tf_record` format and training meta data should be later passed to
training or evaluation scripts. The task-specific arguments are described in
following sections:

### Creating finetuninig data
* Example CoLA

```bash
export GLUE_DIR=glue_data/
export ALBERT_DIR=large/

export TASK_NAME=CoLA
export OUTPUT_DIR=cola_processed
mkdir $OUTPUT_DIR

python create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/ \
 --spm_model_file=${ALBERT_DIR}/vocab/30k-clean.model \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME}
```

### Running classifier

```bash
export MODEL_DIR=CoLA_OUT
python run_classifer.py \
--train_data_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
--eval_data_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
--input_meta_data_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
--albert_config_file=${ALBERT_DIR}/config.json \
--task_name=${TASK_NAME} \
--spm_model_file=${ALBERT_DIR}/vocab/30k-clean.model \
--output_dir=${MODEL_DIR} \
--init_checkpoint=${ALBERT_DIR}/tf2_model.h5 \
--do_train \
--do_eval \
--train_batch_size=16 \
--learning_rate=1e-5
```

By default run_classifier will run 3 epochs. and evaluate on development set

Above cmd would result in dev set `accuracy` of `76.22` in CoLA task

The above code tested on TITAN RTX 24GB single GPU

### Ignore
Below warning will be there at end of each epoch. Issue with training steps calcuation when `tf.data` provided to `model.fit()`
Have no effect on model performance so ignore. Mostly will fixed in the next tf2 relase . [Issue-link](https://github.com/tensorflow/tensorflow/issues/25254)
```
2019-10-31 13:35:48.322897: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Out of range:
End of sequence
         [[{{node IteratorGetNext}}]]
         [[model_1/albert_model/word_embeddings/Shape/_10]]
2019-10-31 13:36:03.302722: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Out of range:
End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_4]]
```


### SQuAD

#### Training Data Preparation
```bash
export SQUAD_DIR=SQuAD
export SQUAD_VERSION=v1.1
export ALBERT_DIR=large
export OUTPUT_DIR=squad_out_${SQUAD_VERSION}
mkdir $OUTPUT_DIR

python create_finetuning_data.py \
--squad_data_file=${SQUAD_DIR}/train-${SQUAD_VERSION}.json \
--spm_model_file=${ALBERT_DIR}/vocab/30k-clean.model  \
--train_data_output_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_train.tf_record  \
--meta_data_file_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_meta_data \
--fine_tuning_task_type=squad \
--max_seq_length=384
```

### Running Model
```bash

python run_squad.py \
--mode=train_and_predict \
--input_meta_data_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_meta_data \
--train_data_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
--predict_file=${SQUAD_DIR}/dev-${SQUAD_VERSION}.json \
--albert_config_file=${ALBERT_DIR}/config.json \
--init_checkpoint=${ALBERT_DIR}/tf2_model.h5 \
--spm_model_file=${ALBERT_DIR}/vocab/30k-clean.model \
--train_batch_size=48 \
--predict_batch_size=48 \
--learning_rate=1e-5 \
--num_train_epochs=3 \
--model_dir=${OUTPUT_DIR} \
--strategy_type=mirror
```



### Multi-GPU training

Use flag `--strategy_type=mirror` for Multi GPU training. Currently All the exsisting GPUs in the enviorment will be used.



## References

lots of code in this repo are adpted from multiple repos. No reference are added now. Will add evevrything.
