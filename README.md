# ALBERT-TF2.0
ALBERT model Fine Tuning using TF2.0 [WIP] 

`warning` 🐞🐞🐞

## Requirements
- python3
- pip install -r requirements.txt

## Download ALBERT TF 2.0 weights

- [base](https://drive.google.com/open?id=1WDz1193fEo8vROpi-hWn3hveMmddLjpy)
- [large](https://drive.google.com/open?id=1j4ePHivAXHNqqNucZOocwlkyneQyUROl)
- [xlarge](https://drive.google.com/open?id=10o7l7c7Y5UlkSQmFca0_iaRsGIPmJ5Ya)
- [xxlarge](https://drive.google.com/open?id=1gl5lOiAHq29C_sG6GoXLeZJHKDD2Gfju)

unzip the model inside repo.

Above weights does not contain the final layer in orginal model. Now can only be used for fine tuning downstream tasks. 

Above weights are converted from `tf_hub version 1`
 checkpoints. converted weights are tested with `tf_hub` module and produces idential results.


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
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --spm_model_file=${ALBERT_DIR}/vocab/30k-clean.model \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME}
```

### Running classifier

```bash
python run_classifer.py \
--train_data_path=cola_processed/CoLA_train.tf_record \
--eval_data_path=cola_processed/CoLA_eval.tf_record \
--input_meta_data_path=cola_processed/CoLA_meta_data \
--albert_config_file=large/config.json \
--task_name=CoLA \
--spm_model_file=large/vocab/30k-clean.model \
--output_dir=CoLA_OUT \
--init_checkpoint=large/tf2_model.h5 \
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

- WIP


### Multi-GPU training 

- WIP

Not Enabled. Currently all the model will run only in single gpu. Adjust max_seq_length and batch size according to your gpu capacity.

### More Examples 

- WIP

## References

lots of code in this repo are adpted from multiple repos. No reference are added now. Will add evevrything.
