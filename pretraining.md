## Pretraining ALBERT Model

### Input Data Format

Documents or Paragraphs should be split by a blank line. Each line contains one sentence from Document/paragraph. The sentence order should be the same as in the original Document. This is important for the SOP task.

#### Sample Data
```
This text is included to make sure Unicode is handled properly: 力加勝北区ᴵᴺᵀᵃছজটডণত
Text should be one-sentence-per-line, with empty lines between documents.
This sample text is public domain and was randomly selected from Project Guttenberg.

The rain had only ceased with the gray streaks of morning at Blazing Star, and the settlement awoke to a moral sense of cleanliness, and the finding of forgotten knives, tin cups, and smaller camp utensils, where the heavy showers had washed away the debris and dust heaps before the cabin doors.
Indeed, it was recorded in Blazing Star that a fortunate early riser had once picked up on the highway a solid chunk of gold quartz which the rain had freed from its incumbering soil, and washed into immediate and glittering popularity.
Possibly this may have been the reason why early risers in that locality, during the rainy season, adopted a thoughtful habit of body, and seldom lifted their eyes to the rifted or india-ink washed skies above them.
"Cass" Beard had risen early that morning, but not with a view to discovery.
A leak in his cabin roof,--quite consistent with his careless, improvident habits,--had roused him at 4 A. M., with a flooded "bunk" and wet blankets.
The chips from his wood pile refused to kindle a fire to dry his bed-clothes, and he had recourse to a more provident neighbor's to supply the deficiency.
This was nearly opposite.
Mr. Cassius crossed the highway, and stopped suddenly.
Something glittered in the nearest red pool before him.
Gold, surely!
But, wonderful to relate, not an irregular, shapeless fragment of crude ore, fresh from Nature's crucible, but a bit of jeweler's handicraft in the form of a plain gold ring.
Looking at it more attentively, he saw that it bore the inscription, "May to Cass."
Like most of his fellow gold-seekers, Cass was superstitious.

The fountain of classic wisdom, Hypatia herself.
As the ancient sage--the name is unimportant to a monk--pumped water nightly that he might study by day, so I, the guardian of cloaks and parasols, at the sacred doors of her lecture-room, imbibe celestial knowledge.
From my youth I felt in me a soul above the matter-entangled herd.
She revealed to me the glorious fact, that I am a spark of Divinity itself.
A fallen star, I am, sir!' continued he, pensively, stroking his lean stomach--'a fallen star!--fallen, if the dignity of philosophy will allow of the simile, among the hogs of the lower world--indeed, even into the hog-bucket itself. Well, after all, I will show you the way to the Archbishop's.
There is a philosophic pleasure in opening one's treasures to the modest young.
Perhaps you will assist me by carrying this basket of fruit?' And the little man jumped up, put his basket on Philammon's head, and trotted off up a neighbouring street.
Philammon followed, half contemptuous, half wondering at what this philosophy might be, which could feed the self-conceit of anything so abject as his ragged little apish guide;
but the novel roar and whirl of the street, the perpetual stream of busy faces, the line of curricles, palanquins, laden asses, camels, elephants, which met and passed him, and squeezed him up steps and into doorways, as they threaded their way through the great Moon-gate into the ample street beyond, drove everything from his mind but wondering curiosity, and a vague, helpless dread of that great living wilderness, more terrible than any dead wilderness of sand which he had left behind.
Already he longed for the repose, the silence of the Laura--for faces which knew him and smiled upon him; but it was too late to turn back now.
His guide held on for more than a mile up the great main street, crossed in the centre of the city, at right angles, by one equally magnificent, at each end of which, miles away, appeared, dim and distant over the heads of the living stream of passengers, the yellow sand-hills of the desert;
while at the end of the vista in front of them gleamed the blue harbour, through a network of countless masts.
At last they reached the quay at the opposite end of the street;
and there burst on Philammon's astonished eyes a vast semicircle of blue sea, ringed with palaces and towers.
He stopped involuntarily; and his little guide stopped also, and looked askance at the young monk, to watch the effect which that grand panorama should produce on him.
```

#### Tokenizer

If the model fine-tuning only on a domain-specific data using a pre-trained model from ALBERT, use the same tokenizer model/vocab file from the ALBERT model dir. 

Train a new sentence piece tokenizer using [sentencepiece](https://github.com/google/sentencepiece). 

## Domain Specific Fine-Tuning

### Setup Model for fine-tuning

```bash
export MODEL_DIR=base
wget https://storage.googleapis.com/tfhub-modules/google/albert_base/2.tar.gz
mkdir ${MODEL_DIR}
tar -xvzf 2.tar.gz --directory=${MODEL_DIR}
# Converting weights to TF 2.0
python converter.py --tf_hub_path=${MODEL_DIR}/ --model_type=albert --version=2 --model=base
# Copy albert_config.json to config.json
cp ${MODEL_DIR}/assets/albert_config.json ${MODEL_DIR}/config.json
# Rename assets to vocab
mv ${MODEL_DIR}/assets/ ${MODEL_DIR}/vocab
# Delete unwanted files
rm -rf ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/variables/ ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/tfhub_module.pb
```
Fine-tuning domain specific data on ALBERT-Base. 
NB: In the released TF-HUB weights sentence order prediction weights are not included. FineTuning the model on domain specific data has to learn those weights from random initialization.

### Creating Pretraining Data
`data/` folder contains *.txt files containes the above sample format data.

```bash
export DATA_DIR=data

export PROCESSED_DATA=processed_data
mkdir $PROCESSED_DATA

python create_pretraining_data.py --input_file=${DATA_DIR}/*.txt --output_file=${PROCESSED_DATA}/train.tf_record --spm_model_file=${MODEL_DIR}/vocab/30k-clean.model --meta_data_file_path=${PROCESSED_DATA}/train_meta_data

## tf_records and metadata files will stored in the `PROCESSED_DATA`

```

### Pretraining Model


```bash
export PRETRAINED_MODEL=pretrained_model
mkdir $PRETRAINED_MODEL

python run_pretraining.py --albert_config_file=${MODEL_DIR}/config.json --do_train --init_checkpoint=${MODEL_DIR}/tf2_model_full.h5 --input_files=${PROCESSED_DATA}/train.tf_record --meta_data_file_path=${PROCESSED_DATA}/train_meta_data --output_dir=${PRETRAINED_MODEL} --strategy_type=mirror --train_batch_size=128 --num_train_epochs=3
```
`num_train_steps = int(total_train_examples / train_batch_size) * num_train_epochs`

After training full model serlized will be based on tensorflow checkpoints, `core_model` weights, which is needed for finetuning will be saved keras model weights as `tf2_model.h5`. 


### Model training From sctrach
- Create tokenizer or use any on existing ALBERT spm model 
- Select model configs from `model_configs` folder. It contains all the 4 configs of ALBERT model. Or create new one,Follow the same format.
- No `--init_checkpoint` in `run_pretraining.py`
