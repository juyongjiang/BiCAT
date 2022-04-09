# BiCAT
This is our TensorFlow implementation for the paper: "**[BiCAT: Sequential Recommendation with Bidirectional Chronological Augmentation of Transformer](https://arxiv.org/abs/2112.06460)**".
Our code is implemented based on Tensorflow version of [SASRec](https://github.com/kang205/SASRec) and [ASReP](https://github.com/DyGRec/ASReP).

## Environment
* TensorFlow 1.12
* Python 3.6.*

## Datasets Prepare
**Benchmarks**: Amazon Review datasets Beauty, Movie Lens and Cell_Phones_and_Accessories. 
The data split is done in the `leave-one-out` setting. Make sure you download the datasets from the [link](https://jmcauley.ucsd.edu/data/amazon/). Please, use the `DataProcessing.py` under the `data/`, and make sure you change the DATASET variable value to your dataset name, then you run:

```
python DataProcessing.py
```

You will find the processed dataset in the directory with the name of your input dataset.

## Beauty
**1. Reversely Pre-training and Short Sequence Augmentation**

Pre-train the model and output 20 items for sequences with length <= 20.

```
python main.py \
       --dataset=Beauty \
       --train_dir=default \
       --lr=0.001 \
       --hidden_units=128 \
       --maxlen=100 \
       --dropout_rate=0.7 \
       --num_blocks=2 \
       --l2_emb=0.0 \
       --num_heads=4 \
       --evalnegsample 100 \
       --reversed 1 \
       --reversed_gen_num 20 \
       --M 20
```
**2. Next-Item Prediction with Reversed-Pre-Trained Model and Augmented dataset**

```
python main.py \
       --dataset=Beauty \
       --train_dir=default \
       --lr=0.001 \
       --hidden_units=128 \
       --maxlen=100 \
       --dropout_rate=0.7 \
       --num_blocks=2 \
       --l2_emb=0.0 \
       --num_heads=4 \
       --evalnegsample 100 \
       --reversed_pretrain 1 \
       --aug_traindata 15 \
       --M 18
```

## Cell_Phones_and_Accessories
**1. Reversely Pre-training and Short Sequence Augmentation**

Pre-train the model and output 20 items for sequences with length <= 20.

```
python main.py \
       --dataset=Cell_Phones_and_Accessories \
       --train_dir=default \
       --lr=0.001 \
       --hidden_units=32 \
       --maxlen=100 \
       --dropout_rate=0.5 \
       --num_blocks=2 \
       --l2_emb=0.0 \
       --num_heads=2 \
       --evalnegsample 100 \
       --reversed 1 \
       --reversed_gen_num 20 \
       --M 20

```
**2. Next-Item Prediction with Reversed-Pre-Trained Model and Augmented dataset**

```
python main.py \
       --dataset=Cell_Phones_and_Accessories \
       --train_dir=default \
       --lr=0.001 \
       --hidden_units=32 \
       --maxlen=100 \
       --dropout_rate=0.5 \
       --num_blocks=2 \
       --l2_emb=0.0 \
       --num_heads=2 \
       --evalnegsample 100 \
       --reversed_pretrain 1 \ 
       --aug_traindata 17 \
       --M 18
```

## Citation
```
@misc{jiang2021sequential,
      title={Sequential Recommendation with Bidirectional Chronological Augmentation of Transformer}, 
      author={Juyong Jiang and Yingtao Luo and Jae Boum Kim and Kai Zhang and Sunghun Kim},
      year={2021},
      eprint={2112.06460},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```