## 33rd place solution code for Kaggle [TGS Salt Identification Chanllenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)

This example shows how to enable AutoML for competition code by running it on NNI without any code change.
To run this code on NNI, firstly you need to run it standalone, then configure the config.yml and:
```
nnictl create --config config.yml
```

This code can still run standalone, the code is for reference, it requires at least one week effort to reproduce the competition result.

[Solution summary](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69593)

Preparation:

Download competition data, run preprocess.py to prepare training data.

Stage 1:

Train fold 0-3 for 100 epochs, for each fold, train 3 models:
```
python3 train.py --ifolds 0 --epochs 100 --model_name UNetResNetV4
python3 train.py --ifolds 0 --epochs 100 --model_name UNetResNetV5 --layers 50
python3 train.py --ifolds 0 --epochs 100 --model_name UNetResNetV6
```

Stage 2:

Fine tune stage 1 models for 300 epochs with cosine annealing lr scheduler:

```
python3 train.py --ifolds 0 --epochs 300 --lrs cosine --lr 0.001 --min_lr 0.0001 --model_name UNetResNetV4
```

Stage 3:

Fine tune Stage 2 models with depths channel:

```
python3 train.py --ifolds 0 --epochs 300 --lrs cosine --lr 0.001 --min_lr 0.0001 --model_name UNetResNetV4 --depths
```

Stage 4:

Make prediction for each model,  then ensemble the result to generate peasdo labels.

Stage 5:

Fine tune stage 3 models with pseudo labels

```
python3 train.py --ifolds 0 --epochs 300 --lrs cosine --lr 0.001 --min_lr 0.0001 --model_name UNetResNetV4 --depths --pseudo
```

Stage 6:
Ensemble all stage 3 and stage 5 models.

