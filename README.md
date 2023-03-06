# TS2VecAR

This repo is the implemtation of the [TS2VecAR](https://github.com/constantin-crailsheim/TS2VecAR/tree/main/paper/text/ts2vecar.pdf) paper written in the Unsupervised Deep Learning Seminar at LMU Munich in the winter term 2022/23 under the supervision of [Dr. Mina Rezaei](https://www.slds.stat.uni-muenchen.de/people/minar/)

## Model structure

<p align="center">
<img src="paper/text/fig/model_setup.png" width="800" class="center">
</p>

## Experiments

### Hyperparameter

The recommended default hyperparameter are:

| Hyperparameter  | Value  |
|---|---|
| Total iterations | 600 |
| m  | 5  |
| \lambda  | 5  |
| k  | 10  |
| Context dimensions  | 100  |
| AR learning rate  | 3e-4  |
| AR weight decay  | 3e-4  |

### Datasets

The datasets used for the paper can be found at [UEA dataset](http://www.timeseriesclassification.com/dataset.php). They should be stored in `datasets/UEA/` such that they can be found in `datasets/UEA/<dataset_name>/<dataset_name>_*.arff`.

### Results

The results of the experiments for different values of the share of iterations which did not include the autoregressive temporal contrasting component are as following:

Dataset|AR (s=0)|AR (s=0.1)|AR (s=0.2)|Replicated|Type
---|---|---|---|---|---
SelfRegulationSCP2|0.544|0.550|0.550|<strong>0.556</strong>|EEG
StandWalkJump|<strong>0.533|0.467|0.400|0.467|ECG
SpokenArabicDigits|0.967|0.959|0.977|<strong>0.989</strong>|Speech
DuckDuckGeese|0.460|0.460|<strong>0.540</strong>|0.520|Audio
ArticularyWordRecognition|<strong>0.987</strong>|0.970|0.977|0.977|Motion
CharacterTrajectories|0.991|0.993|<strong>0.994</strong>|0.992|Motion
EigenWorms|0.786|0.756|0.809|<strong>0.863</strong>|Motion
PenDigits|0.988|0.986|<strong>0.990</strong>|0.989|Motion
Handwriting|<strong>0.556|0.551|0.548|0.531|HAR
NATOPS|0.911|0.839|0.878|<strong>0.939</strong>|HAR
RacketSports|0.888|0.888|<strong>0.895</strong>|0.855|HAR
UWaveGestureLibrary|0.919|<strong>0.925</strong>|0.919|0.906|HAR
Mean (all datasets)|0.794|0.779|0.790|<strong>0.799</strong>|
Mean (HAR datasets)|<strong>0.819</strong>|0.801|0.810|0.808|

# Setup

## Requirements

The recommended requirements can be installed with:

```(bash)
pip install -r requirements.txt
```

# Training

To train a model for a dataset use the following command, where further optional arguments can be added. 

```(bash)
python train.py <dataset> <run_name> --loader <loader_name>
```

To replicate the results of the paper run:

```(bash)
bash scripts/uea.sh
```


# Attribution

This implementation is mostly based on TS2Vec: https://github.com/yuezhihan/ts2vec.

The autoregressive model is based on TS-TCC: https://github.com/emadeldeen24/TS-TCC
