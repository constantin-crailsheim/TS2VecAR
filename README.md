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

Dataset|AR (s=0)|AR (s=0.1)|AR (s=0.2)|Replicated|Type
---|---|---|---|---|---
SelfRegulationSCP2|0.5444444444444444|0.55|0.55|0.5555555555555556|EEG
StandWalkJump|0.5333333333333333|0.4666666666666667|0.4|0.4666666666666667|ECG
SpokenArabicDigits|0.9672578444747613|0.9590723055934516|0.9768076398362893|0.9890859481582538|Speech
DuckDuckGeese|0.46|0.46|0.54|0.52|Audio
ArticularyWordRecognition|0.9866666666666667|0.97|0.9766666666666667|0.9766666666666667|Motion
CharacterTrajectories|0.9909470752089137|0.9930362116991643|0.9937325905292479|0.9923398328690808|Motion
EigenWorms|0.7862595419847328|0.7557251908396947|0.8091603053435115|0.8625954198473282|Motion
PenDigits|0.9877072612921669|0.9859919954259577|0.9899942824471126|0.9885648942252716|Motion
Handwriting|0.5564705882352942|0.5505882352941176|0.548235294117647|0.5305882352941177|HAR
NATOPS|0.9111111111111111|0.8388888888888889|0.8777777777777778|0.9388888888888889|HAR
RacketSports|0.8881578947368421|0.8881578947368421|0.8947368421052632|0.8552631578947368|HAR
UWaveGestureLibrary|0.91875|0.925|0.91875|0.90625|HAR
Mean (all datasets)|0.7942588134573555|0.7785939490953987|0.7896551165686262|0.7985387721722139|
Mean (HAR datasets)|0.8186223985208119|0.8006587547299622|0.809874978500172|0.8077475705194358|
Mean (HAR datasets)|0.8186223985208119|0.8006587547299622|0.809874978500172|0.8077475705194358|
Mean (HAR datasets)|0.8186223985208119|0.8006587547299622|0.809874978500172|0.8077475705194358|


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
