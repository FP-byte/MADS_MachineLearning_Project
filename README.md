## 3. General info
Student: Francesca Paulin
Repo: https://github.com/FP-byte/MADS_EXAM-25_FP

## The data
### Arrhythmia Dataset

- Number of Samples: 109446
- Number of Categories: 5
- Sampling Frequency: 125Hz
- Data Source: Physionet's MIT-BIH Arrhythmia Dataset
- Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_big_train.parq` and `data/heart_big_test.parq`.

Due to imbalance of the train dataset, the dataset has been rebalanced in two ways.
The process of resampling is described in the notebook `notebooks/05_Resampling.ipyb`

### Oversampling traing dataset oversampling
Semi balanced dataset is obtain by upsampeling the minority classes to the majority class, still keeping the unbalanced distribution.
The distribution is as follows:
0.0    72471
4.0    30938
2.0    27743
1.0    10761
3.0     3029

Fully balanced dataset is composed as follows:
Downsampling: 75% of the majority class (55.000) 
Upsampling: all the majority class have been upsampled to 55.000 samples

In both cases the upsampling did not introduce new data but just a repetition of samples.
The semi-sampled has been used for few training rounds and then replaced by the fully oversampled one.


### SMOTE dataset
Smote dataset consist of syntetic data obtained by creating new instances that are interpolated between existing instances of the minority class. It uses a k-nearest neighbors (k-NN) approach to find similar points and generates synthetic examples that are between those points.
Dataset contains 72.471 samples for each class.

The traing of the models has been done on the original dataset and all three datasets.


## Exploration and hypothesis
In `notebooks/00_journal_explore-heart.ipynb` describes all the explorations of models and (manual) hypertunig steps.

## Models
There are two notebooks with for manually hypertuning models, one for 1D models and one for 2D architectures. 

# Reporting
Please find the report in `report/MADS_report_FP.pdf`

# 1. Explore
My explorations (see src/models.py) incuded:
- CNN 2D + ResNet
- Transformer with CNN 2D
- Transformer + CNN 1D + Resnet
- Transformer + CNN 1D + Resnet + Squeeze and Excite (SE)
- Transformer + CNN 2D + ResNet
- Transformer + CNN 2D + ResNet + Squeeze and Excite (SE)
- CNN1D + ResNet
- CNN1D GRU + ResNet
- CNN1D + GRU+ ResNet + Multihead Attention

## 2 Hypertune

Hypertuning of hybrid model 1D CNN + GRU: `hypertune_1DCNNGRU.py` file to hypertune the model with Ray. 
Hypertuning of hybrid model 2D CNN + Resnet: `hypertune_2DCNNResnet.py` file to hypertune the model with Ray. 
The files make use of basis settings from the class Hypertuner (hypertuner.py)


