
# import scipy.io
# import os
import numpy as np
# import pickle
import pandas as pd
from sklearn import preprocessing, model_selection
from getFeatureScore import getFeatureScore, selectByVote


# perpare the labels
labels = pd.read_csv('REFERENCE-v3.csv', header=None)
labels.columns = ["recordName", "className"]
labels["classNum"] = np.nan
labels.at[labels.className == 'N', 'classNum'] = 0
labels.at[labels.className == 'A', 'classNum'] = 1
labels.at[labels.className == 'O', 'classNum'] = 2
labels.at[labels.className == '~', 'classNum'] = 3

# load the features
GenericFeatures = pd.read_csv('GenericFeatures.csv')
SemiGenericFeatures = pd.read_csv('SemiGenericFeatures_meanBeat.csv')
DataDerivenFeatures = pd.read_csv('DataDerivenFeatures.csv')
ModelBasedFeatures = pd.read_csv('ModelBasedFeatures.csv')


# replace any nan with zero
GenericFeatures = GenericFeatures.fillna(0)
SemiGenericFeatures = SemiGenericFeatures.fillna(0)
DataDerivenFeatures = DataDerivenFeatures.fillna(0)
ModelBasedFeatures = ModelBasedFeatures.fillna(0)


# remove constant features
GenericFeatures = GenericFeatures.drop(
    GenericFeatures.columns[GenericFeatures.nunique() <= 1], axis=1)
SemiGenericFeatures = SemiGenericFeatures.drop(
    SemiGenericFeatures.columns[SemiGenericFeatures.nunique() <= 1], axis=1)
DataDerivenFeatures = DataDerivenFeatures.drop(
    DataDerivenFeatures.columns[DataDerivenFeatures.nunique() <= 1], axis=1)
ModelBasedFeatures = ModelBasedFeatures.drop(
    ModelBasedFeatures.columns[ModelBasedFeatures.nunique() <= 1], axis=1)

# # concat all features
# allFeatures = pd.concat([GenericFeatures, SemiGenericFeatures,
#                         DataDerivenFeatures, ModelBasedFeatures], axis=1)


# the feature-selection MUST be included in cross-validation! see: https://www.nodalpoint.com/not-perform-feature-selection/

# train-validation split
GF_train, GF_val, SGF_train, SGF_val, DDF_train, DDF_val, MBF_train, MBF_val, labels_train, labels_val = model_selection.train_test_split(
    GenericFeatures, SemiGenericFeatures, DataDerivenFeatures, ModelBasedFeatures, labels, test_size=0.2, random_state=42)


# scaling the features (or standardization?)
GF_scalar = preprocessing.MinMaxScaler().fit(GF_train)
GF_train_sc = pd.DataFrame(GF_scalar.transform(
    GF_train), columns=GF_train.columns)

SGF_scalar = preprocessing.MinMaxScaler().fit(SGF_train)
SGF_train_sc = pd.DataFrame(SGF_scalar.transform(
    SGF_train), columns=SGF_train.columns)

DDF_scalar = preprocessing.MinMaxScaler().fit(DDF_train)
DDF_train_sc = pd.DataFrame(DDF_scalar.transform(
    DDF_train), columns=DDF_train.columns)

MBF_scalar = preprocessing.MinMaxScaler().fit(MBF_train)
MBF_train_sc = pd.DataFrame(MBF_scalar.transform(
    MBF_train), columns=MBF_train.columns)


# initial selection
methods = ['var_threshold', 'chi2_test', 'f_value', 'mutual_info']
GFnames_best100 = selectByVote(getFeatureScore(
    GF_train_sc, labels_train.classNum, methods=methods), 100)
SGFnames_best100 = selectByVote(getFeatureScore(
    SGF_train_sc, labels_train.classNum, methods=methods), 100)
DDFnames_best100 = selectByVote(getFeatureScore(
    DDF_train_sc, labels_train.classNum, methods=methods), 100)
MBFnames_best100 = selectByVote(getFeatureScore(
    MBF_train_sc, labels_train.classNum, methods=methods), 100)


features_best400 = pd.concat(
    [GF_train_sc[GFnames_best100], SGF_train_sc[SGFnames_best100],
     DDF_train_sc[DDFnames_best100], MBF_train_sc[MBFnames_best100]], axis=1)

methods = ['var_threshold', 'chi2_test', 'f_value', 'mutual_info', 'nca',
           'relieff', 'surf', 'surf*', 'multisurf', 'multisurf*']
fscores_best400 = getFeatureScore(
    GF_train_sc, labels_train.classNum, methods=methods)
allFeatures_best100 = selectByVote(fscores_best400, 100)

# =============================================================================
# Scenario 1: selecting among all the features

# =============================================================================
# =============================================================================
# Scenario 2: selecting from each type and ranking the selected features

# =============================================================================
# =============================================================================
# # Scenario 3: Classification results


# =============================================================================
