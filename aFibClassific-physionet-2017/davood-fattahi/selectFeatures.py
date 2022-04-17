
# import scipy.io
# import os
import numpy as np
# import pickle
import pandas as pd
from sklearn import preprocessing, model_selection
from getFeatureScore import getFeatureScore, selectByVote
from sklearn.ensemble import RandomForestClassifier


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
# DataDerivenFeatures, ModelBasedFeatures], axis=1)


# the feature-selection MUST be included in cross-validation!
#   see: https://www.nodalpoint.com/not-perform-feature-selection/
# why we don not use pipeline from sklearn? 1- feature selection must be
#   included in CV, and the fs methods are not standard methods of sklearn.
#   2- we use two steps of feature selection. 3- we need to keep the results of
#   each fs step. 4- by the way, we can write a sklearn compatible module, but
#   it is time-consuming and not our priority for now.
cvtimes = 5
for cvturn in range(1, cvtimes+1):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=(cvturn))
    for train_index, val_index in kf.split(labels):
        # train-validation split
        GF_train, SGF_train, DDF_train, MBF_train, labels_train = GenericFeatures.iloc[train_index], SemiGenericFeatures.iloc[
            train_index], DataDerivenFeatures.iloc[train_index], ModelBasedFeatures.iloc[train_index], labels.iloc[train_index]

        GF_val, SGF_val, DDF_val, MBF_val, labels_val = GenericFeatures.iloc[val_index], SemiGenericFeatures.iloc[
            val_index], DataDerivenFeatures.iloc[val_index], ModelBasedFeatures.iloc[val_index], labels.iloc[val_index]

        # scaling the features (or standardization?)
        GF_scalar = preprocessing.MinMaxScaler().fit(GF_train)
        GF_train = pd.DataFrame(GF_scalar.transform(
            GF_train), columns=GF_train.columns)
        GF_val = pd.DataFrame(GF_scalar.transform(
            GF_val), columns=GF_val.columns)

        SGF_scalar = preprocessing.MinMaxScaler().fit(SGF_train)
        SGF_train = pd.DataFrame(SGF_scalar.transform(
            SGF_train), columns=SGF_train.columns)
        SGF_val = pd.DataFrame(SGF_scalar.transform(
            SGF_val), columns=SGF_val.columns)

        DDF_scalar = preprocessing.MinMaxScaler().fit(DDF_train)
        DDF_train = pd.DataFrame(DDF_scalar.transform(
            DDF_train), columns=DDF_train.columns)
        DDF_val = pd.DataFrame(DDF_scalar.transform(
            DDF_val), columns=DDF_val.columns)

        MBF_scalar = preprocessing.MinMaxScaler().fit(MBF_train)
        MBF_train = pd.DataFrame(MBF_scalar.transform(
            MBF_train), columns=MBF_train.columns)
        MBF_val = pd.DataFrame(MBF_scalar.transform(
            MBF_val), columns=MBF_val.columns)

        # initial selection
        methods = ['var_threshold', 'chi2_test', 'f_value', 'mutual_info']
        GFnames_initBest = selectByVote(getFeatureScore(
            GF_train, labels_train.classNum, methods=methods), 150)
        SGFnames_initBest = selectByVote(getFeatureScore(
            SGF_train, labels_train.classNum, methods=methods), 150)
        DDFnames_initBest = selectByVote(getFeatureScore(
            DDF_train, labels_train.classNum, methods=methods), 150)
        MBFnames_initBest = selectByVote(getFeatureScore(
            MBF_train, labels_train.classNum, methods=methods), 150)

        allF_train_initBest = pd.concat(
            [GF_train[GFnames_initBest], SGF_train[SGFnames_initBest],
             DDF_train[DDFnames_initBest], MBF_train[MBFnames_initBest]], axis=1)
        scalar = preprocessing.MinMaxScaler().fit(allF_train_initBest)
        allF_train_initBest = pd.DataFrame(GF_scalar.transform(
            allF_train_initBest), columns=allF_train_initBest.columns)
        methods = ['var_threshold', 'chi2_test', 'f_value', 'mutual_info', 'nca',
                   'relieff', 'surf', 'surf*', 'multisurf', 'multisurf*']
        fscores_finalBest = getFeatureScore(
            allF_train_initBest, labels_train.classNum, methods=methods)

        clf = RandomForestClassifier(max_depth=2, random_state=0)
        for n in range(4*150):
            allFnames_finalBest = selectByVote(fscores_finalBest, n)
            clf.fit(
                features_finalBest[allFnames_finalBest], labels_train.classNum)


# =============================================================================
# Scenario 1: selecting among all the features

# =============================================================================
# =============================================================================
# Scenario 2: selecting from each type and ranking the selected features

# =============================================================================
# =============================================================================
# # Scenario 3: Classification results


# =============================================================================
