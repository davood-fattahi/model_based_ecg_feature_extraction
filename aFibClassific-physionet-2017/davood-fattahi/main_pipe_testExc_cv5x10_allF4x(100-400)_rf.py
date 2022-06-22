"""
Created on Sun May 29 18:06:19 2022

@author: Davood Fattahi
"""


if __name__ == '__main__':

    # import scipy.io
    import os
    import numpy as np
    import pickle
    import pandas as pd
    from sklearn import preprocessing, model_selection
    from getFeatureScore import getFeatureScore, selectByVote
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
    import time
    import datetime
    import dill
    import matplotlib.pyplot as plt

    def CinC2017f1score(true, pred):
        f = 0
        return f

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_is_fitted
    # The ColumnsSelector class inherits from the sklearn.base classes
    # (BaseEstimator, TransformerMixin). This makes it compatible with
    # scikit-learn’s Pipelines

    class MyStandardScaler(BaseEstimator, TransformerMixin):
        def __init__(self):
            return None

        def fit(self, X, y=None):
            print(type(X))
            # the type of X might be a DataFrame or a NumPy array
            # depending on the previous transformer object that
            # you use in the pipeline
            self.means = np.mean(X, axis=0)    # calculate the mean
            # calculate the standard deviation
            self.stds = np.std(X, axis=0)
            return self

        def transform(self, X, y=None):
            return (X - self.means) / self.stds
# %%
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
    DataDrivenFeatures = pd.read_csv('DataDerivenFeatures.csv')
    ModelBasedFeatures = pd.read_csv('ModelBasedFeatures.csv')

    # Split data to train and test
    ttspliter = model_selection.StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=0)
    ttspliter = model_selection.StratifiedKFold(
        n_splits=4, shuffle=True, random_state=(0))
    train_index, test_index = ttspliter.split(
        np.zeros(labels.shape[0]), labels.classNum)

    # replace any nan with zero
    GenericFeatures = GenericFeatures.fillna(0)
    SemiGenericFeatures = SemiGenericFeatures.fillna(0)
    DataDrivenFeatures = DataDrivenFeatures.fillna(0)
    ModelBasedFeatures = ModelBasedFeatures.fillna(0)

    # remove constant features
    GenericFeatures = GenericFeatures.drop(
        GenericFeatures.columns[GenericFeatures.nunique() <= 1], axis=1)
    SemiGenericFeatures = SemiGenericFeatures.drop(
        SemiGenericFeatures.columns[SemiGenericFeatures.nunique() <= 1], axis=1)
    DataDrivenFeatures = DataDrivenFeatures.drop(
        DataDrivenFeatures.columns[DataDrivenFeatures.nunique() <= 1], axis=1)
    ModelBasedFeatures = ModelBasedFeatures.drop(
        ModelBasedFeatures.columns[ModelBasedFeatures.nunique() <= 1], axis=1)

# %%
    """ the feature-selection MUST be included in cross-validation!
       see: https://www.nodalpoint.com/not-perform-feature-selection/
     why we don not use pipeline from sklearn? 1- feature selection must be
       included in CV, and the fs methods are not standard methods of sklearn.
    #  2- we use two steps of feature selection. 3- we need to keep the results of
       each fs step. 4- by the way, we can write a sklearn compatible module, but
       it is time-consuming and not our priority for now."""

    scalar_GenericFeatures_train = []
    scalar_SemiGenericFeatures_train = []
    scalar_DataDrivenFeatures_train = []
    scalar_ModelBasedFeatures_train = []
    score_GenericFeatures = []
    score_SemiGenericFeatures = []
    score_DataDrivenFeatures = []
    score_ModelBasedFeatures = []
    valClaasifResult = []
    CLFs = []
    kf = []

    # iterables:
    cvtimes = 5
    cvk = 10
    n_features = [100, 200, 300, 400]  # 4 rounds
    n_estimators = range(200, 1400, 200)  # 6 rounds

    # CV loops
    for cvturn in range(cvtimes):
        kf.append(model_selection.StratifiedKFold(
            n_splits=cvk, shuffle=True, random_state=(cvturn+1)))
        for train_index, val_index in kf[-1].split(np.zeros(labels.shape[0]), labels.classNum):
            # train-validation split
            GenericFeatures_train, SemiGenericFeatures_train, DataDrivenFeatures_train, ModelBasedFeatures_train,  labels_train = GenericFeatures.iloc[
                train_index], SemiGenericFeatures.iloc[train_index], DataDrivenFeatures.iloc[train_index], ModelBasedFeatures.iloc[train_index], labels.iloc[train_index]
            GenericFeatures_val, SemiGenericFeatures_val, DataDrivenFeatures_val, ModelBasedFeatures_val,  labels_val = GenericFeatures.iloc[
                val_index], SemiGenericFeatures.iloc[val_index], DataDrivenFeatures.iloc[val_index], ModelBasedFeatures.iloc[val_index], labels.iloc[val_index]

            # scaling the features (or standardization?)
            scalar_GenericFeatures_train.append(
                preprocessing.StandardScaler().fit(GenericFeatures_train))
            GenericFeatures_train = pd.DataFrame(scalar_GenericFeatures_train[-1].transform(
                GenericFeatures_train), columns=GenericFeatures_train.columns)
            GenericFeatures_val = pd.DataFrame(scalar_GenericFeatures_train[-1].transform(
                GenericFeatures_val), columns=GenericFeatures_val.columns)

            scalar_SemiGenericFeatures_train.append(preprocessing.StandardScaler().fit(
                SemiGenericFeatures_train))
            SemiGenericFeatures_train = pd.DataFrame(scalar_SemiGenericFeatures_train[-1].transform(
                SemiGenericFeatures_train), columns=SemiGenericFeatures_train.columns)
            SemiGenericFeatures_val = pd.DataFrame(scalar_SemiGenericFeatures_train[-1].transform(
                SemiGenericFeatures_val), columns=SemiGenericFeatures_val.columns)

            scalar_DataDrivenFeatures_train.append(preprocessing.StandardScaler().fit(
                DataDrivenFeatures_train))
            DataDrivenFeatures_train = pd.DataFrame(scalar_DataDrivenFeatures_train[-1].transform(
                DataDrivenFeatures_train), columns=DataDrivenFeatures_train.columns)
            DataDrivenFeatures_val = pd.DataFrame(scalar_DataDrivenFeatures_train[-1].transform(
                DataDrivenFeatures_val), columns=DataDrivenFeatures_val.columns)

            scalar_ModelBasedFeatures_train.append(preprocessing.StandardScaler().fit(
                ModelBasedFeatures_train))
            ModelBasedFeatures_train = pd.DataFrame(scalar_ModelBasedFeatures_train[-1].transform(
                ModelBasedFeatures_train), columns=ModelBasedFeatures_train.columns)
            ModelBasedFeatures_val = pd.DataFrame(scalar_ModelBasedFeatures_train[-1].transform(
                ModelBasedFeatures_val), columns=ModelBasedFeatures_val.columns)

            # initial feature selection
            methods = ['var_threshold', 'chi2_test', 'f_value', 'mutual_info']

            score_GenericFeatures.append(getFeatureScore(
                GenericFeatures_train, labels_train.classNum, methods=methods))
            score_SemiGenericFeatures.append(getFeatureScore(
                SemiGenericFeatures_train, labels_train.classNum, methods=methods))
            score_DataDrivenFeatures.append(getFeatureScore(
                DataDrivenFeatures_train, labels_train.classNum, methods=methods))
            score_ModelBasedFeatures.append(getFeatureScore(
                ModelBasedFeatures_train, labels_train.classNum, methods=methods))

            for n_f in n_features:
                iniSlctFeat_train = pd.concat([GenericFeatures_train[selectByVote(score_GenericFeatures[-1], n_f)], SemiGenericFeatures_train[selectByVote(score_SemiGenericFeatures[-1], n_f)],
                                               DataDrivenFeatures_train[selectByVote(score_DataDrivenFeatures[-1], n_f)], ModelBasedFeatures_train[selectByVote(score_ModelBasedFeatures[-1], n_f)]], axis=1)
                iniSlctFeat_val = pd.concat([GenericFeatures_val[selectByVote(score_GenericFeatures[-1], n_f)], SemiGenericFeatures_val[selectByVote(score_SemiGenericFeatures[-1], n_f)],
                                            DataDrivenFeatures_val[selectByVote(score_DataDrivenFeatures[-1], n_f)], ModelBasedFeatures_val[selectByVote(score_ModelBasedFeatures[-1], n_f)]], axis=1)
                # classifier learning
                for n_est in n_estimators:
                    s = time.time()
                    clf = RandomForestClassifier(
                        n_estimators=n_est, criterion="entropy", random_state=0)
                    clf.fit(iniSlctFeat_train, labels_train.classNum)
                    print("classifier learning time:", time.time()-s)

                    # predict the validation labels
                    pred = clf.predict(iniSlctFeat_val)

                    # scoring save the results
                    f1Sc = f1_score(labels_val.classNum, pred, average='macro')
                    sucRate = f1_score(labels_val.classNum,
                                       pred, average='micro')
                    mf1Sc = CinC2017f1score(labels_val.classNum, pred)
                    valClaasifResult.append(pd.DataFrame(
                        data={'predict': pred, 'label': labels_val.classNum,
                              'success': sucRate, 'f1_score': f1Sc, 'modified f1_score': mf1Sc}))
                    CLFs.append(clf)

        # x = datetime.datetime.now()
        # fileName = os.path.basename(__file__)[:-3]+'tempSave_'+x.strftime("%Y%m%d%H%M")+'.pkl'
            fileName = os.path.basename(__file__)[:-3]+'_tempSave_2.pkl'
            dill.dump_session(fileName)
        # to restore session:
        # dill.load_session(fileName)

    # ClassifRate = np.zeros([25, 1])
    # ClassifRep = np.zeros([25, 1])
    # for subCV in range(25):
    #     ClassifRate[subCV] = valClaasifResult[subCV].success.sum(
    #     )/valClaasifResult[subCV].success.size
    #     ClassifRep[subCV] = f1_score(
    #         valClaasifResult[subCV].label, valClaasifResult[subCV].predict, average='macro')


# plt.plot(range(10, 600, 10), 1-np.mean(ClassifRate, axis=0))
