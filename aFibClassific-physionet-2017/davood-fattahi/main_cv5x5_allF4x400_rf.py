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
    cvtimes = 5
    cvk = 5
    n_features = 400
    n_Features = 400
    n_estimators = 500
    for cvturn in range(cvtimes):
        kf.append(model_selection.KFold(
            n_splits=cvk, shuffle=True, random_state=(cvturn+1)))
        for train_index, val_index in kf[-1].split(labels):
            # train-validation split
            GenericFeatures_train, SemiGenericFeatures_train, DataDrivenFeatures_train, ModelBasedFeatures_train,  labels_train = GenericFeatures.iloc[
                train_index], SemiGenericFeatures.iloc[train_index], DataDrivenFeatures.iloc[train_index], ModelBasedFeatures.iloc[train_index], labels.iloc[train_index]
            GenericFeatures_val, SemiGenericFeatures_val, DataDrivenFeatures_val, ModelBasedFeatures_val,  labels_val = GenericFeatures.iloc[
                val_index], SemiGenericFeatures.iloc[val_index], DataDrivenFeatures.iloc[val_index], ModelBasedFeatures.iloc[val_index], labels.iloc[val_index]

            # scaling the features (or standardization?)
            scalar_GenericFeatures_train.append(
                preprocessing.MinMaxScaler().fit(GenericFeatures_train))
            GenericFeatures_train = pd.DataFrame(scalar_GenericFeatures_train[-1].transform(
                GenericFeatures_train), columns=GenericFeatures_train.columns)
            GenericFeatures_val = pd.DataFrame(scalar_GenericFeatures_train[-1].transform(
                GenericFeatures_val), columns=GenericFeatures_val.columns)

            scalar_SemiGenericFeatures_train.append(preprocessing.MinMaxScaler().fit(
                SemiGenericFeatures_train))
            SemiGenericFeatures_train = pd.DataFrame(scalar_SemiGenericFeatures_train[-1].transform(
                SemiGenericFeatures_train), columns=SemiGenericFeatures_train.columns)
            SemiGenericFeatures_val = pd.DataFrame(scalar_SemiGenericFeatures_train[-1].transform(
                SemiGenericFeatures_val), columns=SemiGenericFeatures_val.columns)

            scalar_DataDrivenFeatures_train.append(preprocessing.MinMaxScaler().fit(
                DataDrivenFeatures_train))
            DataDrivenFeatures_train = pd.DataFrame(scalar_DataDrivenFeatures_train[-1].transform(
                DataDrivenFeatures_train), columns=DataDrivenFeatures_train.columns)
            DataDrivenFeatures_val = pd.DataFrame(scalar_DataDrivenFeatures_train[-1].transform(
                DataDrivenFeatures_val), columns=DataDrivenFeatures_val.columns)

            scalar_ModelBasedFeatures_train.append(preprocessing.MinMaxScaler().fit(
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

            iniSlctFeat_train = pd.concat([GenericFeatures_train[selectByVote(score_GenericFeatures[-1], n_features)], SemiGenericFeatures_train[selectByVote(score_SemiGenericFeatures[-1], n_features)],
                                           DataDrivenFeatures_train[selectByVote(score_DataDrivenFeatures[-1], n_features)], ModelBasedFeatures_train[selectByVote(score_ModelBasedFeatures[-1], n_features)]], axis=1)
            iniSlctFeat_val = pd.concat([GenericFeatures_val[selectByVote(score_GenericFeatures[-1], n_features)], SemiGenericFeatures_val[selectByVote(score_SemiGenericFeatures[-1], n_features)],
                                        DataDrivenFeatures_val[selectByVote(score_DataDrivenFeatures[-1], n_features)], ModelBasedFeatures_val[selectByVote(score_ModelBasedFeatures[-1], n_features)]], axis=1)
            # classifier learning
            s = time.time()
            clf = RandomForestClassifier(
                n_estimators=n_estimators, criterion="entropy", random_state=0)
            clf.fit(iniSlctFeat_train, labels_train.classNum)
            print("classifier learning time:", time.time()-s)

            # save the results
            valClaasifResult.append(pd.DataFrame(data={'predict': clf.predict(
                iniSlctFeat_val), 'label': labels_val.classNum, 'success': clf.predict(iniSlctFeat_val) == labels_val.classNum}))
            CLFs.append(clf)

            # x = datetime.datetime.now()
            # fileName = os.path.basename(__file__)[:-3]+'tempSave_'+x.strftime("%Y%m%d%H%M")+'.pkl'
            fileName = os.path.basename(__file__)[:-3]+'_tempSave_1.pkl'
            dill.dump_session(fileName)
            # to restore session:
            # dill.load_session(fileName)

            # f = open(fileName, 'wb')
            # pickle.dump([scalar_GenericFeatures_train, scalar_SemiGenericFeatures_train, scalar_DataDrivenFeatures_train, scalar_ModelBasedFeatures_train,
            #             score_GenericFeatures, score_SemiGenericFeatures, score_DataDrivenFeatures, score_ModelBasedFeatures, valClaasifResult, CLFs, kf], f, -1)
            # f.close()


# f = open('tempSave.pkl', 'rb')
# allFscore, valClaasifResult, CLFs = pickle.load(f)

    ClassifRate = np.zeros([25, 1])
    ClassifRep = np.zeros([25, 1])
    for subCV in range(25):
        ClassifRate[subCV] = valClaasifResult[subCV].success.sum(
        )/valClaasifResult[subCV].success.size
        ClassifRep[subCV] = f1_score(
            valClaasifResult[subCV].label, valClaasifResult[subCV].predict, average='macro')


# plt.plot(range(10, 600, 10), 1-np.mean(ClassifRate, axis=0))
