"""
Created on Sun May 29 18:06:19 2022

@author: Davood Fattahi
"""


if __name__ == '__main__':

    # import scipy.io
    # import os
    import numpy as np
    import pickle
    import pandas as pd
    from sklearn import preprocessing, model_selection
    from getFeatureScore import getFeatureScore, selectByVote
    from sklearn.ensemble import RandomForestClassifier
    # from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
    from sklearn.metrics import f1_score
    import time
    # import datetime
    # import dill as pickle
    # import matplotlib.pyplot as plt

    def CinC2017f1score(true, pred):
        f = 0
        return f
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
    DataDerivenFeatures = pd.read_csv('DataDerivenFeatures.csv')
    ModelBasedFeatures = pd.read_csv('ModelBasedFeatures.csv')

    # Split data to train and test
    [train_index, test_index] = list(model_selection.StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=0).split(np.zeros(labels.shape[0]), labels.classNum))[0]

    labels, labels_Test = labels.iloc[train_index], labels.iloc[test_index]

    GenericFeatures, GenericFeatures_Test = GenericFeatures.iloc[
        train_index], GenericFeatures.iloc[test_index]
    SemiGenericFeatures, SemiGenericFeatures_Test = SemiGenericFeatures.iloc[
        train_index], SemiGenericFeatures.iloc[test_index]
    DataDerivenFeatures, DataDerivenFeatures_Test = DataDerivenFeatures.iloc[
        train_index], DataDerivenFeatures.iloc[test_index]
    ModelBasedFeatures, ModelBasedFeatures_Test = ModelBasedFeatures.iloc[
        train_index], ModelBasedFeatures.iloc[test_index]

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
    scalar_DataDerivenFeatures_train = []
    scalar_ModelBasedFeatures_train = []
    score_GenericFeatures = []
    score_SemiGenericFeatures = []
    score_DataDerivenFeatures = []
    score_ModelBasedFeatures = []
    kf = []

    # iterables:
    cvtimes = 5
    cvk = 10
    n_features = [400, 800, 1200, 1600]  # 4 rounds
    n_estimators = range(200, 1400, 200)  # 6 rounds

    # CV loops
    for cvturn in range(2, cvtimes):
        kf = (model_selection.StratifiedKFold(
            n_splits=cvk, shuffle=True, random_state=(cvturn+1)))
        k = 0
        for train_index, val_index in kf.split(np.zeros(labels.shape[0]), labels.classNum):
            print('turn: '+str(cvturn+1)+', fold no.: ' + str(k+1))
            # train-validation split
            GenericFeatures_train, SemiGenericFeatures_train, DataDerivenFeatures_train, ModelBasedFeatures_train,  labels_train = GenericFeatures.iloc[
                train_index], SemiGenericFeatures.iloc[train_index], DataDerivenFeatures.iloc[train_index], ModelBasedFeatures.iloc[train_index], labels.iloc[train_index]
            GenericFeatures_val, SemiGenericFeatures_val, DataDerivenFeatures_val, ModelBasedFeatures_val,  labels_val = GenericFeatures.iloc[
                val_index], SemiGenericFeatures.iloc[val_index], DataDerivenFeatures.iloc[val_index], ModelBasedFeatures.iloc[val_index], labels.iloc[val_index]

            # scaling the features (or standardization?)
            scalar_GenericFeatures_train = (
                preprocessing.StandardScaler().fit(GenericFeatures_train))
            GenericFeatures_train = pd.DataFrame(scalar_GenericFeatures_train.transform(
                GenericFeatures_train), columns=GenericFeatures_train.columns)
            GenericFeatures_val = pd.DataFrame(scalar_GenericFeatures_train.transform(
                GenericFeatures_val), columns=GenericFeatures_val.columns)

            scalar_SemiGenericFeatures_train = (preprocessing.StandardScaler().fit(
                SemiGenericFeatures_train))
            SemiGenericFeatures_train = pd.DataFrame(scalar_SemiGenericFeatures_train.transform(
                SemiGenericFeatures_train), columns=SemiGenericFeatures_train.columns)
            SemiGenericFeatures_val = pd.DataFrame(scalar_SemiGenericFeatures_train.transform(
                SemiGenericFeatures_val), columns=SemiGenericFeatures_val.columns)

            scalar_DataDerivenFeatures_train = (preprocessing.StandardScaler().fit(
                DataDerivenFeatures_train))
            DataDerivenFeatures_train = pd.DataFrame(scalar_DataDerivenFeatures_train.transform(
                DataDerivenFeatures_train), columns=DataDerivenFeatures_train.columns)
            DataDerivenFeatures_val = pd.DataFrame(scalar_DataDerivenFeatures_train.transform(
                DataDerivenFeatures_val), columns=DataDerivenFeatures_val.columns)

            scalar_ModelBasedFeatures_train = (preprocessing.StandardScaler().fit(
                ModelBasedFeatures_train))
            ModelBasedFeatures_train = pd.DataFrame(scalar_ModelBasedFeatures_train.transform(
                ModelBasedFeatures_train), columns=ModelBasedFeatures_train.columns)
            ModelBasedFeatures_val = pd.DataFrame(scalar_ModelBasedFeatures_train.transform(
                ModelBasedFeatures_val), columns=ModelBasedFeatures_val.columns)

            # initial feature selection
            methods = ['var_threshold', 'chi2_test', 'f_value', 'mutual_info']

            score_GenericFeatures = (getFeatureScore(
                GenericFeatures_train, labels_train.classNum, methods=methods))
            score_SemiGenericFeatures = (getFeatureScore(
                SemiGenericFeatures_train, labels_train.classNum, methods=methods))
            score_DataDerivenFeatures = (getFeatureScore(
                DataDerivenFeatures_train, labels_train.classNum, methods=methods))
            score_ModelBasedFeatures = (getFeatureScore(
                ModelBasedFeatures_train, labels_train.classNum, methods=methods))

            # grid search on hyperparameters
            CVresults_GenFeat = []
            # CLFs_GenFeat = []
            CVresults_SemGenFeat = []
            # CLFs_SemGenFeat = []
            CVresults_DDerFeat = []
            # CLFs_DDerFeat = []
            CVresults_MdlBsdFeat = []
            # CLFs_MdlBsdFeat = []

            # iterables
            n_fs_GenF = range(
                GenericFeatures_train.shape[1]//4-1, GenericFeatures_train.shape[1], GenericFeatures_train.shape[1]//4)
            n_fs_SGenF = range(
                SemiGenericFeatures_train.shape[1]//4-1, SemiGenericFeatures_train.shape[1], SemiGenericFeatures_train.shape[1]//4)
            n_fs_DDF = range(
                DataDerivenFeatures_train.shape[1]//4-1, DataDerivenFeatures_train.shape[1], DataDerivenFeatures_train.shape[1]//4)
            # nf_rng4 = range(ModelBasedFeatures_train.shape[1]//4, ModelBasedFeatures_train.shape[1], ModelBasedFeatures_train.shape[1]//4)
            n_fs_MBF = range(400, 1601, 400)
            for n_f_GenF, n_f_SGenF, n_f_DDF, n_f_MBF in zip(n_fs_GenF, n_fs_SGenF, n_fs_DDF, n_fs_MBF):
                # classifier learning
                for n_est in n_estimators:

                    # Generic Features -----------------------------------------
                    s = time.time()
                    clf = RandomForestClassifier(
                        n_estimators=n_est, criterion="entropy", random_state=0)
                    clf.fit(GenericFeatures_train[selectByVote(
                        score_GenericFeatures, n_f_GenF)], labels_train.classNum)
                    print("classifier learning time:", time.time()-s)

                    # predict the validation labels
                    pred = clf.predict(
                        GenericFeatures_val[selectByVote(score_GenericFeatures, n_f_GenF)])

                    # scoring and save the results
                    f1Sc = f1_score(labels_val.classNum,
                                    pred, average='macro')
                    sucRate = f1_score(labels_val.classNum,
                                       pred, average='micro')
                    mf1Sc = CinC2017f1score(labels_val.classNum, pred)
                    CVresults_GenFeat.append(pd.DataFrame(
                        data={'predict': pred, 'label': labels_val.classNum,
                              'success': sucRate, 'f1_score': f1Sc, 'modified f1_score': mf1Sc}))
                    # CLFs_GenFeat.append(clf)

                    # Semi-Generic Features ------------------------------------
                    s = time.time()
                    clf = RandomForestClassifier(
                        n_estimators=n_est, criterion="entropy", random_state=0)
                    clf.fit(SemiGenericFeatures_train[selectByVote(
                        score_SemiGenericFeatures, n_f_SGenF)], labels_train.classNum)
                    print("classifier learning time:", time.time()-s)

                    # predict the validation labels
                    pred = clf.predict(
                        SemiGenericFeatures_val[selectByVote(score_SemiGenericFeatures, n_f_SGenF)])

                    # scoring and save the results
                    f1Sc = f1_score(labels_val.classNum,
                                    pred, average='macro')
                    sucRate = f1_score(labels_val.classNum,
                                       pred, average='micro')
                    mf1Sc = CinC2017f1score(labels_val.classNum, pred)
                    CVresults_SemGenFeat.append(pd.DataFrame(
                        data={'predict': pred, 'label': labels_val.classNum,
                              'success': sucRate, 'f1_score': f1Sc, 'modified f1_score': mf1Sc}))
                    # CLFs_SemGenFeat.append(clf)

                    # Data-Deriven Features ------------------------------------
                    s = time.time()
                    clf = RandomForestClassifier(
                        n_estimators=n_est, criterion="entropy", random_state=0)
                    clf.fit(DataDerivenFeatures_train[selectByVote(
                        score_DataDerivenFeatures, n_f_DDF)], labels_train.classNum)
                    print("classifier learning time:", time.time()-s)

                    # predict the validation labels
                    pred = clf.predict(
                        DataDerivenFeatures_val[selectByVote(score_DataDerivenFeatures, n_f_DDF)])

                    # scoring and save the results
                    f1Sc = f1_score(labels_val.classNum,
                                    pred, average='macro')
                    sucRate = f1_score(labels_val.classNum,
                                       pred, average='micro')
                    mf1Sc = CinC2017f1score(labels_val.classNum, pred)
                    CVresults_DDerFeat.append(pd.DataFrame(
                        data={'predict': pred, 'label': labels_val.classNum,
                              'success': sucRate, 'f1_score': f1Sc, 'modified f1_score': mf1Sc}))
                    # CLFs_DDerFeat.append(clf)

                    # Model-Based Features -------------------------------------
                    s = time.time()
                    clf = RandomForestClassifier(
                        n_estimators=n_est, criterion="entropy", random_state=0)
                    clf.fit(ModelBasedFeatures_train[selectByVote(
                        score_ModelBasedFeatures, n_f_MBF)], labels_train.classNum)
                    print("classifier learning time:", time.time()-s)

                    # predict the validation labels
                    pred = clf.predict(
                        ModelBasedFeatures_val[selectByVote(score_ModelBasedFeatures, n_f_MBF)])

                    # scoring and save the results
                    f1Sc = f1_score(labels_val.classNum,
                                    pred, average='macro')
                    sucRate = f1_score(labels_val.classNum,
                                       pred, average='micro')
                    mf1Sc = CinC2017f1score(labels_val.classNum, pred)
                    CVresults_MdlBsdFeat.append(pd.DataFrame(
                        data={'predict': pred, 'label': labels_val.classNum,
                              'success': sucRate, 'f1_score': f1Sc, 'modified f1_score': mf1Sc}))
                    # CLFs_MdlBsdFeat.append(clf)

            # x = datetime.datetime.now()
            # fileName = os.path.basename(__file__)[:-3]+'tempSave_'+x.strftime("%Y%m%d%H%M")+'.pkl'
            fileName = 'tempSave_perF_cvturn' + \
                str(cvturn+1) + '_cvk'+str(k+1)+'.pkl'

            f = open(fileName, 'wb')
            pickle.dump([CVresults_GenFeat, CVresults_SemGenFeat,
                        CVresults_DDerFeat, CVresults_MdlBsdFeat, scalar_GenericFeatures_train,
                        scalar_SemiGenericFeatures_train, scalar_DataDerivenFeatures_train,
                        scalar_ModelBasedFeatures_train, score_GenericFeatures,
                        score_SemiGenericFeatures, score_DataDerivenFeatures,
                        score_ModelBasedFeatures, n_estimators, n_features,
                        val_index, cvk, cvtimes, methods, cvturn, fileName,
                        train_index, test_index, k, kf, n_est, n_f_GenF, n_f_SGenF,
                        n_f_DDF, n_f_MBF], f, -1)
            f.close()
            # pickle.dump_session(fileName)
            print(fileName+' is saved!')
            k = k+1
            # to restore session:
            # pickle.load_session(fileName)

    # f = open('tempSave_cvturn1_cvk1.pkl', 'rb')
    # valClassifResult, CLFs, scalar_GenericFeatures_train, scalar_SemiGenericFeatures_train, scalar_DataDerivenFeatures_train, scalar_ModelBasedFeatures_train, score_GenericFeatures, score_SemiGenericFeatures, score_DataDerivenFeatures, score_ModelBasedFeatures, n_estimators, n_features, val_index, cvk, cvtimes, methods, cvturn, fileName, train_index, test_index, k, kf, n_est, n_f = pickle.load(
    #     f)

    # ClassifRate = np.zeros([25, 1])
    # ClassifRep = np.zeros([25, 1])
    # for subCV in range(25):
    #     ClassifRate[subCV] = valClaasifResult[subCV].success.sum(
    #     )/valClaasifResult[subCV].success.size
    #     ClassifRep[subCV] = f1_score(
    #         valClaasifResult[subCV].label, valClaasifResult[subCV].predict, average='macro')


# plt.plot(range(10, 600, 10), 1-np.mean(ClassifRate, axis=0))
