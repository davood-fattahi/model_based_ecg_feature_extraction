if __name__ == '__main__':

    # import scipy.io
    # import os
    import numpy as np
    import pickle
    import pandas as pd
    from sklearn import preprocessing, model_selection
    from getFeatureScore import getFeatureScore, selectByVote
    from sklearn.ensemble import RandomForestClassifier
    import time

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

    # concat all features
    allFeatures = pd.concat([GenericFeatures, SemiGenericFeatures,
                             DataDerivenFeatures, ModelBasedFeatures], axis=1)

    # remove constant features
    allFeatures = allFeatures.drop(
        allFeatures.columns[allFeatures.nunique() <= 1], axis=1)

    # the feature-selection MUST be included in cross-validation!
    #   see: https://www.nodalpoint.com/not-perform-feature-selection/
    # why we don not use pipeline from sklearn? 1- feature selection must be
    #   included in CV, and the fs methods are not standard methods of sklearn.
    #   2- we use two steps of feature selection. 3- we need to keep the results of
    #   each fs step. 4- by the way, we can write a sklearn compatible module, but
    #   it is time-consuming and not our priority for now.
    allFscore = []
    valClaasifResult = []
    cvtimes = 5
    for cvturn in range(cvtimes):
        kf = model_selection.KFold(
            n_splits=5, shuffle=True, random_state=(cvturn+1))
        for train_index, val_index in kf.split(labels):
            # train-validation split
            allF_train, labels_train = allFeatures.iloc[train_index], labels.iloc[train_index]

            allF_val, labels_val = allFeatures.iloc[val_index], labels.iloc[val_index]

            # scaling the features (or standardization?)
            scalar = preprocessing.MinMaxScaler().fit(allF_train)
            allF_train = pd.DataFrame(scalar.transform(
                allF_train), columns=allF_train.columns)
            allF_val = pd.DataFrame(scalar.transform(
                allF_val), columns=allF_val.columns)

            # feature selection
            methods = ['var_threshold', 'chi2_test', 'f_value', 'mutual_info']
            allFscore.append(getFeatureScore(
                allF_train, labels_train.classNum, methods=methods))

            clf = RandomForestClassifier(
                n_estimators=300, criterion="entropy", random_state=0)
            cr = []
            for n in range(10, 600, 10):
                bestFnames = selectByVote(allFscore[-1], n)
                s = time.time()
                clf.fit(allF_train[bestFnames], labels_train.classNum)
                print("classifier learning time:", time.time()-s)
                cr.append(clf.predict(
                    allF_val[bestFnames]) == labels_val.classNum)

            valClaasifResult.append(cr)

            f = open('tempSave.pkl', 'wb')
            pickle.dump([allFscore, valClaasifResult], f, -1)
            f.close()


# f = open('tempSave.pkl', 'rb')
# allFscore, valClaasifResult = pickle.load(f)
