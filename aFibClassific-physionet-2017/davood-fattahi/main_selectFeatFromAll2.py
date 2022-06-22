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
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline, make_pipeline

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

    make_pipeline(selectByVote(), getFeatureScore(), RandomForestClassifier())
