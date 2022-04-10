from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TuRF
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2, SelectKBest, VarianceThreshold
from ncafs import NCAFSC
import pymrmr
import numpy as np


def getFeatureScore(features, labels):

    # import pickle
    # import skfeature

    # replace any nan with zero
    features = features.fillna(0)

    # normalizing the data
    features[:] = preprocessing.MinMaxScaler(
    ).fit_transform(features)

    # # 1-1- variance thresholding
    vthSelector = VarianceThreshold()
    vthSelector.fit(features, labels)
    vtFeatures_scores = vthSelector.variances_

    # # 1-2- chi-squared test
    chi2Selector = SelectKBest(
        chi2).fit(features, labels)
    chi2testlFeatures_scores = chi2Selector.scores_

    # # 1-3- ANOVA F-value
    fvSelector = SelectKBest(f_classif).fit(
        features, labels)
    # fvaluelFeatures_ind = fvSelector.get_support(indices=True)
    fvaluelFeatures_scores = fvSelector.scores_

    # # 1-4-  mutual_info
    mutInfSelector = SelectKBest(
        mutual_info_classif).fit(features, labels)
    mutInfoFeatures_scores = mutInfSelector.scores_

    # # 1-5- MRMR
    df = features.copy()
    df.insert(0, "labels", labels)
    mrmrSelector = pymrmr.mRMR(df, 'MIQ', features.shape[1])
    # mrmrSelector = pymrmr.mRMR(df, 'MID', features.shape[1])
    mrmrFeatures_score = features.shape[1] - \
        np.array([mrmrSelector.index(i) for i in features.columns])

    # # 1-6- NCAFSC
    ncaSelector = NCAFSC()
    ncaSelector.fit(features, labels)
    ncaFeatures_scores = ncaSelector.weights_

    # # 1-7- ReliefF
    reliffSelector = ReliefF()
    reliffSelector.fit(features.to_numpy(), labels.to_numpy())
    reliefFeatures_scores = reliffSelector.feature_importances_

    # # 1-8- SURF
    surfSelector = SURF()
    surfSelector.fit(features.to_numpy(), labels.to_numpy())
    reliefFeatures_scores = reliffSelector.feature_importances_

    # # 1-9- SURFstar
    surfstarSelector = SURFstar()
    surfstarSelector.fit(features.to_numpy(), labels.to_numpy())

    # # 1-10- MultiSURF
    multisurfSelector = MultiSURF()
    multisurfSelector.fit(features.to_numpy(), labels.to_numpy())

    # # 1-11- MultiSURFstar
    multisurfstarSelector = MultiSURFstar()
    multisurfstarSelector.fit(features.to_numpy(),
                              labels.to_numpy())

    headers = list(features)
    turfSelector = TuRF(core_algorithm="ReliefF",
                        n_features_to_select=2, pct=0.5, verbose=True)
    turfSelector.fit(features.to_numpy(),
                     labels.to_numpy(), headers)

    turfSelector.feature_importances_

    # # (skipped) 1-12- Laplacian score
    # a = skfeature.function.similarity_based.lap_score(features)

    featurescores = []
    return featurescores
