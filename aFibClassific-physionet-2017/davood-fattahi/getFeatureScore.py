from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TuRF
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2, SelectKBest, VarianceThreshold
from ncafs import NCAFSC
# import pymrmr
import numpy as np
import time


def getFeatureScore(features, labels, methods=[]):

    # import pickle
    # import skfeature

    # replace any nan with zero
    features = features.fillna(0)

    # # scaling and standardizing the data
    features = pd.DataFrame(preprocessing.MinMaxScaler(
    ).fit_transform(features), columns=features.columns)

    scores = {}
    if (not(bool(methods))) | ('var_threshold' in methods):
        # # 1-1- variance thresholding
        s = time.time()
        vthSelector = VarianceThreshold()
        vthSelector.fit(features, labels)
        scores['var_threshold'] = vthSelector.variances_
        print("var_threshold running time:", time.time()-s)

    if (not bool(methods)) | ('chi2_test' in methods):
        # # 1-2- chi-squared test
        s = time.time()
        chi2Selector = SelectKBest(
            chi2).fit(features, labels)
        scores['chi2_test'] = chi2Selector.scores_
        print("chi2_test running time:", time.time()-s)

    if (not bool(methods)) | ('f_value' in methods):
        # # 1-3- ANOVA F-value
        s = time.time()
        fvSelector = SelectKBest(f_classif).fit(
            features, labels)
        # fvaluelfeatures_ind = fvSelector.get_support(indices=True)
        scores['f_value'] = fvSelector.scores_
        print("f_value running time:", time.time()-s)

    if (not bool(methods)) | ('mutual_info' in methods):
        # # 1-4-  mutual_info
        s = time.time()
        mutInfSelector = SelectKBest(
            mutual_info_classif).fit(features, labels)
        scores['mutual_info'] = mutInfSelector.scores_
        print("mutual_info running time:", time.time()-s)

    # if (not bool(methods)) | ('mrmr' in methods):
    #     # # (skipped) 1-5- MRMR
    #     s = time.time()
    #     df = features.copy()
    #     df.insert(0, "labels", labels)
    #     mrmrSelector = pymrmr.mRMR(df, 'MIQ', features.shape[1])
    #     # mrmrSelector = pymrmr.mRMR(df, 'MID', features.shape[1])
    #     scores['mrmr'] = features.shape[1] - \
    #         np.array([mrmrSelector.index(i) for i in features.columns])
    #     print("mrmr running time:", time.time()-s)

    if (not bool(methods)) | ('nca' in methods):
        # # 1-6- NCAFSC (heavy comp cost)
        s = time.time()
        ncaSelector = NCAFSC()
        ncaSelector.fit(features, labels)
        scores['nca'] = ncaSelector.weights_
        print("nca running time:", time.time()-s)

    if (not bool(methods)) | ('relieff' in methods):
        # # 1-7- ReliefF (very heavy comp cost)
        s = time.time()
        reliffSelector = ReliefF()
        reliffSelector.fit(features.to_numpy(), labels.to_numpy())
        scores['relieff'] = reliffSelector.feature_importances_
        print("relieff running time:", time.time()-s)

    if (not bool(methods)) | ('surf' in methods):
        # # 1-8- SURF (very heavy comp cost)
        s = time.time()
        surfSelector = SURF()
        surfSelector.fit(features.to_numpy(), labels.to_numpy())
        scores['surf'] = surfSelector.feature_importances_
        print("surf running time:", time.time()-s)

    if (not bool(methods)) | ('surf*' in methods):
        # # 1-9- SURFstar (very heavy comp cost)
        s = time.time()
        surfstarSelector = SURFstar()
        surfstarSelector.fit(features.to_numpy(), labels.to_numpy())
        scores['surf*'] = surfstarSelector.feature_importances_
        print("surf* running time:", time.time()-s)

    if (not bool(methods)) | ('multisurf' in methods):
       # # 1-10- MultiSURF (very heavy comp cost)
        s = time.time()
        multisurfSelector = MultiSURF()
        multisurfSelector.fit(features.to_numpy(), labels.to_numpy())
        scores['multisurf'] = multisurfSelector.feature_importances_
        print("multisurf running time:", time.time()-s)

    if (not bool(methods)) | ('multisurf*' in methods):
        # # 1-11- MultiSURFstar (heavy comp cost)
        s = time.time()
        multisurfstarSelector = MultiSURFstar()
        multisurfstarSelector.fit(features.to_numpy(), labels.to_numpy())
        scores['multisurf*'] = multisurfstarSelector.feature_importances_
        print("multisurf* running time:", time.time()-s)

    if (not bool(methods)) | ('turf' in methods):
        # 1-12- turf (very heavy comp cost)
        # turf is a iterative approach of relieff algorithm, suitable for high dimensional feature space (>10000)
        s = time.time()
        headers = list(features)
        turfSelector = TuRF(core_algorithm="ReliefF", verbose=True)
        turfSelector.fit(features.to_numpy(),
                         labels.to_numpy(), headers)
        scores['turf'] = turfSelector.feature_importances_
        print("turf running time:", time.time()-s)

    # # (skipped) 1-12- Laplacian score
    # a = skfeature.function.similarity_based.lap_score(features)

    featureScores = pd.DataFrame(data=scores, index=features.columns)

    return featureScores


def selectByVote(df, n):
    a = []
    for i in df.columns:
        a = a+list(df.sort_values(
            by=i, ascending=False).index[0:n])
    values, counts = np.unique(a, return_counts=True)
    f = values[(-counts).argsort()][0:n]
    return f


# def selectByWeight(df, n):
#     a = []
#     for i in df.columns:
#         a = a+list(df.sort_values(
#             by=i, ascending=False).index[0:n])
#     values, counts = np.unique(a, return_counts=True)
#     f = values[(-counts).argsort()][0:n]
#     return f
