
# import scipy.io
# import os
# import tsfresh
import numpy as np
# import pickle
import pymrmr
from ncafs import NCAFSC
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2, SelectKBest, VarianceThreshold
import pandas as pd
from sklearn import preprocessing
# import skfeature
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TuRF


GenericFeatures = pd.read_csv('GenericFeatures.csv')
SemiGenericFeatures_meanBeat = pd.read_csv('SemiGenericFeatures_meanBeat.csv')
DataDerivenFeatures = pd.read_csv('DataDerivenFeatures.csv')
ModelBasedFeatures = pd.read_csv('ModelBasedFeatures.csv')

# replace any nan with zero
GenericFeatures = GenericFeatures.fillna(0)
SemiGenericFeatures_meanBeat = SemiGenericFeatures_meanBeat.fillna(0)
DataDerivenFeatures = DataDerivenFeatures.fillna(0)
ModelBasedFeatures = ModelBasedFeatures.fillna(0)

# concat all features
allFeatures = pd.concat([GenericFeatures, SemiGenericFeatures_meanBeat,
                        DataDerivenFeatures, ModelBasedFeatures], axis=1)

# normalizing the data
GenericFeatures[:] = preprocessing.MinMaxScaler(
).fit_transform(GenericFeatures)
SemiGenericFeatures_meanBeat[:] = preprocessing.MinMaxScaler(
).fit_transform(SemiGenericFeatures_meanBeat)
DataDerivenFeatures[:] = preprocessing.MinMaxScaler(
).fit_transform(DataDerivenFeatures)
ModelBasedFeatures[:] = preprocessing.MinMaxScaler(
).fit_transform(ModelBasedFeatures)
allFeatures[:] = preprocessing.MinMaxScaler().fit_transform(allFeatures)


# perpare the labels
labels = pd.read_csv('REFERENCE-v3.csv', header=None)
labels.columns = ["recordName", "className"]
labels["classNum"] = np.nan
labels.at[labels.className == 'N', 'classNum'] = 0
labels.at[labels.className == 'A', 'classNum'] = 1
labels.at[labels.className == 'O', 'classNum'] = 2
labels.at[labels.className == '~', 'classNum'] = 3


# =============================================================================
# Scenario 1: selecting among all the features

# # 1-1- variance thresholding
vthSelector = VarianceThreshold(threshold=.005)
vthSelector.fit(allFeatures, labels)
vthFeatures_ind = vthSelector.get_support(indices=True)
vthFeatures = allFeatures.iloc[:, vthFeatures_ind]

# # 1-2- chi-squared test
chi2Selector = SelectKBest(
    chi2, k=100).fit(vthFeatures, labels.className)
chi2testlFeatures_ind = chi2Selector.get_support(indices=True)

# # 1-3- ANOVA F-value
fvSelector = SelectKBest(f_classif, k=100).fit(
    vthFeatures, labels.className)
fvaluelFeatures_ind = fvSelector.get_support(indices=True)
# a = fvaluelFeatures.scores_
# b = fvaluelFeatures.pvalues_

# # 1-4-  mutual_info
mutInfSelector = SelectKBest(
    mutual_info_classif, k=100).fit(vthFeatures, labels.className)
mutInfoFeatures_ind = mutInfSelector.get_support(indices=True)

# # 1-5- MRMR
df = vthFeatures.copy()
df.insert(0, "labels", labels.classNum)
mrmrSelector = pymrmr.mRMR(df, 'MIQ', 100)

# # 1-6- NCAFSC
ncaSelector = NCAFSC()
ncaSelector.fit(vthFeatures, labels.classNum)
w = ncaSelector.weights

# # 1-7- ReliefF
reliffSelector = ReliefF()
reliffSelector.fit(vthFeatures.to_numpy(), labels.classNum.to_numpy())

# # 1-8- SURF
surfSelector = SURF()
surfSelector.fit(vthFeatures.to_numpy(), labels.classNum.to_numpy())

# # 1-9- SURFstar
surfstarSelector = SURFstar()
surfstarSelector.fit(vthFeatures.to_numpy(), labels.classNum.to_numpy())


# # 1-10- MultiSURF
multisurfSelector = MultiSURF()
multisurfSelector.fit(vthFeatures.to_numpy(), labels.classNum.to_numpy())

# # 1-11- MultiSURFstar
multisurfstarSelector = MultiSURFstar()
multisurfstarSelector.fit(vthFeatures.to_numpy(), labels.classNum.to_numpy())

headers = list(vthFeatures)
turfSelector = TuRF(core_algorithm="ReliefF",
                    n_features_to_select=2, pct=0.5, verbose=True)
turfSelector.fit(vthFeatures.to_numpy(), labels.classNum.to_numpy(), headers)

turfSelector.feature_importances_

# # (skipped) 1-12- Laplacian score
# a = skfeature.function.similarity_based.lap_score(vthFeatures)

# =============================================================================
# =============================================================================
# Scenario 2: selecting from each type and ranking the selected features

# # 2-1- variance thresholding
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
DataDerivenFeatures_vth = sel.fit_transform(DataDerivenFeatures)

# # 2-2- chi-squared test
DataDerivenFeatures_chi2test = SelectKBest(
    chi2, k=100).fit_transform(DataDerivenFeatures, labels.className)

# # 2-3- ANOVA F-value
DataDerivenFeatures_fvalue = SelectKBest(f_classif, k=100).fit_transform(
    DataDerivenFeatures, labels.className)

# # 2-4-  mutual_info
DataDerivenFeatures_mutualInfo = SelectKBest(
    mutual_info_classif, k=100).fit_transform(DataDerivenFeatures, labels.className)

# # 2-5- NCAFSC
fs_clf = NCAFSC()
fs_clf.fit(DataDerivenFeatures, labels.className)
w = fs_clf.weights

# # 2-6- MRMR
df = DataDerivenFeatures.copy()
df.insert(0, "labels", labels.className)
pymrmr.mRMR(df, 'MIQ', 100)
# =============================================================================
# =============================================================================
# # Scenario 3: Classification results


# =============================================================================
