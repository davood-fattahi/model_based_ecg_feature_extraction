
import numpy as np
import pandas as pd

from sklearn import preprocessing
from getFeatureScore import getFeatureScore
from sklearn.feature_selection import VarianceThreshold

import mifs


GenericFeatures = pd.read_csv('GenericFeatures.csv')
GenericFeatures = GenericFeatures.fillna(0)
GenericFeatures[:] = preprocessing.MinMaxScaler(
).fit_transform(GenericFeatures)

# perpare the labels
labels = pd.read_csv('REFERENCE-v3.csv', header=None)
labels.columns = ["recordName", "className"]
labels["classNum"] = np.nan
labels.at[labels.className == 'N', 'classNum'] = 0
labels.at[labels.className == 'A', 'classNum'] = 1
labels.at[labels.className == 'O', 'classNum'] = 2
labels.at[labels.className == '~', 'classNum'] = 3


fscores = getFeatureScore(features, labels, ['var_threshold', 'chi2_test',
                                             'f_value', 'mutual_info', 'mrmr',
                                             'nca', 'relieff', 'surf', 'surf*',
                                             'multisurf', 'multisurf*'])


df = pd.read_csv('test_lung_s3.csv')

# # 1-1- variance thresholding
vthSelector = VarianceThreshold(threshold=.05)
vthSelector.fit(GenericFeatures, labels)
vthFeatures_ind = vthSelector.get_support(indices=True)
vthFeatures = GenericFeatures.iloc[:, vthFeatures_ind]
featureScores = getFeatureScore(vthFeatures, labels.classNum)
