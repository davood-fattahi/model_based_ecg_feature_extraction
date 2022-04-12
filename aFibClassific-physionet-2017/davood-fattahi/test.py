# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:00:15 2022

@author: RPH
"""

fscores_best400 = getFeatureScore(
    GF_train_sc, labels_train.classNum, methods=methods)
allFeatures_best100 = selectByVote(fscores_best400, 100)
