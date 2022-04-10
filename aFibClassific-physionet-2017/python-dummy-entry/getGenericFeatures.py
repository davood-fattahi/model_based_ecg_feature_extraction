# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:36:34 2022

@author: Davood
"""

import scipy.io
import os
import tsfresh


DIR="../training2017/"
for record in os.listdir(DIR):
    if record.endswith(".mat"):
        mat_data = scipy.io.loadmat(DIR + record)
        ecg = mat_data['val']
        extracted_features = tsfresh.extract_features(ecg, column_id="id", column_sort="time")
        
