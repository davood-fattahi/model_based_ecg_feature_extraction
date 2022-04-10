

if __name__ == '__main__':
    import scipy.io
    import os
    import tsfresh
    import pandas as pd
    import numpy as np
    import pickle
   
    
    DIR="../training2017/"
    fs=300
    itnum=5837
    features=pd.DataFrame()
    for record in os.listdir(DIR)[11674:]:
        if record.endswith(".mat"):
            itnum +=1
            print(itnum)
            print(record)
            sdf = pd.DataFrame()
            mat_data = scipy.io.loadmat(DIR + record)
            ecg = mat_data['val']
            sdf['value']=pd.DataFrame(ecg.T)
            sdf['time']= np.arange(0, sdf.size, 1, dtype=int).reshape(sdf.size, 1)/fs
            sdf['id']=itnum
            extracted_features = tsfresh.extract_features(sdf, column_id="id", column_sort="time")
            features = features.append(extracted_features)
    
    features.to_csv('GenericFeatures.csv', index = False)
    
    f = open('GenericFeatures.pkl', 'wb')
    pickle.dump(features, f, -1)
    f.close()   
