

if __name__ == '__main__':
    import scipy.io
    # import os
    import tsfresh
    import pandas as pd
    import numpy as np
    import pickle
    
    
    mat_data = scipy.io.loadmat('BeatMean.mat')
    BeatMean = mat_data['BeatMean']
    BeatMedian = mat_data['BeatMedian']
    fs=300
    id=0
    features=pd.DataFrame()
    for n in np.arange(5, BeatMean.size, 1):
        id +=1
        print(id)
        sdf = pd.DataFrame()
        sdf['mean']=pd.DataFrame(BeatMean[n,0].T)
        sdf['median']=pd.DataFrame(BeatMedian[n,0].T)
        sdf['time']= np.arange(0, sdf.shape[0], 1, dtype=int).reshape(sdf.shape[0], 1)/fs
        sdf['id']=id
        extracted_features = tsfresh.extract_features(sdf, column_id="id", column_sort="time")
        features = features.append(extracted_features)
    
    features.to_csv('SemiGenericFeatures_meanBeat.csv', index = False)
    
    f = open('SemiGenericFeatures_meanBeat.pkl', 'wb')
    pickle.dump(features, f, -1)
    f.close()   


# features = pd.read_pickle("SemiGenericFeatures_meanBeat.pkl")