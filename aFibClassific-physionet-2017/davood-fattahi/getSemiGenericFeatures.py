

if __name__ == '__main__':
    import scipy.io
    import os
    import tsfresh
    import pandas as pd
    import numpy as np
    
    
    DIR="segmentedBeats/"
    fs=300
    df = pd.DataFrame()
    id=0
    features=pd.DataFrame()
    for record in os.listdir(DIR):
        if record.endswith(".mat"):
            id +=1
            print(id)
            sdf = pd.DataFrame()
            mat_data = scipy.io.loadmat(DIR + record)
            ecgBeats = mat_data['EcgBeats']
            sdf=pd.DataFrame(np.array(ecgBeats.T))
            sdf['time']= np.arange(0, sdf.shape[0], 1, dtype=int).reshape(sdf.shape[0], 1)/fs
            sdf['id']=id
            # df = df.append(sdf)
            extracted_features = tsfresh.extract_features(sdf, column_id="id", column_sort="time")
            features = features.append(extracted_features)
    
    features.to_csv('semiGenericFeatures_beats.csv')

    # features = pd.read_csv ('GenericFeatures.csv')
    
    
