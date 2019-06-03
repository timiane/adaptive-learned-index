import rpy2.robjects as robj
import pandas as pd
import numpy as np
import hellinger.complexity_curve as cc


def extract_meta_features(data):
    print("Extracting meta features")
    robj.globalenv['args'] = robj.FloatVector(data)
    robj.r.source("C:/Users/Timian/Documents/Code/ARR/metaFeatureExtraction.R")
    frame = robj.globalenv['resultForPython']
    pd_df = pd.DataFrame.from_dict({key: np.asarray(frame.rx2(key)) for key in frame.names})
    pd_df.rename(columns = {'x': 'CORR'}, inplace=True)
    pd_df['CC'] = cc.calculate_data_complexity(data, 30, 5, (int(data.size/data.size)))[0]
    print("Meta features extracted")
    return pd_df

