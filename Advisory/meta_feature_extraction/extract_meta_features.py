import rpy2.robjects as robj
import pandas as pd
import numpy as np
import hellinger.complexity_curve as cc
import Path


def extract_meta_features(data):
    robj.globalenv['args'] = robj.FloatVector(data)
    robj.r.source(Path.BasePath + "Advisory/meta_feature_extraction/metaFeatureExtraction.R")
    frame = robj.globalenv['resultForPython']
    pd_df = pd.DataFrame.from_dict({key: np.asarray(frame.rx2(key)) for key in frame.names})
    pd_df.rename(columns = {'x': 'CORR'}, inplace=True)
    pd_df['CC'] = cc.calculate_data_complexity(data, 30, 5, (int(data.size/data.size)))[0]
    return pd_df

