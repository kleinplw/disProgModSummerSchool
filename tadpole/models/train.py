from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd


def preprocess(data, window_size=2):

    # convert timestamp of examdate
    data['EXAMDATE'] = data['EXAMDATE'].apply(pd.to_numeric)
    label_cols = ['ADAS']

    # Feature selection
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    # data scaling
    data = data.select_dtypes(include=numerics)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data)

    # data filtering
    patient_ids = data['RID'].unique()
    pdata = []
    plabels = []
    for pid in patient_ids:
        exams = data.loc[data['RID'] == pid].sort_values(by=['EXAMDATE'])
        tl_exams = []
        tl_labels = []
        # transform in time lag series
        rows, col = exams.shape()
        for idx, exam in enumerate(exams.iterrows()):
            if idx >= window_size - 1:
                # + current date
                for jdx in range(idx, rows):
                    tl_exams.append(list(exams.loc[idx - window_size:idx]) + list(exams.loc[jdx, 'EXAMDATE']))
                    tl_labels.append(list(exams.loc[idx, label_cols]))
                plabels.append(tl_labels)
                pdata.append(tl_exams)
    return pdata, plabels


def train(data, labels):
    # fit model
    rf = RandomForestRegressor().fit(data, labels)
    return rf


def predict(model, scaler, data, n_pred, window_size):
    data['EXAMDATE'] = data['EXAMDATE'].apply(pd.to_numeric)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=numerics).sort_values(by=['EXAMDATE'])
    data = scaler.transform(data)
    tl_data = data.loc[-window_size:-1]
    predictions = []
    for idx in range(n_pred):
        last_date = tl_data.loc[-1, 'EXAM_DATE']
        curr_date = pd.offsets.MonthOffset(idx) + last_date
        predictions.append(model.predict(list(tl_data) + list(curr_date)))
    return predictions

