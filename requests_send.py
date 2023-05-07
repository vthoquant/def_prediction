# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:11:12 2023

@author: vivin
"""

import json
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


endpoint = "https://bxq98dklyj.execute-api.us-west-1.amazonaws.com/classify"

def generate_sample_data():
    # Generate sample data for prediction
    GLOBAL_PATH = 'C:\\Users\\vivin\\Spyder projects\\def_prediction\\'
    TEST_FRACTION = 0.1

    y_col = 'default'
    X_cols = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m', 'age', 'num_unpaid_bills']

    df_raw = pd.read_csv('{}dataset.csv'.format(GLOBAL_PATH), delimiter=";")
    #cleaning: remove nan from output
    df_raw = df_raw[~np.isnan(df_raw['default'])]
    df_raw.set_index('uuid', inplace=True)
    #cleaning: some other cleaning of other columns - make another function out of this
    df_raw.loc[:, 'account_days_in_dc_12_24m'].fillna(-1, inplace=True)

    X = df_raw[X_cols]
    y = df_raw[y_col]

    _, X_test, _, _ = train_test_split(X, y, test_size=TEST_FRACTION, random_state=100)
    return X_test

X_test = generate_sample_data()

sample_json = json.dumps(X_test.values.tolist())

response = requests.post(endpoint, headers={"content-type": "application/json"}, data=sample_json)