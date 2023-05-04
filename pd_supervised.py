# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

GLOBAL_PATH = 'C:\\Users\\vivin\\Spyder projects\\klarna\\'
TEST_FRACTION = 0.1
N_NEIGHBORS = 5

y_col = 'default'
X_cols = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m', 'age', 'num_unpaid_bills']
X_cols = ['account_days_in_dc_12_24m']

df_raw = pd.read_csv('{}dataset.csv'.format(GLOBAL_PATH), delimiter=";")
#cleaning: remove nan from output
df_raw = df_raw[~np.isnan(df_raw['default'])]
df_raw.set_index('uuid', inplace=True)
#cleaning: some other cleaning of other columns - make another function out of this
df_raw.loc[:, 'account_days_in_dc_12_24m'].fillna(-1, inplace=True)

X = df_raw[X_cols]
y = df_raw[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, random_state=100)
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)