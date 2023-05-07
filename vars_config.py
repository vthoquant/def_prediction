# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:03:51 2023

@author: vivin
"""

DEFAULT_PROBABIITY_PREDICTION = {
    "index": "uuid",
    "dep_var": "default",
    "indep_vars": ['account_amount_added_12_24m', 'account_days_in_dc_12_24m', 'age', 'num_unpaid_bills'],
    #"sample_weights": [1, 1, 1, 1],
    "na_fill_cols": {
        "account_days_in_dc_12_24m": -1
    }
}

VARS_COMBOS = {
    "def_probab_predict": DEFAULT_PROBABIITY_PREDICTION
}