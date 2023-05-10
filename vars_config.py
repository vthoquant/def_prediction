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

DEFAULT_PROBABIITY_PREDICTION_UPSAMPLE = {
    "index": "uuid",
    "dep_var": "default",
    "indep_vars": ['account_amount_added_12_24m', 'account_days_in_dc_12_24m', 'age', 'num_unpaid_bills'],
    "imbalance_handling": "over-sample", #or under-sample
    #"sample_weights": [1, 1, 1, 1],
    "na_fill_cols": {
        "account_days_in_dc_12_24m": -1
    }
}

DEFAULT_PROBABIITY_PREDICTION_DNSAMPLE = {
    "index": "uuid",
    "dep_var": "default",
    "indep_vars": ['account_amount_added_12_24m', 'account_days_in_dc_12_24m', 'age', 'num_unpaid_bills'],
    "imbalance_handling": "under-sample", #or under-sample
    #"sample_weights": [1, 1, 1, 1],
    "na_fill_cols": {
        "account_days_in_dc_12_24m": -1
    }
}

ALLINDEPS_DNSAMPLE = {
    "index": "uuid",
    "dep_var": "default",
    "imbalance_handling": "under-sample", #or under-sample
    "na_fill_cols": {
        "account_days_in_dc_12_24m": -1,
        "account_days_in_rem_12_24m": -1,
        "account_days_in_term_12_24m": -1,
        "account_incoming_debt_vs_paid_0_24m": -1,
        "account_status": -1,
        "account_worst_status_0_3m": -1,
        "account_worst_status_12_24m": -1,
        "account_worst_status_3_6m": -1,
        "account_worst_status_6_12m": -1,
        "avg_payment_span_0_12m": -1,
        "avg_payment_span_0_3m": -1,
        "num_active_div_by_paid_inv_0_12m": -1,        
        "num_arch_written_off_0_12m": -1,
        "num_arch_written_off_12_24m": -1,
        "worst_status_active_inv": -1,
    },
    "categories_convert": ["merchant_category", "merchant_group", "name_in_email"],
    "use_pca_transform": True,
    #"feature_selector": "sequential_2:forward"
}

VARS_COMBOS = {
    "def_data_handle": DEFAULT_PROBABIITY_PREDICTION,
    "def_data_handle_usample": DEFAULT_PROBABIITY_PREDICTION_UPSAMPLE,
    "def_data_handle_dsample": DEFAULT_PROBABIITY_PREDICTION_DNSAMPLE,
    "all_indeps_dsample": ALLINDEPS_DNSAMPLE,
}