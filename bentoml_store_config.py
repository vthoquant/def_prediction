# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:07:31 2023

@author: vivin
"""

PD_PREDICT_RF_DSAMPLE_SEQ5 = {
    "model_name": "pd_predict_rf_ds_seq5",
    "version": "latest",
    "model_class": "sklearn"
}

PD_PREDICT_RF_USAMPLE_SEQ5 = {
    "model_name": "pd_predict_rf_us_seq5",
    "version": "latest",
    "model_class": "sklearn"
}

PD_PREDICT_RF_DSAMPLE_SEQ10 = {
    "model_name": "pd_predict_rf_ds_seq10",
    "version": "latest",
    "model_class": "sklearn"
}

PD_PREDICT_RF_USAMPLE_SEQ10 = {
    "model_name": "pd_predict_rf_us_seq10",
    "version": "latest",
    "model_class": "sklearn"
}

PD_PREDICT_RF_DSAMPLE = {
    "model_name": "pd_predict_rf_ds",
    "version": "latest",
    "model_class": "sklearn"
}

PD_PREDICT_RF_USAMPLE = {
    "model_name": "pd_predict_rf_us",
    "version": "latest",
    "model_class": "sklearn"
}

PD_PREDICT_LR_BAL = {
    "model_name": "pd_predict_lr_bal",
    "version": "latest",
    "model_class": "sklearn"
}

BENTOML_STORE = {
    "pd_predict_rf_ds_seq5": PD_PREDICT_RF_DSAMPLE_SEQ5,
    "pd_predict_rf_us_seq5": PD_PREDICT_RF_USAMPLE_SEQ5,
    "pd_predict_rf_ds_seq10": PD_PREDICT_RF_DSAMPLE_SEQ10,
    "pd_predict_rf_us_seq10": PD_PREDICT_RF_USAMPLE_SEQ10,
    "pd_predict_rf_ds": PD_PREDICT_RF_DSAMPLE,
    "pd_predict_rf_us": PD_PREDICT_RF_USAMPLE,
    "pd_predict_lr_bal": PD_PREDICT_LR_BAL
}