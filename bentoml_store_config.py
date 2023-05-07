# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:07:31 2023

@author: vivin
"""

DEFAULT_PREDICT_MODELS_SKLEARN = {
    "model_name": "default_predictor",
    "version": "latest",
    "model_class": "sklearn"
}

BENTOML_STORE = {
    "default_predict": DEFAULT_PREDICT_MODELS_SKLEARN
}