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

DEFAULT_PREDICT_MODELS_SKLEARN_PCA = {
    "model_name": "default_predictor_w_pca",
    "version": "latest",
    "model_class": "sklearn"
}

DEFAULT_PREDICT_MODELS_SKLEARN_TEST = {
    "model_name": "default_predictor_test",
    "version": "latest",
    "model_class": "sklearn"
}

BENTOML_STORE = {
    "default_predict": DEFAULT_PREDICT_MODELS_SKLEARN,
    "default_predict_with_pca": DEFAULT_PREDICT_MODELS_SKLEARN_PCA,
    "default_predict_test": DEFAULT_PREDICT_MODELS_SKLEARN_TEST,
}