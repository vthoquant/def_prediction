# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:36:10 2023

@author: vivin
"""

KNN_BASE = {
    "model": "KNN",
    "params": {"n_neighbors": 5}
}

SVC_BASE = {
    "model": "SVC",
    "params": {
        "kernel": "rbf"
    }
}

SVC_BALANCED = {
    "model": "SVC",
    "params": {
        "kernel": "linear",
        "class_weight": "balanced"
    }
}

RF_CLASSIFIER = {
    "model": "RandomForestClassifier",
    "params": {
        "n_estimators": 100,
        "max_depth": 10
    }
}

ALL_MODELS = {
    "knn_base": KNN_BASE,
    "svc_base": SVC_BASE,
    "svc_balanced": SVC_BALANCED,
    "random_forest": RF_CLASSIFIER
}