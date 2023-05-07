# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:36:47 2023

@author: vivin
"""
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, Text

bento_knn = bentoml.models.get("knn_initial:latest")

knn_runner = bento_knn.to_runner()

#create service object
svc = bentoml.Service("default_classifier", runners=[knn_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series) -> np.ndarray:
    # Convert the input string to numpy array
    label = knn_runner.predict.run(input_series)

    return label

@svc.api(input=Text(), output=Text())
def classify_test(inp):
    # Convert the input string to numpy array
    label = "Output is : {}".format(inp)

    return label