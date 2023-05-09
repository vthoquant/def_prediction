# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:36:47 2023

@author: vivin
"""
import bentoml
import pandas as pd
from io import StringIO
from bentoml.io import Text, PandasDataFrame, Multipart, JSON

bentoml_model_name = 'default_predictor'
bentoml_model_version = 'latest'
bentoml_service_name = 'default_classifier_v2'


bento_model = bentoml.models.get("{}:{}".format(bentoml_model_name, bentoml_model_version))

indep_vars = bento_model.info.metadata['indep_vars']
dep_var = bento_model.info.metadata['dep_var']
index_col = bento_model.info.metadata['index_col']

bento_runner = bento_model.to_runner()

#create service object
svc = bentoml.Service(bentoml_service_name, runners=[bento_runner])

@svc.api(input=Multipart(data=Text()), output=PandasDataFrame())
def run_classify(data):
    data_file = StringIO(data)
    data_df = pd.read_csv(data_file, delimiter=";")
    data_df = data_df[[index_col] + indep_vars]
    data_df.set_index(index_col, inplace=True)
    data_df.dropna(how='any', inplace=True)
    labels = bento_runner.predict_proba.run(data_df)[:,1] #probability of default==1
    ret_df = pd.DataFrame(data={'uuid':data_df.index.values, 'pd':labels})
    return ret_df

@svc.api(input=JSON(), output=PandasDataFrame())
def run_classify_json(inp_json):    
    data_df = pd.read_json(inp_json)
    data_df = data_df[[index_col] + indep_vars]
    data_df.set_index(index_col, inplace=True)
    data_df.dropna(how='any', inplace=True)
    labels = bento_runner.predict_proba.run(data_df)[:,1] #probability of default==1
    ret_df = pd.DataFrame(data={'uuid':data_df.index.values, 'pd':labels})
    
    return ret_df
    
    