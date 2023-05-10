# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:36:47 2023

@author: vivin
"""
import bentoml
import pandas as pd
from io import StringIO
from bentoml.io import Text, PandasDataFrame, Multipart, JSON
from sklearn.preprocessing import LabelEncoder

bentoml_model_name = 'default_predictor_w_pca'
bentoml_model_version = 'latest'
bentoml_service_name = 'default_classifier_pca_test'


bento_model = bentoml.models.get("{}:{}".format(bentoml_model_name, bentoml_model_version))

indep_vars = bento_model.info.metadata['indep_vars']
dep_var = bento_model.info.metadata['dep_var']
index_col = bento_model.info.metadata['index_col']
feature_selector = bento_model.info.metadata.get('feature_selector', "")
feature_selector_model_name = bento_model.info.metadata.get('feature_selector_model_name', "")
use_pca = bento_model.info.metadata.get('use_pca_transform', False)
pca_model_name = bento_model.info.metadata.get('pca_model_name', None)
categories = bento_model.info.metadata.get('categories', {})

bento_runner = bento_model.to_runner()
all_runners = [bento_runner]

if feature_selector != "":
    fs_model = bentoml.models.get("{}:{}".format(feature_selector_model_name, "latest"))
    fs_runner = fs_model.to_runner()
    all_runners.append(fs_runner)

if use_pca:
    pca_model = bentoml.models.get("{}:{}".format(pca_model_name, "latest"))
    pca_runner = pca_model.to_runner()
    all_runners.append(pca_runner)

#create service object
svc = bentoml.Service(bentoml_service_name, runners=all_runners)


@svc.api(input=Multipart(data=Text()), output=PandasDataFrame())
def run_classify(data):
    data_file = StringIO(data)
    data_df = pd.read_csv(data_file, delimiter=";")
    data_df = data_df[[index_col] + indep_vars]
    data_df.set_index(index_col, inplace=True)
    data_df.dropna(how='any', inplace=True)
    uuids = data_df.index.values
    for col_name, cat_names in categories.items():
        LE = LabelEncoder()
        LE.fit(cat_names)
        data_df.loc[:, col_name] = LE.transform(data_df.loc[:, col_name])
    if feature_selector is not None:
        data_df = fs_runner.transform.run(data_df)
    if use_pca:
        data_df = pca_runner.transform.run(data_df)
    labels = bento_runner.predict_proba.run(data_df)[:,1] #probability of default==1
    ret_df = pd.DataFrame(data={'uuid': uuids, 'pd': labels})
    return ret_df

@svc.api(input=JSON(), output=PandasDataFrame())
def run_classify_json(inp_json):    
    data_df = pd.read_json(inp_json)
    data_df = data_df[[index_col] + indep_vars]
    data_df.set_index(index_col, inplace=True)
    data_df.dropna(how='any', inplace=True)
    uuids = data_df.index.values
    for col_name, cat_names in categories.items():
        LE = LabelEncoder()
        LE.fit(cat_names)
        data_df.loc[:, col_name] = LE.transform(data_df.loc[:, col_name])
    if feature_selector is not None:
        data_df = fs_runner.transform.run(data_df)
    if use_pca:
        data_df = pca_runner.transform.run(data_df)
    labels = bento_runner.predict_proba.run(data_df)[:,1] #probability of default==1
    ret_df = pd.DataFrame(data={'uuid': uuids, 'pd': labels})
    
    return ret_df
    
    