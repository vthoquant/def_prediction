# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:17:11 2023

@author: vivin
"""

import bentoml
import pandas as pd
import numpy as np
from io import StringIO
from bentoml.io import Text, PandasDataFrame, Multipart
from sklearn.preprocessing import LabelEncoder

bentoml_service_name = 'pd_predictor_rf_multi'
bentoml_model_names_and_versions = {
    "pd_predict_rf_ds": "latest",
    "pd_predict_rf_us": "latest",
    "pd_predict_rf_ds_seq5": "latest",
    "pd_predict_rf_ds_seq10": "latest"
}

bento_model_by_modelname = {}
indep_vars_by_modelname = {}
dep_var_by_modelname = {}
index_col_by_modelname = {}
na_fill_cols_by_modelname = {}
feature_selector_tag_by_modelname = {}
feature_selector_model_name_by_modelname = {}
use_pca_by_modelname = {}
pca_model_name_by_modelname = {}
categories_by_modelname = {}
bento_runner_by_modelname = {}
fs_runner_by_modelname = {}
pca_runner_by_modelname = {}
all_runners = []

for bentoml_model_name, bentoml_model_version in bentoml_model_names_and_versions.items():
    bento_model = bentoml.models.get("{}:{}".format(bentoml_model_name, bentoml_model_version))
    bento_model_by_modelname[bentoml_model_name] = bento_model
    indep_vars_by_modelname[bentoml_model_name] = bento_model.info.metadata['indep_vars']
    dep_var_by_modelname[bentoml_model_name] = bento_model.info.metadata['dep_var']
    index_col_by_modelname[bentoml_model_name] = bento_model.info.metadata['index_col']
    na_fill_cols_by_modelname[bentoml_model_name] = bento_model.info.metadata['na_fill_cols']
    feature_selector = bento_model.info.metadata.get('feature_selector', "")
    feature_selector_model_name = bento_model.info.metadata.get('feature_selector_model_name', "")
    feature_selector_tag_by_modelname[bentoml_model_name] = feature_selector
    feature_selector_model_name_by_modelname[bentoml_model_name] = feature_selector_model_name
    use_pca = bento_model.info.metadata.get('use_pca_transform', False)
    pca_model_name = bento_model.info.metadata.get('pca_model_name', None)
    use_pca_by_modelname[bentoml_model_name] = use_pca
    pca_model_name_by_modelname[bentoml_model_name] = pca_model_name
    categories_by_modelname[bentoml_model_name] = bento_model.info.metadata.get('categories', {})
    bento_runner = bento_model.to_runner()
    bento_runner_by_modelname[bentoml_model_name] = bento_runner
    all_runners.append(bento_runner)

    if feature_selector != "":
        fs_model = bentoml.models.get("{}:{}".format(feature_selector_model_name, "latest"))
        fs_runner = fs_model.to_runner()
        all_runners.append(fs_runner)
        fs_runner_by_modelname[bentoml_model_name] = fs_runner

    if use_pca:
        pca_model = bentoml.models.get("{}:{}".format(pca_model_name, "latest"))
        pca_runner = pca_model.to_runner()
        all_runners.append(pca_runner)
        pca_runner_by_modelname[bentoml_model_name] = pca_runner
        
#create service object
svc = bentoml.Service(bentoml_service_name, runners=all_runners)


@svc.api(input=Multipart(data=Text(), model_tag = Text()), output=PandasDataFrame())
def run_classify(data, model_tag):
    data_file = StringIO(data)
    data_df = pd.read_csv(data_file, delimiter=";")
    ret_df = predict_on_input_df(data_df, model_tag, index_col_by_modelname, indep_vars_by_modelname, dep_var_by_modelname, na_fill_cols_by_modelname, categories_by_modelname, use_pca_by_modelname, feature_selector_tag_by_modelname, bento_runner_by_modelname, fs_runner_by_modelname, pca_runner_by_modelname)
    
    return ret_df


def predict_on_input_df(data_df, model_tag, index_col_by_modelname, indep_vars_by_modelname, dep_var_by_modelname, na_fill_cols_by_modelname, categories_by_modelname, use_pca_by_modelname, feature_selector_tag_by_modelname, bento_runner_by_modelname, fs_runner_by_modelname, pca_runner_by_modelname):
    index_col = index_col_by_modelname[model_tag]
    indep_vars = indep_vars_by_modelname[model_tag]
    dep_var = dep_var_by_modelname[model_tag]
    na_fill_cols = na_fill_cols_by_modelname[model_tag]
    categories = categories_by_modelname[model_tag]
    use_pca = use_pca_by_modelname[model_tag]
    feature_selector = feature_selector_tag_by_modelname[model_tag]
    #only predict on those datapoints where the dep_var is null/NA
    data_df = data_df[np.isnan(data_df[dep_var])]
    data_df = data_df[[index_col] + indep_vars]
    data_df.set_index(index_col, inplace=True)
    data_clean(data_df, na_fill_cols)
    uuids = data_df.index.values
    for col_name, cat_names in categories.items():
        LE = LabelEncoder()
        LE.fit(cat_names)
        data_df.loc[:, col_name] = LE.transform(data_df.loc[:, col_name])
    if feature_selector != "":
        data_df = fs_runner_by_modelname[model_tag].transform.run(data_df)
    if use_pca:
        data_df = pca_runner_by_modelname[model_tag].transform.run(data_df)
    labels = bento_runner_by_modelname[model_tag].predict_proba.run(data_df)[:,1] #probability of default==1
    ret_df = pd.DataFrame(data={'uuid': uuids, 'pd': labels})
    
    return ret_df

def data_clean(data_df, na_fill_cols):
    for col_name, fill_val in na_fill_cols.items():
        data_df.loc[:, col_name].fillna(fill_val, inplace=True)
    
    data_df.dropna(how='any', inplace=True)