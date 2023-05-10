# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import bentoml
import argparse
import ast
from model_configs import ALL_MODELS
from vars_config import VARS_COMBOS
from bentoml_store_config import BENTOML_STORE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
    
class DEFAULT_PREDICTOR_VANILLA(object):
    GLOBAL_PATH = 'C:\\Users\\vivin\\Spyder projects\\def_prediction\\'
    IMBLEARN_RANDOM_STATE = 0
    
    def __init__(self, model_config_name, var_config_name, bentoml_config_name, dataset_filename, test_fraction=0.1, refit=False, save_model=True):
        self.model_config = ALL_MODELS.get(model_config_name, None)
        if self.model_config is None:
            raise AttributeError("No model configuration with the specified name found")
        self.model_name = self.model_config.get("model", None)
        self.model_params = self.model_config.get("params", {})
        self.base_model_params = self.model_config.get("base_model_params", {})
        if self.model_name is None:
            raise AttributeError("specified model config has no model name")
        self.dataset_filename = dataset_filename
        self.test_fraction = test_fraction
        self.refit = refit
        self.save_model = save_model
        
        #bentoml configs
        self.bentoml_config = BENTOML_STORE.get(bentoml_config_name, None)
        if self.bentoml_config is None:
            raise AttributeError("please specify a correct bentoml config name")
        self.bentoml_model_name = self.bentoml_config.get("model_name", None)
        self.bentoml_model_class = self.bentoml_config.get("model_class", None)
        self.bentoml_model_version = self.bentoml_config.get("version", "latest")
        if self.bentoml_model_name is None or self.bentoml_model_class is None:
            raise AttributeError("config specified is missing either the model name or model class")
        self.__bentoml_model_consistency_check()
        
        # variable names
        self.var_config = VARS_COMBOS.get(var_config_name, None)
        if self.var_config is None:
            raise AttributeError("No variable config with the specified name found")
        self.X_cols = self.var_config.get("indep_vars", None) #X_col can be none. it means that we need to consider all the available columns as indep vars
        self.y_col = self.var_config.get("dep_var", None)
        self.sample_weights = self.var_config.get("sample_weights", None)
        self.__sample_weights_consistency_check()
        self.index_col = self.var_config.get("index", None)
        self.na_fill_cols = self.var_config.get("na_fill_cols", {})
        if self.y_col is None:
            raise AttributeError("specified variable config has no dep vars specified")
        self.imbalance_handling = self.var_config.get("imbalance_handling", None)
        self.categories_convert = self.var_config.get("categories_convert", [])
        self.categories = {}
        self.use_pca_transform = self.var_config.get("use_pca_transform", False)
        self.pca_model = None
        
        #test and train data
        self.load_train_test_data()
        
        #model params
        
    def __sample_weights_consistency_check(self):
        if type(self.sample_weights) not in [np.ndarray, list]:
            self.sample_weights = None
        if self.sample_weights is not None and self.X_cols is not None and len(self.sample_weights) != len(self.X_cols):
            raise AttributeError("sample weights are not the same length as the feature set size")
        if type(self.sample_weights) in [np.ndarray, list] and self.X_cols is None:
            raise AttributeError("Youve explicitly specified the weights but not explicitly specified the indep var names. This is dangerous")
        
    def __bentoml_model_consistency_check(self):
        if self.bentoml_model_class == 'sklearn' and self.model_name == "xgboost":
            raise AttributeError("bentoml_model_class and model_names are inconsistent")
            
    def __imbalance_handling(self):
        if self.imbalance_handling is not None:
            if self.imbalance_handling == "over-sample":
                from imblearn.over_sampling import RandomOverSampler
                ros = RandomOverSampler(random_state=self.IMBLEARN_RANDOM_STATE)
                self.X_train, self.y_train = ros.fit_resample(self.X_train, self.y_train)
            elif self.imbalance_handling == "under-sample":
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=self.IMBLEARN_RANDOM_STATE)
                self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
            else:
                raise AttributeError("incorrect imbalance handling method specified")
                
    def __variable_dim_reduction(self):
        if self.use_pca_transform:
            #hardcode variances for now
            from sklearn.decomposition import PCA
            clf = PCA(0.95, whiten=True, svd_solver='full')
            X_train_new = clf.fit_transform(self.X_train)
            #X_cols_pca = np.arange(1, len(X_train_new.columns.values), 1)
            self.X_train = X_train_new
            self.pca_model = clf
            
    def load_train_test_data(self):
        df_raw = pd.read_csv('{}datasets\\{}.csv'.format(self.GLOBAL_PATH, self.dataset_filename), delimiter=";")
        #cleaning: remove nan from predict output
        df_raw = df_raw[~np.isnan(df_raw[self.y_col])]
        if self.index_col is not None:
            df_raw.set_index(self.index_col, inplace=True)
        #cleaning: some other cleaning of other columns - make another function out of this
        for col_name, fill_val in self.na_fill_cols.items():
            df_raw.loc[:, col_name].fillna(fill_val, inplace=True)
        #convert string columns to categories
        for col_name in self.categories_convert:
            LE = LabelEncoder()
            df_raw.loc[:, col_name] = LE.fit_transform(df_raw.loc[:, col_name])
            self.categories[col_name] = LE.classes_

        all_potential_indep_var_cols = list(df_raw.columns.values)
        all_potential_indep_var_cols.remove(self.y_col)
        self.X_cols = all_potential_indep_var_cols if self.X_cols is None else self.X_cols
        X = df_raw[self.X_cols]
        y = df_raw[self.y_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_fraction, random_state=100)
        self.__imbalance_handling()
        self.__variable_dim_reduction()
        
    def fit(self):
        if self.refit:
            print("generating a fully re-fitted model from scratch")
            self.construct_model()
        else:
            print("loading latest fitted model from store")
            if self.bentoml_model_class == "sklearn":
                self.model = bentoml.sklearn.load_model("{}:{}".format(self.bentoml_model_name, self.bentoml_model_version))
            elif self.bentoml_model_class == "xgboost":
                self.model = self.model = bentoml.xgboost.load_model("{}:{}".format(self.bentoml_model_name, self.bentoml_model_version))
            else:
                raise AttributeError("unknown bentoml_model_class")
        
        self.__bentoml_save_model()
                
    def __bentoml_save_model(self):
        if self.refit and self.save_model:
            print("saving-down the model as a bentoml object")
            pca_model_name = "PCA_{}".format(self.bentoml_model_name)
            metadata = {
                'indep_vars': self.X_cols, 
                'dep_var': self.y_col, 
                'index_col': self.index_col,
                'categories': self.categories,
                'use_pca_transform': self.use_pca_transform,
                'pca_model_name': pca_model_name if self.use_pca_transform else None
            }
            if self.bentoml_model_class == "sklearn":
                _ = bentoml.sklearn.save_model(self.bentoml_model_name, self.model, metadata=metadata, signatures={'predict_proba': {}, 'predict': {}})
            elif self.bentoml_model_class == "xgboost":
                _ = bentoml.xgboost.save_model(self.bentoml_model_name, self.model, metadata=metadata, signatures={'predict_proba': {}, 'predict': {}})
            else:
                raise AttributeError("unknown bentoml_model_class")
                
            if self.use_pca_transform:
                #save-down the pca model as well
                _ = bentoml.sklearn.save_model(pca_model_name, self.pca_model, metadata=metadata, signatures={'transform': {}})
                

    def construct_model(self):
        train_data = (self.X_train, self.y_train, self.sample_weights) if self.sample_weights is not None else (self.X_train, self.y_train)
        if self.model_name == 'DecisionTreeClassifier':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(**self.model_params).fit(*train_data)
        elif self.model_name == 'SVC':
            from sklearn.svm import SVC
            model = SVC(**self.model_params).fit(*train_data)
        elif self.model_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier 
            model = KNeighborsClassifier(**self.model_params).fit(*train_data)
        elif self.model_name == 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB(**self.model_params).fit(*train_data)
        elif self.model_name == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**self.model_params).fit(*train_data)
        elif self.model_name == 'GradientBoostingClassifier':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**self.model_params).fit(*train_data)
        elif self.model_name == 'BaggedKNN':
            from sklearn.ensemble import BaggingClassifier
            from sklearn.neighbors import KNeighborsClassifier
            model = BaggingClassifier(KNeighborsClassifier(**self.base_model_params), **self.model_params).fit(*train_data)
        elif self.model_name == 'AdaBoostedTree':
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            model = AdaBoostClassifier(DecisionTreeClassifier(**self.base_model_params), **self.model_params).fit(*train_data)
        elif self.model_name == 'XGBoostClassifier':
            from xgboost import XGBClassifier
            model = XGBClassifier(**self.model_params).fit(*train_data)
        elif self.model_name == 'RidgeClassifier':
            from sklearn.linear_model import RidgeClassifier
            model = RidgeClassifier(**self.model_params).fit(*train_data)
        elif self.model_name == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**self.model_params).fit(*train_data)
        elif self.model_name == 'LDA':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis(**self.model_params).fit(*train_data)
        elif self.model_name == 'QDA':
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            model = QuadraticDiscriminantAnalysis(**self.model_params).fit(*train_data)
        elif self.model_name == 'MLP':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(**self.model_params).fit(*train_data)
        elif self.model_name == 'NearestCentroid':
            from sklearn.neighbors import NearestCentroid
            model = NearestCentroid(**self.model_params).fit(*train_data)
        elif self.model_name == 'RadiusNeighborsClassifier':
            from sklearn.neighbors import RadiusNeighborsClassifier
            model = RadiusNeighborsClassifier(**self.model_params).fit(*train_data)
        else:
            raise ValueError("unknown ML model passed in model_name")
        
        self.model = model
        
    def get_model_metrics(self):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
        y_pred = self.model_predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        #y_pred_probab = self.model.predict_proba(self.X_test)
        print("Accuracy:", accuracy)
        print("balanced accuracy:", balanced_accuracy)
        print("Precision:", precision)
        print("recall:", recall)
        
    def model_predict(self, X_test):
        if self.use_pca_transform:
            X_test_trans = self.pca_model.transform(X_test)
            y_pred = self.model.predict(X_test_trans)
        else:
            y_pred = self.model.predict(X_test)
        return y_pred
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse arguments')
    parser.add_argument('--model_config_name', default='random_forest')
    parser.add_argument('--var_config_name', default='all_indeps_dsample')
    parser.add_argument('--bentoml_config_name', default='default_predict_with_pca')
    parser.add_argument('--dataset_filename', default='dataset')
    parser.add_argument('--test_fraction', default=0.1, type=float)
    parser.add_argument('--refit', default='False', type=str)
    parser.add_argument('--save_model', default='False', type=str)
    args = parser.parse_args()
    
    predictor = DEFAULT_PREDICTOR_VANILLA(
        model_config_name=args.model_config_name,
        var_config_name=args.var_config_name,
        bentoml_config_name=args.bentoml_config_name,
        dataset_filename=args.dataset_filename,
        test_fraction=args.test_fraction,
        refit=ast.literal_eval(args.refit),
        save_model=ast.literal_eval(args.save_model),
    )
    predictor.fit()
    predictor.get_model_metrics()
        
        