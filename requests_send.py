# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:11:12 2023

@author: vivin
"""

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


endpoint = "https://bxq98dklyj.execute-api.us-west-1.amazonaws.com/run_classify"
#endpoint = "http://127.0.0.1:3000/run_classify"
csv_file_loc = "C:\\Users\\vivin\\Downloads\\dataset_short.csv"
predict_model = "pd_predict_rf_ds" #pd_predict_rf_us, pd_predict_rf_ds_seq5, pd_predict_rf_ds_seq10


multipart_data = MultipartEncoder(fields={"data": ("filename", open(csv_file_loc, "rb"), "text/csv"), "model_tag": predict_model})
response = requests.post(endpoint, headers={"content-type": multipart_data.content_type}, data=multipart_data)
