


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc


host = 0


# Path .....
if host == 0:
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis/debug'
else:
    path = '/home/aronhack/stock_forecast/dashboard'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk
import ultra_tuner_v0_25_dev as ut



# 自動設定區 -------
path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     

pd.set_option('display.max_columns', 30)
 

model_data = pd.read_csv(path + '/model_data.csv')
model_data = cbyz.df_conv_col_type(df=model_data, cols='SYMBOL',
                                   to='str')

model_data = model_data[model_data['SYMBOL'].str.slice(0, 3)=='260']



model_data = model_data \
    .drop(['OPEN_CHANGE_RATIO', 'HIGH_CHANGE_RATIO', 
           'LOW_CHANGE_RATIO'], axis=1)



# Training Model ......
import xgboost as xgb
# from sklearn.model_selection import GridSearchCV    
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor    

model_params = [
                {'model': LinearRegression(),
                  'params': {
                      'normalize': [True, False],
                      }
                  },
                {'model': xgb.XGBRegressor(),
                 'params': {
                    # 'n_estimators': [200],
                    'eta': [0.05],
                    'min_child_weight': [1],
                    'max_depth':[8],
                    'subsample':[0.8]
                  }
                },
                {'model': SGDRegressor(),
                  'params': {
                      # 'max_iter': [1000],
                      # 'tol': [1e-3],
                      # 'penalty': ['l2', 'l1'],
                      }                     
                  }
               ] 

# 1. 如果selectkbest的k設得太小時，importance最高的可能都是industry，導致同產業
#    的預測值完全相同
global pred_result, pred_scores, pred_params, pred_features

    

tuner = ut.Ultra_Tuner(id_keys=['WORK_DATE', 'SYMBOL'],
                       y='CLOSE_CHANGE_RATIO', 
                       model_type='reg', suffix='',
                       compete_mode=1,
                       train_mode=0, 
                       path=path)


return_result, return_scores, return_params, return_features, \
        log_scores, log_params, log_features = \
            tuner.fit(data=model_data, model_params=model_params,
                      k='all', cv=2, threshold=500, 
                      norm_orig=[],
                      export_model=True, export_log=True)



tuner.greater_is_better


return_scres是linear regression，但最好的是XGB
檢查saved_model是哪一個

import pickle

model_path = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis/debug/reg_model_close_change_ratio_20211231_004339.sav'
saved_model = pickle.load(open(model_path, 'rb'))








