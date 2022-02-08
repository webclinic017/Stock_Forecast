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
    path = '/Users/aron/Documents/GitHub/Arsenal/Dev'
else:
    path = '/home/aronhack/stock_forecast/dashboard'


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/Users/aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk



# 自動設定區 -------
path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     

pd.set_option('display.max_columns', 30)
 


data = stk.get_data(data_begin=20220107, data_end=20220211, market='tw', 
                    restore=True,
             unit='d', symbol=[], adj=True, price_change=False,
             price_limit=True, trade_value=False)

data



