#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:35:30 2020

@author: Aron
"""

import os
import re
import requests
import numpy as np
import pandas as pd
import datetime
# from flask import Flask, request


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar




for i in range(1, len(data)):
    
    data.loc[i, 'LAST_CLOSE'] = data.loc[i-1, 'Close']
    print(i)

data['PRICE_DIFF'] = data['Close'] - data['LAST_CLOSE']
data['PRICE_DIFF_RATIO'] = data['PRICE_DIFF'] / data['LAST_CLOSE']
data['LIMIT_UP'] = data['PRICE_DIFF_RATIO'] > 0.095
data['LIMIT_DOWN'] = data['PRICE_DIFF_RATIO'] < -0.095



def initialize(path):

    # 新增工作資料夾
    global path_resource, path_function, path_temp, path_export
    path_resource = path + '/Resource'
    path_function = path + '/Function'
    path_temp = path + '/Temp'
    path_export = path + '/Export'
    
    
    cbyz.create_folder(path=[path_resource, path_function, 
                             path_temp, path_export])        
    return ''




def load_data():
    '''
    讀取資料及重新整理
    '''
    return ''



def master():
    '''
    主工作區
    '''
    
    return ''




def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()




