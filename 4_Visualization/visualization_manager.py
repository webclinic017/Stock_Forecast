#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:48:32 2020

@author: Aron
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/4.Visualization'
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


# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 



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



