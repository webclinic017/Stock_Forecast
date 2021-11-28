#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:05:39 2021

@author: Aron
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc


local = False
local = True


# Path .....
if local:
    # path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/1_Data_Collection/1_Stock_Ratio'
    path = r'D:\Data_Mining\Projects\Codebase_YZ\arsenal'
else:
    path = '/home/aronhack/stock_forecast/dashboard/'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ/',
                 path + 'Function/']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk




# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

# 新增工作資料夾
global path_resource, path_function, path_temp, path_export
path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])      





data = pd.read_csv(path + '/stock_ratio.csv')

data = data \
    .drop('序', axis=1) \
    .rename(columns={'持股/單位數分級':'LEVEL',
                     '人數':'HOLDER',
                     '股數/單位數':'STOCK',
                     '占集保庫存數比例 (%)':'RATIO'})

data['STOCK_SYMBOL'] = data['STOCK_SYMBOL'].str.replace('ID_', '')


# Out[23]: 
#    COLUMN  NA_COUNT
# 0       序      2555
# 1   LEVEL         2
# 2  HOLDER      3655
# 3   STOCK      2566
# 4   RATIO      2567
cbyz.df_chk_col_na(df=data)    


data = cbyz.df_conv_na(df=data, cols=['LEVEL'], value='NA')
data = data[(~data['LEVEL'].str.contains('計')) \
            & (data['LEVEL']!='NA')]

# Process LEVEL column ...
data = cbyz.df_conv_na(df=data, cols=['LEVEL'], value='NA')    

# 1. 20211112 - 大約1093列有「差異數調整」，先移除，但不確定是否有重大影響
# 2. 目前確定爬蟲爬下來的可能會有「差異數調整」，不確定直接下載csv的會不會有
data = data[~data['LEVEL'].str.contains('差異數調整')]

# 20211112 - 執行完這個步驟後，2562列只剩下個位數
data = data[data['LEVEL']!='Null']

# Process HOLDER / STOCK / RATIO column ...
data = cbyz.df_conv_na(df=data, cols=['HOLDER', 'STOCK', 'RATIO'], value=0)



# Analyze ......
data = data \
            .copy() \
            .sort_values(by=['STOCK_SYMBOL', 'LEVEL', 'WORK_DATE']) \
            .reset_index(drop=True)

data, _ = cbyz.df_add_shift(df=data, 
                            cols=['HOLDER', 'RATIO'], shift=1, 
                            group_by=['STOCK_SYMBOL', 'LEVEL'], 
                            suffix='_PREV', remove_na=False)


data['RATIO_CHANGE'] = abs(data['RATIO'] - data['RATIO_PREV']) \
                        / data['RATIO_PREV']

data['RATIO_PER_HOLDER'] = data['RATIO'] / data['HOLDER']

data.tail(10)
