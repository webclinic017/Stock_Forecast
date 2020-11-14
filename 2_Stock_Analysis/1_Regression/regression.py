#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:04:08 2020

@author: Aron
"""

"""
Version Note
1. Add dropdown
"""


# %% 讀取套件 -------

import pandas as pd
import numpy as np
import sys, time, os, gc
import arrow
import statsmodels
import statsmodels.api as sm


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'


# Codebase
path_codebase = ['/Users/Aron/Documents/GitHub/Arsenal',
                 '/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import arsenal as ar
import codebase_yz as cbyz



# 手動設定區 -------
end_date = arrow.now()
begin_date = end_date.shift(months=-6)
# begin_date = end_date.shift(years=-1)


end_date = end_date.format('YYYYMMDD')
end_date = int(end_date)

begin_date = begin_date.format('YYYYMMDD')
begin_date = int(begin_date)

# stock_type = 'us'
stock_type = 'tw'



# 自動設定區 -------
pd.set_option('display.max_columns', 30)


# 新增工作資料夾
path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])



def init(path):
    
    return ''


def load_data():
    '''
    讀取資料及重新整理
    '''

    # Load Data --------------------------
    
    # Historical Data .....
    stock_data = ar.get_stock_data(begin_date, end_date, 
                                stock_type=stock_type,
                                local=local)
    
    # Stock Name .....
    global stock_list
    if stock_type == 'tw':
        stock_list = ar.stock_get_list(stock_type=stock_type)
    elif stock_type == 'us':
        stock_list = ar.stock_get_list(stock_type=stock_type)


    # Work Area -------------
    global main_data
    main_data = stock_data.copy()
    main_data = main_data.sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                .reset_index(drop=True)
    
    main_data = main_data.merge(stock_list, 
                                how='left', 
                                on=['STOCK_SYMBOL'])
    
    # main_data['WORK_DATE'] = main_data['WORK_DATE'].apply(ar.ymd)

    main_data['STOCK'] = 'STOCK_' \
                          + main_data['STOCK_SYMBOL'] 

    
    return ''




def modeling(stock=['0050', '0056']):
    '''
    主工作區
    '''
    
    global main_data
    
    
    model_data = main_data[main_data['STOCK_SYMBOL'].isin(stock)]
    model_data = pd.pivot_table(model_data,
                                values='CLOSE',
                                index='WORK_DATE',
                                columns='STOCK')
    
    model_data = model_data.reset_index()
    model_data = model_data[(~model_data['STOCK_'+stock[0]].isna()) &
                            (~model_data['STOCK_'+stock[1]].isna())]
    
    
    # 
    stock1 = model_data['STOCK_'+stock[0]].reset_index(drop=True)
    stock1 = stock1.values
    
    stock2 = model_data['STOCK_'+stock[1]].reset_index(drop=True)
    stock2 = stock2.values
    
    
    # 
    x = stock1
    x = sm.add_constant(x)
    y = stock2

    model = sm.OLS(y, x)    
    results = model.fit()
    results.params
    
    results.tvalues

    # print(results.t_test([1, 0]))
    
    return ''


def master():
    
    load_data()
    
    global stock_list
    cross_df = stock_list[['STOCK_SYMBOL']]
    cross_stock_list = ar.df_cross_join(cross_df, 
                                     cross_df)
    
    
    fit_results = pd.DataFrame()
    
    for i in range(0, 100):
        
        cross_stock_list['STOCK_SYMBOL_x'][i]
        cross_stock_list['STOCK_SYMBOL_y'][i]
        
        temp = modeling()
        fit_results = fit_results.append(temp)
    
    
    return ''



def check():
    '''
    資料驗證
    '''    
    return ''



if __name__ == '__main__':
    master()













