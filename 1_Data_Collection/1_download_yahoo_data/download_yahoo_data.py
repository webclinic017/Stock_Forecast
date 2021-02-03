#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:09:08 2020

@author: Aron
"""

import os, sys
import re
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date
# from flask import Flask, request
import time
import h5py
import yfinance as yf # https://pypi.org/project/yfinance/


# 設定工作目錄 .....
path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'

# Codebase
path_codebase = ['/Users/Aron/Documents/GitHub/Arsenal',
                 '/Users/Aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import arsenal as ar
import codebase_yz as cbyz



stock_type = 'tw'
# stock_type = 'us'

local = False
local = True


# Load Data ----------------

# 自動設定區 -------
pd.set_option('display.max_columns', 30)


def init(path):
    
    global path_resource, path_function, path_temp, path_export
    
    # 新增工作資料夾
    path_resource = path + '/Resource'
    path_function = path + '/Function'
    path_temp = path + '/Temp'
    path_export = path + '/Export'
    
    
    cbyz.os_create_folder(path=[path_resource, path_function, 
                             path_temp, path_export])



def load_data(stock_list=None):
    '''
    讀取資料及重新整理
    '''
   
    # Get stock list
    
    
    if stock_list == None:
        stock_list = ar.stk_get_list(stock_type=stock_type)
    else:
        stock_list = pd.DataFrame({'STOCK_SYMBOL':[2474, 1714, 2385]})
        stock_list['STOCK_SYMBOL'] = stock_list['STOCK_SYMBOL'].astype(str)
    
    
    begin_time = datetime.now()   
    
    # Get historical data
    hist_data_raw = pd.DataFrame()
    
    # for i in range(0, 10):
    for i in range(0, len(stock_list)):
    # for i in range(190, 200):
    
        # Get data
        if stock_type == 'tw':
            stock_id = stock_list.loc[i, 'STOCK_SYMBOL'] + '.TW'
        elif stock_type == 'us':
            stock_id = stock_list.loc[i, 'STOCK_SYMBOL']
    
        data = yf.Ticker(stock_id)
        df = data.history(period="max")
        
        df['STOCK_SYMBOL'] = stock_list.loc[i, 'STOCK_SYMBOL']
        
        if len(df) > 0:
            hist_data_raw = hist_data_raw.append(df, sort=False)


        time.sleep(0.8)
        print(i)
    
    
    end_time = datetime.now() 
    print(end_time - begin_time)

    
    hist_data = hist_data_raw.copy()
    hist_data.reset_index(level=0, inplace=True)
    hist_data['Date'] = hist_data['Date'].astype('str')
    
    
    # Rename
    hist_data = hist_data \
                .rename(columns={'Date':'WORK_DATE',
                                 'Open':'OPEN',
                                 'High':'HIGH',
                                 'Low':'LOW',
                                 'Close':'CLOSE',
                                 'Volume':'VOLUME'})
    
    
    # Filter columns
    cols = ['WORK_DATE', 'STOCK_SYMBOL', 'OPEN', 
            'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    
    hist_data = hist_data[cols]
    
    
    
    # Upload
    if stock_type == 'tw':
        ar.db_upload(data=hist_data, 
                     table_name='stock_data_tw',
                     local=local)
        
    elif stock_type == 'us':
        ar.db_upload(data=hist_data, 
                     table_name='stock_data_us',
                     local=local)

    
    # Export
    global path_export
    time_seq = cbyz.get_time_serial(with_time=True)
    hist_data.to_hdf(path_export + '/yahoo_finance_data_'+time_seq+'.h5', 
                     key='s')
    
    hist_data.to_csv(path_export + '/yahoo_finance_data_'+time_seq+'.csv', 
                     index=False)
    
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


def get_us_stock_data():

        
    # US Stock --------
    
    results = pd.DataFrame()
    
    for i in us_stock:
        
        data = yf.Ticker(i)
        df = data.history(period="max")
        df['STOCK_SYMBOL'] = i
    
        results = results.append(df)
            
    
    results.reset_index(level=0, inplace=True)
    
    
    # Rename
    results.rename(columns={'Date':'WORK_DATE',
                                'Open':'OPEN',
                                'High':'HIGH',
                                'Low':'LOW',
                                'Close':'CLOSE',
                                'Volume':'VOLUME'
                                    }, 
                           inplace=True)
    
    # Filter columns
    cols = ['WORK_DATE', 'STOCK_SYMBOL', 'OPEN', 
            'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    results = results[cols]
    
    
    Failed processing format-parameters; Python 'timestamp' cannot be converted to a MySQL type
    
    # results['WORK_DATE'] = results['WORK_DATE'].apply(cbyz.ymd)
    results['WORK_DATE'] = results['WORK_DATE'].astype('str')
    
    ar.db_upload(results,
                 'stock_data_us')

    
    return ''


