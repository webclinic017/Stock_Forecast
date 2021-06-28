#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:09:08 2020

@author: Aron
"""

# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_squared_error
import yfinance as yf 


local = False
local = True

stock_type = 'tw'


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Foreceast/1_Data_Collection/1_download_yahoo_data'
else:
    path = '/home/aronhack/stock_predict/2_Stock_Analysis'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',]


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk




# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])        



# Load Data ----------------

def download_data(stock_list=[], chunk_begin=None, chunk_end=None, 
                  upload=False):
    '''
    讀取資料及重新整理
    '''
   
    global stock_type
    

    # Get stock list
    if len(stock_list) > 0:
        stock_list = pd.DataFrame({'STOCK_SYMBOL':stock_list})
        stock_list['STOCK_SYMBOL'] = stock_list['STOCK_SYMBOL'].astype(str)
    else:
        # stock_list = stk.get_list(stock_type=stock_type)
        # stock_list = stk.tw_get_company_info()
        stock_list = stk.twse_get_data(upload=False)
        stock_list = stock_list[['STOCK_SYMBOL', 'NAME']].drop_duplicates()
    
    
    # Split as chunk ......
    if chunk_begin != None and chunk_end != None:
        stock_list = stock_list.loc[chunk_begin:chunk_end, :]
        
    elif chunk_end != None and chunk_end >= len(stock_list):
        stock_list = stock_list.loc[chunk_begin:len(stock_list), :]
    
    stock_list = stock_list[['STOCK_SYMBOL']].reset_index(drop=True)
    stock_list = cbyz.df_conv_col_type(df=stock_list, cols='STOCK_SYMBOL',
                                       to='str')
    
    
    # Get historical data ......
    hist_data_raw = pd.DataFrame()
    
    
    # for i in range(190, 200):
    for i in range(0, len(stock_list)):
    
        
        # Get stock symbol
        if stock_type == 'tw':
            stock_id = stock_list.loc[i, 'STOCK_SYMBOL'] + '.TW'
            
        elif stock_type == 'us':
            stock_id = stock_list.loc[i, 'STOCK_SYMBOL']
    
    
        # Call API
        try:
            data = yf.Ticker(stock_id)
            df = data.history(period="max")
        except:
            continue
        
        
        if len(df) > 0:
            df['STOCK_SYMBOL'] = stock_list.loc[i, 'STOCK_SYMBOL']
            hist_data_raw = hist_data_raw.append(df, sort=False)


        time.sleep(0.8)
        print(str(i) + '/' + str(len(stock_list)))
    
    
    # Organize data ......
    global hist_data
    hist_data = hist_data_raw.copy()
    hist_data = hist_data.reset_index()
    hist_data['Date'] = hist_data['Date'].astype('str')
    
    # There are some na values.
    hist_data = hist_data[~hist_data['Open'].isna()]
    
    
    # Rename
    hist_data = hist_data \
                .rename(columns={'Date':'WORK_DATE',
                                 'Open':'OPEN',
                                 'High':'HIGH',
                                 'Low':'LOW',
                                 'Close':'CLOSE',
                                 'Volume':'VOLUME'})
    
    hist_data = hist_data[['WORK_DATE', 'STOCK_SYMBOL', 'OPEN', 
                           'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
    
    
    # Upload ......
    if upload:
        if stock_type == 'tw':
            ar.db_upload(data=hist_data, 
                         table_name='stock_data_tw',
                         local=local)
            
        elif stock_type == 'us':
            ar.db_upload(data=hist_data, 
                         table_name='stock_data_us',
                         local=local)

    else:
        hist_data = hist_data.to_csv(path_temp + '/hist_data_' \
                                     + str(chunk_begin) + '_' \
                                     + str(chunk_end) + '.csv',
                                     index=False)



def master():
    '''
    主工作區
    '''
    
    # Delete old data
    sql = "truncate table stock_data_tw"
    ar.db_execute(sql, local=True, fetch=False)
    
    
    # download_data(stock_list)
    download_data(stock_list=[], chunk_begin=None, chunk_end=None, 
                  upload=True)
    
    
    stock_list = stk.get_list(stock_type=stock_type)    
    
    
    chk = stock_list[['STOCK_SYMBOL']].drop_duplicates()
    
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


