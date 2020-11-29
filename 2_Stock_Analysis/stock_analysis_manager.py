#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:08 2020

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
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/2_Stock_Analysis'
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
    
    
    target = ar.stk_get_list(stock_type='tw', 
                          stock_info=False, 
                          update=False,
                          local=True)    
    # Dev
    target = target.iloc[0:10, :]
    target = target['STOCK_SYMBOL'].tolist()
    
    data_raw = ar.stk_get_data(begin_date=None, end_date=None, 
                   stock_type='tw', stock_symbol=target, 
                   local=True)    
    
    return data_raw




def get_hold_stock():
    
    
    return ''



def get_buy_signal(data, hold_stocks=None):
    
    loc_data = data.copy()
    loc_data['SIGNAL_THRESHOLD'] = loc_data.groupby('STOCK_SYMBOL')['CLOSE'] \
        .transform('max')

    loc_data['SIGNAL_THRESHOLD'] = loc_data['SIGNAL_THRESHOLD'] * 0.95        
        
    loc_data['PREV_PRICE'] = loc_data \
                            .groupby(['STOCK_SYMBOL'])['CLOSE'] \
                            .shift(-1) \
                            .reset_index(drop=True)        
        
     
        
    loc_data.loc[loc_data['PREV_PRICE'] > loc_data['SIGNAL_THRESHOLD'], 
                 'SIGNAL'] = True
    
    loc_data = loc_data[loc_data['SIGNAL']==True] \
            .reset_index(drop=True)
    
    return loc_data


# ..............


def get_sell_signal(data, hold_stocks=None):
    
    return ''


def analyze_center(data):
    '''
    List all analysis here
    '''    
    
    analyze_results = get_top_price(data)
    
    
    
    # Results format
    # (1) Only stock passed test will show in the results
    # STOCK_SYMBOL
    # ANALYSIS_ID
    
    
    
    return analyze_results



# %% Dev ---------
    

def get_top_price(data):
    '''
    Dev
    '''
    
    # data = stock_data.copy()
    loc_data = data.copy()
    # loc_data = loc_data[['STOCK_SYMBOL', 'CLOSE']]
    
    top_price = data.groupby('STOCK_SYMBOL')['CLOSE'] \
                .aggregate(max) \
                .reset_index() \
                .rename(columns={'CLOSE':'MAX_PRICE'})

        
    results_pre = loc_data.merge(top_price, 
                         how='left', 
                         on='STOCK_SYMBOL')
    
    
    # Temp ---------
    # Add a test here
    cur_price = results_pre \
        .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE'],
                     ascending=[True, False]) \
        .drop_duplicates(subset=['STOCK_SYMBOL'])
        
    cur_price = cur_price[['STOCK_SYMBOL', 'CLOSE']] \
        .rename(columns={'CLOSE':'CUR_PRICE'})
     
    # Reorganize -------        
    results = top_price.merge(cur_price, 
                         on='STOCK_SYMBOL')
    
    
    results.loc[results['CUR_PRICE'] > results['MAX_PRICE'] * 0.95,
                'BUY_SIGNAL'] = True
    
    
    results.loc[results['CUR_PRICE'] < results['MAX_PRICE'] * 0.3,
                'SELL_SIGNAL'] = True
    
    
    
    results = results[(~results['BUY_SIGNAL'].isna()) |
                      (~results['SELL_SIGNAL'].isna())] \
        .reset_index(drop=True)

        
    results['ANALYSIS_ID'] = 'TM01'
    results = results[['STOCK_SYMBOL', 'ANALYSIS_ID',
                       'BUY_SIGNAL', 'SELL_SIGNAL']]

    return results   



# %% Master ------
    

# 停損點
def master(today=None, hold_stocks=None, roi=10, limit=90):
    '''
    主工作區
    roi:     percent
    limit:   days
    '''
    
    global stock_data
    stock_data = load_data()
    
    global analyze_results
    analyze_results = analyze_center(data=stock_data)
    
    
    # v0
    # buy_signal = get_buy_signal(data=stock_data,
    #                             hold_stocks=hold_stocks)
    
    # sell_signal = get_sell_signal(data=analyze_results,
    #                               hold_stocks=hold_stocks)
    
    # master_results = {'RESULTS':analyze_results,
    #                   'BUY_SIGNAL':buy_signal,
    #                   'SELL_SIGNAL':sell_signal}
    
    return analyze_results



def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()






