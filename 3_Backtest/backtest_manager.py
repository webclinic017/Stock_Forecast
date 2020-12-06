#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:09 2020

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
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/3_Backtest'
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




def get_stock_fee():
    
    
    return ''



def master(begin_date, periods=5,
           signal=None, budget=None, split_budget=False):
    '''
    主工作區
    '''
    
    # fee = get_stock_fee()

    
    # Variables    
    # (1) Fix missing date issues
    
    
    time_seq = cbyz.get_time_seq(begin_date=begin_date,
                      periods=periods,
                      unit='m', 
                      simplify_date=True)
   
    
    
   # backtest_multiple()
    backtest_results = pd.DataFrame()    
    
    for i in range(0, len(time_seq)):
        
        single = backtest_single(time_seq.loc[i, 'TIME_UNIT'],
                                 days=60, volume=1000)
        
        backtest_results = backtest_results.append(single)

        
    backtest_results = backtest_results \
                        .reset_index(drop=True)        
    
    
    return ''



def backtest_single(begin_date, days=60, volume=None):
    
    # ........    
    # Bug, add begin_date as arguments
    bts_stock_data = load_data()
    
    
    # bts_stock_data = bts_stock_data[bts_stock_data['STOCK_SYMBOL']=='0050']
    bts_stock_data = bts_stock_data.drop(['HIGH', 'LOW'], axis=1)
    
    
    bts_stock_data = bts_stock_data[
        bts_stock_data['WORK_DATE']>=begin_date] \
                        .reset_index(drop=True)
    
    
    bts_stock_data['FAIL'] = ''
    
    
    unique_symbol = bts_stock_data[['STOCK_SYMBOL']] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    
    
    
    
    backtest_results = pd.DataFrame()
    
    
    
    for j in range(0, 5):
    # for j in range(0, len(unique_symbol)):
        
        cur_symbol = unique_symbol.loc[j, 'STOCK_SYMBOL']
        
        temp = bts_stock_data[bts_stock_data['STOCK_SYMBOL']==cur_symbol] \
                .reset_index(drop=True)
        
        
        for i in range(0, len(temp)):
            
            
            # Replace with model.
            if i == 0:
                temp.loc[i, 'BUY_VOLUME'] = volume
                temp.loc[i, 'TYPE'] = 'BUY'
                buy_price = temp.loc[i, 'CLOSE']
            
        
            # Optimize append
            if temp.loc[i, 'CLOSE'] > buy_price * 1.05:
                temp.loc[i, 'SELL'] = volume
                temp.loc[i, 'TYPE'] = 'SELL'
                backtest_results = backtest_results.append(temp)
                break
            
            # Optimize append
            if i >= days:
                temp.loc[i, 'SELL'] = volume
                temp.loc[i, 'FAIL'] = True
                temp.loc[i, 'TYPE'] = 'SELL'
                backtest_results = backtest_results.append(temp)
                break
            
            print(i)
            
        print('symbole - ' + str(j) + '/' + str(len(unique_symbol)))
        
        
    
    # Organize results ------
    
    backtest_main_pre = backtest_results.copy()
    backtest_main_pre = backtest_main_pre[
                            (~backtest_main_pre['TYPE'].isna())
                            ]
    
    backtest_main_pre['WORK_DATE'] = backtest_main_pre['WORK_DATE'] \
                                        .apply(cbyz.ymd)
    
    
    buy_info = backtest_main_pre[backtest_main_pre['TYPE']=='BUY']
    sell_info = backtest_main_pre[backtest_main_pre['TYPE']=='SELL']
    
    
    backtest_main = buy_info.merge(sell_info, how='outer', 
                                      on='STOCK_SYMBOL')
    
    backtest_main = backtest_main \
                        .drop(['TYPE_x', 'SELL_x', 'FAIL_x',
                               'TYPE_y'], axis=1)
    
    
    backtest_main = backtest_main \
        .rename(columns={'WORK_DATE_x':'WORK_DATE_BUY',
                         'CLOSE_x':'PRICE_BUY',
                         'BUY_VOLUME_x':'VOLUME_BUY',
                         'WORK_DATE_y':'WORK_DATE_SELL',
                         'CLOSE_y':'PRICE_SELL',
                         'BUY_VOLUME_y':'VOLUME_SELL',
                         'SELL_y':'SELL',
                         'FAIL_y':'FAIL'
                         })
    
    # Calculate days ------
    backtest_main['DAYS'] = backtest_main['WORK_DATE_SELL'] \
                                - backtest_main['WORK_DATE_BUY']
    
    
    backtest_main['DAYS'] = backtest_main['DAYS'].astype(str)
    
    backtest_main['DAYS'] = backtest_main['DAYS'] \
                                .str.replace(' days', '')
                                
    backtest_main['DAYS'] = backtest_main['DAYS'].astype(int)
    
    
    
    # Calculate Profits ------
    # Replace VOLUME_BUY with VOLUME_SELL
    backtest_main['PROFIT'] = (backtest_main['PRICE_SELL'] \
                               - backtest_main['PRICE_BUY']) \
                                * backtest_main['VOLUME_BUY']
    
    
    backtest_main['ROI'] = backtest_main['PROFIT'] \
                                / (backtest_main['VOLUME_BUY'] \
                                   * backtest_main['PRICE_BUY'])
    
    
    return backtest_main



def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()














