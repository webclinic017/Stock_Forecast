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
    master_path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'
    master_path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 master_path + '/2_Stock_Analysis',]


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import stock_analysis_manager as sam

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




def load_data(begin_date, end_date=end_date):
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
    
    data_raw = ar.stk_get_data(begin_date=begin_date, 
                               end_date=end_date, 
                               stock_type='tw', stock_symbol=target,
                               local=True)    
    
    return data_raw




def get_stock_fee():
    
    
    return ''


# ..........
    
begin_date = 20190301
days=60
volume=None
budget=None


def backtest_single(begin_date, days=60, volume=None, budget=None):
    
   
    # ........    
    # Bug, 股市休盤space issues
    end_date = cbyz.date_cal(begin_date, amount=days, unit='d')
    
    bts_stock_data = load_data(begin_date=begin_date,
                               end_date=end_date)
    
    bts_stock_data = bts_stock_data.drop(['HIGH', 'LOW'], axis=1)
    
    
    bts_stock_data['FAIL'] = ''
    volume = 1000
    
    roi_goal = 1.05
    
    
    unique_symbol = bts_stock_data[['STOCK_SYMBOL']] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    
    
    backtest_results = pd.DataFrame()
    
    
    # Iterate by symbol ......
    for j in range(0, 5):
    # for j in range(0, len(unique_symbol)):
        
        cur_symbol = unique_symbol.loc[j, 'STOCK_SYMBOL']
        
        temp = bts_stock_data[bts_stock_data['STOCK_SYMBOL']==cur_symbol] \
                .reset_index(drop=True)

        
        # Consider buy and sell multiplie times, but only one currently.
        buy_lock = False
        sell_lock = False

        
        for i in range(0, len(temp)):
            
            
            today = temp.loc[i, 'WORK_DATE']
            model_end = cbyz.date_cal(today, amount=days, unit='d')
            

            # Model results .......
            # (1) Update, should deal with multiple signal issues.
            #     Currently, only consider the first signal.
            model_results_raw = sam.dev_model(data_begin=today,
                                          data_end=model_end,
                                          stock_symbol=cur_symbol,
                                          remove_none=False)
            
            
            cur_price = temp.loc[i, 'CLOSE']
            
            buy_signal_today = model_results_raw[
                model_results_raw['WORK_DATE']==today]



            # buy_signal = model_results_raw[
            #     model_results_raw['BUY_SIGNAL']==True]


            # sell_signal = model_results_raw[
            #     model_results_raw['SELL_SIGNAL']==True]
            
            
            # Replace with model.
            if buy_lock == False and \
                buy_signal_today.loc[0, 'BUY_SIGNAL'] == True:
                
                temp.loc[i, 'BUY_VOLUME'] = volume
                temp.loc[i, 'TYPE'] = 'BUY'
                buy_price = temp.loc[i, 'CLOSE']
                buy_lock = True
                continue
            
        
            # Optimize append
            roi = cur_price / buy_price
            if buy_lock == True and sell_lock == False \
                and roi >= roi_goal:
                    
                temp.loc[i, 'SELL_VOLUME'] = volume
                temp.loc[i, 'TYPE'] = 'SELL'
                backtest_results = backtest_results.append(temp)
                break

            
            # Optimize append
            if i >= days:
                temp.loc[i, 'SELL_VOLUME'] = volume
                temp.loc[i, 'FAIL'] = True
                temp.loc[i, 'TYPE'] = 'SELL'
                backtest_results = backtest_results.append(temp)
                break
            
            print(str(i) + '_' + str(model_begin) + '_' + str(today))
           
        print('symbole - ' + str(j) + '/' + str(len(unique_symbol)))
    
    
    # Remove na ------    
    backtest_results = backtest_results[~backtest_results['TYPE'].isna()]
        
    
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





def master(begin_date, periods=5,
           signal=None, budget=None, split_budget=False):
    '''
    主工作區
    '''
    
    # fee = get_stock_fee()

    
    # Variables    
    # (1) Fix missing date issues
    
    begin_date = 20190401
    
    
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




def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()














