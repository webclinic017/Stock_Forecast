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




def load_data(begin_date, end_date=None):
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
    

# begin_date = 20190301
# days=60
# volume=None
# budget=None
# roi_goal = 0.02

def backtest_single(begin_date, days=60, volume=None, budget=None,
                    roi_goal = 0.02):
    
   
    # ........    
    # begin_date = cbyz.ymd(begin_date)
    end_date = cbyz.date_cal(begin_date, amount=days, unit='d')
    
    bts_stock_data = load_data(begin_date=begin_date,
                               end_date=end_date)
    
    bts_stock_data = bts_stock_data.drop(['HIGH', 'LOW'], axis=1)
    
    
    bts_stock_data['FAIL'] = ''
    volume = 1000
    
    

    
    unique_symbol = bts_stock_data[['STOCK_SYMBOL']] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    
    
    backtest_results = pd.DataFrame()
    
    model_list = sam.get_model_list()
    
    # Iterate by symbol ......
    for k in range(0, 5):
    # for j in range(0, len(unique_symbol)):
        
        cur_symbol = unique_symbol.loc[k, 'STOCK_SYMBOL']
        
        temp = bts_stock_data[
            bts_stock_data['STOCK_SYMBOL']==cur_symbol] \
                .reset_index(drop=True)
                
          
        # Consider buy and sell multiplie times, but only one currently.

        for j in range(0, len(model_list)):

            buy_lock = False
            sell_lock = False
            
            cur_model = model_list[j]
            
            for i in range(0, len(temp)):
                
                today = temp.loc[i, 'WORK_DATE']
                model_end = cbyz.date_cal(today, amount=days, unit='d')
                
    
                # Model results .......
                # (1) Update, should deal with multiple signal issues.
                #     Currently, only consider the first signal.
                
                model_results_raw = cur_model(data_begin=today,
                                              data_end=model_end,
                                              stock_symbol=cur_symbol,
                                              remove_none=False)                
                
                
                
                cur_price = temp.loc[i, 'CLOSE']
                
                buy_signal_today = model_results_raw[
                    model_results_raw['WORK_DATE']==today]
    
    
                # Remove temporaily
                # buy_signal = model_results_raw[
                #     model_results_raw['BUY_SIGNAL']==True]
    
    
                # sell_signal = model_results_raw[
                #     model_results_raw['SELL_SIGNAL']==True]
                
                
                # Replace with model.
                if buy_lock == False and \
                    buy_signal_today.loc[0, 'BUY_SIGNAL'] == True:
                    
                    temp.loc[i, 'TRADE_VOLUME'] = volume
                    temp.loc[i, 'MODEL'] = cur_model.__name__
                    temp.loc[i, 'TYPE'] = 'BUY'
                    buy_price = temp.loc[i, 'CLOSE']
                    buy_lock = True
                    continue
                
            
                # Optimize append            
                if buy_lock == True and sell_lock == False:
                    
                    roi = (cur_price - buy_price) / buy_price
                    
                    if roi >= roi_goal:
                        temp.loc[i, 'TRADE_VOLUME'] = -volume
                        temp.loc[i, 'MODEL'] = cur_model.__name__
                        temp.loc[i, 'TYPE'] = 'SELL'
                        backtest_results = backtest_results.append(temp)
                        
                        print(str(j) + '_' + str(today) + '_' + str(model_end))
                        break
    
                
                # Optimize append
                if i >= days:
                    temp.loc[i, 'SELL_VOLUME'] = volume
                    temp.loc[i, 'FAIL'] = True
                    temp.loc[i, 'TYPE'] = 'SELL'
                    backtest_results = backtest_results.append(temp)
                    break
            
           
                print('symbol - ' + str(i) + '/' \
                      + str(j) + '/' \
                      + str(k) + '/' \
                      + str(len(unique_symbol)))
    

    
    if len(backtest_results) == 0:
        print('backtest_single return 0 row.')
        return pd.DataFrame()
    
    
    # Remove na ------    
    backtest_results = backtest_results[
        ~backtest_results['TYPE'].isna()] \
                    .reset_index(drop=True)
                
                
    if len(backtest_results) == 0:
        print('backtest_single return 0 row.')
        return pd.DataFrame()
        
    
    # Organize results ------
    backtest_main_pre = backtest_results.copy()
    
    backtest_main_pre['WORK_DATE'] = backtest_main_pre['WORK_DATE'] \
                                        .apply(cbyz.ymd)

    
    backtest_main_pre['AMOUNT'] = -1 * backtest_main_pre['CLOSE'] \
        * backtest_main_pre['TRADE_VOLUME']       
    
    
    # summary = backtest_main_pre \
    #             .groupby(['STOCK_SYMBOL', 'TYPE']) \
    #             .aggregate({'AMOUNT':'sum'}) \
    #                 .reset_index()


  
    # # Calculate days ------
    # backtest_main['DAYS'] = backtest_main['WORK_DATE_SELL'] \
    #                             - backtest_main['WORK_DATE_BUY']
    
    
    # backtest_main['DAYS'] = backtest_main['DAYS'].astype(str)
    
    # backtest_main['DAYS'] = backtest_main['DAYS'] \
    #                             .str.replace(' days', '')
                                
    # backtest_main['DAYS'] = backtest_main['DAYS'].astype(int)
    
    
    
    # # Calculate Profits ------
    # # Replace VOLUME_BUY with VOLUME_SELL
    # backtest_main['PROFIT'] = (backtest_main['PRICE_SELL'] \
    #                            - backtest_main['PRICE_BUY']) \
    #                             * backtest_main['VOLUME_BUY']
    
    
    # backtest_main['ROI'] = backtest_main['PROFIT'] \
    #                             / (backtest_main['VOLUME_BUY'] \
    #                                * backtest_main['PRICE_BUY'])
    
    
    # results = {'RESULTS':backtest_results,
    #            'SUMMARY':summary_pivot}
    
    # return results
    return backtest_results



periods=5
signal=None
budget=None
split_budget=False


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
        
        single = backtest_single(begin_date=time_seq.loc[i, 'WORK_DATE'],
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





