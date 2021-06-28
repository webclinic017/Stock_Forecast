#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:09 2020

@author: Aron
"""


# Worklist
# 1.Add price increse but model didn't catch

# ROI不該用固定值，而應該用停損點的概念下去看。
# 當購買後的最高價 - 購買價，價差點跌破8成的時候就售出。



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
    


def backtest_single(begin_date, stock_symbol=['0050', '0056'],
                    model_data_period=60, volume=1000, budget=None, 
                    forecast_period=30, backtest_times=5,
                    roi_goal=0.015):
    
    # .......
    time_seq = cbyz.get_time_seq(begin_date=begin_date,
                                 periods=backtest_times,
                                 unit='m',
                                 simplify_date=True)
    
    time_seq = time_seq['WORK_DATE'].tolist()

    
    # Work area ----------
    backtest_results = pd.DataFrame()
    model_list = sam.get_model_list()
    
    
    # Date .........
    for j in range(0, len(time_seq)):
        
        begin_date = time_seq[j]
        end_date = cbyz.date_cal(begin_date, 
                                 amount=model_data_period, 
                                 unit='d')
        
        # ......
        # Bug, end_date可能沒有開盤，導致buy_price為空值
        real_data = load_data(begin_date=begin_date,
                                    end_date=end_date)
        
        real_data = real_data.drop(['HIGH', 'LOW'], axis=1)        
        
    
        # Buy Price
        buy_price = real_data[real_data['WORK_DATE']==end_date] \
            .rename(columns={'CLOSE':'BUY_PRICE'}) 
            
        buy_price = buy_price[['STOCK_SYMBOL', 'BUY_PRICE']] \
                    .reset_index(drop=True)
    
    
        # Model ......
        for i in range(0, len(model_list)):

            cur_model = model_list[i]
            
            # Model results .......
            # (1) Update, should deal with multiple signal issues.
            #     Currently, only consider the first signal.
            # (2) Add data
            
            global model_results_raw
            model_results_raw = cur_model(data_begin=begin_date,
                                          data_end=end_date,
                                          stock_symbol=stock_symbol,
                                          forecast_period=forecast_period,
                                          remove_none=False)                
            
            
            
            temp_results = model_results_raw['RESULTS']
            temp_results = temp_results.merge(buy_price, 
                                                how='left',
                                                on='STOCK_SYMBOL')
            
            
            temp_results['ROI'] = (temp_results['CLOSE'] \
                                    - temp_results['BUY_PRICE']) \
                                    / temp_results['BUY_PRICE']


            temp_results = temp_results[temp_results['ROI'] >= roi_goal] \
                .drop_duplicates(subset='STOCK_SYMBOL')
            
            
            temp_results['MODEL'] = cur_model.__name__
            temp_results['FORECAST_BEGIN'] = begin_date
            temp_results['FORECAST_END'] = end_date
            temp_results['BACKTEST_ID'] = j
            
            backtest_results = backtest_results.append(temp_results)
            
    
    if len(backtest_results) == 0:
        print('backtest_single return 0 row.')
        return pd.DataFrame()
    
    
    # Organize results ------
    backtest_main_pre = backtest_results.copy()


    backtest_main_pre = backtest_main_pre \
        .rename(columns={'WORK_DATE':'BUY_DATE',
                         'CLOSE':'FORECAST_CLOSE'})

    backtest_main_pre['AMOUNT'] = -1 * backtest_main_pre['FORECAST_CLOSE'] \
        * volume
    

    
    # Add Historical Data......
    hist_data_info = backtest_main_pre[['STOCK_SYMBOL', 'FORECAST_BEGIN',
                                   'FORECAST_END', 'FORECAST_CLOSE']] \
                    .reset_index(drop=True)

    
    hist_data_period = hist_data_info[['FORECAST_BEGIN', 'FORECAST_END']] \
                        .drop_duplicates() \
                        .reset_index(drop=True)
    
    hist_data_pre = pd.DataFrame()
        
    for i in range(0, len(hist_data_period)):

        temp_begin = hist_data_period.loc[i, 'FORECAST_BEGIN']
        temp_end = hist_data_period.loc[i, 'FORECAST_END']       
        
        # Symbol ...
        temp_symbol = hist_data_info[
            (hist_data_info['FORECAST_BEGIN']==temp_begin) \
            & (hist_data_info['FORECAST_END']==temp_end)]

        temp_symbol = temp_symbol['STOCK_SYMBOL'].tolist()  
            
            
        new_data = ar.stk_get_data(begin_date=temp_begin, 
                                   end_date=temp_end, 
                                   stock_type='tw', 
                                   stock_symbol=temp_symbol, 
                                   local=local)           
        
        new_data['FORECAST_BEGIN'] = temp_begin
        new_data['FORECAST_END'] = temp_end
        
        hist_data_pre = hist_data_pre.append(new_data)


    hist_data_pre = hist_data_pre.drop(['HIGH', 'LOW'], axis=1) \
                    .reset_index(drop=True)

    hist_data_pre = cbyz.df_ymd(hist_data_pre, cols='WORK_DATE')

        
    # Combine data ......
    hist_data = hist_data_pre.merge(hist_data_info,
                                how='left',
                                on=['FORECAST_BEGIN', 'FORECAST_END', 
                                    'STOCK_SYMBOL']) 
        
    hist_data = hist_data[hist_data['FORECAST_CLOSE'] <= hist_data['CLOSE']] \
        .drop_duplicates(subset=['FORECAST_BEGIN', 'FORECAST_END', 
                                    'STOCK_SYMBOL']) \
        .reset_index(drop=True)
    
      
    # Join forecast and historical data ......
    backtest_main = backtest_main_pre \
                    .merge(hist_data,
                           how='left',
                           on=['STOCK_SYMBOL', 'FORECAST_BEGIN',
                               'FORECAST_END', 'FORECAST_CLOSE'])
        
    
    backtest_main.loc[~backtest_main['CLOSE'].isna(), 'SUCCESS'] = True 
    backtest_main.loc[backtest_main['CLOSE'].isna(), 'SUCCESS'] = False     
        
        
    backtest_main = cbyz.df_ymd(backtest_main, 
                                cols=['BUY_DATE', 'FORECAST_BEGIN',
                                      'FORECAST_END'])
    
    return backtest_main




def master(begin_date, periods=5,
           signal=None, budget=None, split_budget=False):
    '''
    主工作區
    Update, 增加台灣上班上課行事曆，如果是end_date剛好是休假日，直接往前推一天。
    '''
    
    # fee = get_stock_fee()

    
    # Variables    
    # (1) Fix missing date issues
    # begin_date = 20190401
    
    
    time_seq = cbyz.get_time_seq(begin_date=begin_date,
                      periods=periods,
                      unit='m', 
                      simplify_date=True)
   
    # Backtest ----------
    global backtest_results
    backtest_results = pd.DataFrame()    
    
    for i in range(0, len(time_seq)):
        
        single = backtest_single(begin_date=time_seq.loc[i, 'WORK_DATE'],
                                 model_data_period=60, volume=1000, budget=None, 
                                 forecast_period=30, backtest_times=5)
        
        backtest_results = backtest_results.append(single)

        
    backtest_results = backtest_results \
                        .reset_index(drop=True)
    
    
    return backtest_results


# ..............


def check():
    '''
    資料驗證
    '''    
    return ''



if __name__ == '__main__':
    results = master(begin_date=20190401)



# periods=5
# signal=None
# budget=None
# split_budget=False
# days=60
# roi_goal = 0.02
# stock_symbol=['0050', '0056']
# model_data_period=60


# volume=1000
# forecast_period=30
# backtest_times=5
# roi_goal=0.015