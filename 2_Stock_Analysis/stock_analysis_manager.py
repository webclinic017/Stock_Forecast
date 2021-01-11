#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:08 2020

@author: Aron
"""

# Worklist
# 1. Some models only work for some stock symbol.
# > Clustering


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
    path = '/home/aronhack/stock_forecast/2_Stock_Analysis'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 r'/home/aronhack/stock_forecast/Function']


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




def sam_load_data(begin_date, end_date=None, period=None, 
              stock_symbol=None):
    '''
    讀取資料及重新整理
    '''
    
    
    if stock_symbol==None:
        target = ar.stk_get_list(stock_type='tw', 
                              stock_info=False, 
                              update=False,
                              local=local)    
        # Dev
        target = target.iloc[0:10, :]
        stock_symbol = target['STOCK_SYMBOL'].tolist()
    
    
    data_raw = ar.stk_get_data(begin_date=begin_date, end_date=end_date, 
                   stock_type='tw', stock_symbol=stock_symbol, 
                   local=local)    
    
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


# .............
    


def model_1(data=None, data_end=None, data_begin=None, 
            data_period=150, forecast_end=None, forecast_period=15,
            stock_symbol=['0050', '0056'],
              remove_none=True):
    '''
    Linear regression
    '''
    from sklearn.linear_model import LinearRegression
    
    periods = cbyz.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    
    
    loc_data = sam_load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)    
    
    
    # Bug, forecast_period的長度和data_period太接近時會出錯
    loc_data['PRICE_PRE'] = loc_data \
        .groupby('STOCK_SYMBOL')['CLOSE'] \
                            .shift(forecast_period)
    
    model_data = loc_data[~loc_data['PRICE_PRE'].isna()]


    # Predict data ......
    forecast_data_pre = cbyz.df_add_rank(df=loc_data,
                                       group_key='STOCK_SYMBOL',
                                       value='WORK_DATE',
                                       reverse=True)
        
    forecast_data = forecast_data_pre[(forecast_data_pre['RANK']>=0) \
                                  & (forecast_data_pre['RANK']<forecast_period)]
    
    # Date
    forecast_date = cbyz.time_get_seq(begin_date=periods['FORECAST_BEGIN'],
                                      periods=forecast_period,
                                      unit='d', simplify_date=True)
        
    forecast_date = forecast_date['WORK_DATE'].tolist()
        
    # Model ........
    model_info = pd.DataFrame()
    forecast_results = pd.DataFrame()
    
    for i in range(0, len(stock_symbol)):
        
        cur_symbol = stock_symbol[i]
        # print(i)
    
        # Model .........
        # Update, doesn't need reshape with multiple features.
        
        
        x = cbyz.ml_conv_to_nparray(model_data['PRICE_PRE'])
        
        y = model_data['CLOSE'].to_numpy()
        
        reg = LinearRegression().fit(x, y)
        
        reg.score(x, y)
        reg.coef_
        reg.intercept_
    
    
        # Forecast ......
        temp_forecast = forecast_data[
            forecast_data['STOCK_SYMBOL']==cur_symbol]
        
        
        temp_forecast = cbyz.ml_conv_to_nparray(temp_forecast['CLOSE'])       
        temp_results = reg.predict(temp_forecast)    
        
        
        # print(stock_symbol[i])
        # print(forecast_date)
        # print(temp_results)
        
        # ...
        temp_df = pd.DataFrame(data={'WORK_DATE':forecast_date,
                                     'CLOSE':temp_results})
        temp_df['STOCK_SYMBOL'] = cur_symbol
        
        forecast_results = forecast_results.append(temp_df)
        
    
    # Reorganize ------
    cols = ['STOCK_SYMBOL', 'WORK_DATE', 'CLOSE']        
    forecast_results = forecast_results[cols]

    return_dict = {'MODEL_INFO':model_info,
                   'RESULTS':forecast_results}

    return return_dict


# ................


def get_model_list(status=[0,1]):
    '''
    List all analysis here
    '''    

    # (1) List manually
    # (2) List by model historic performance
    # (3) List by function pattern
    
    # function_list = [model_dev1, model_dev2, model_dev3]
    function_list = [model_1]
    
    
    return function_list


# ...............


# def analyze_center(data):
#     '''
#     List all analysis here
#     '''    
    
#     analyze_results = get_top_price(data)
    

#     # Results format
#     # (1) Only stock passed test will show in the results
#     # STOCK_SYMBOL
#     # MODEL_ID, or MODEL_ID
    
#     return analyze_results




# %% Master ------
    

def master(begin_date=20190401, today=None, hold_stocks=None, roi=10, limit=90):
    '''
    主工作區
    roi:     percent
    limit:   days
    '''
    
    global stock_data
    stock_data = sam_load_data(begin_date=begin_date)
    
    # global analyze_results
    # analyze_results = analyze_center(data=stock_data)
    
    
    # v0
    # buy_signal = get_buy_signal(data=stock_data,
    #                             hold_stocks=hold_stocks)
    
    # sell_signal = get_sell_signal(data=analyze_results,
    #                               hold_stocks=hold_stocks)
    
    # master_results = {'RESULTS':analyze_results,
    #                   'BUY_SIGNAL':buy_signal,
    #                   'SELL_SIGNAL':sell_signal}
    
    
    global model1_results
    model1_results = model_1(data=None, data_end=None, data_begin=begin_date, 
                             data_period=150, forecast_end=None, 
                             forecast_period=15,
                             stock_symbol=['0050', '0056'],
                             remove_none=True)
    
    return ''



def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()



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

        
    results['MODEL_ID'] = 'TM01'
    results = results[['STOCK_SYMBOL', 'MODEL_ID',
                       'BUY_SIGNAL', 'SELL_SIGNAL']]

    return results   







def model_template(data_end, data_begin=None, data_period=150,
              forecast_end=None, forecast_period=30,
              stock_symbol=None,
              remove_none=True):
    
    from sklearn.linear_model import LinearRegression
    
    periods = cbyz.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    
    
    loc_data = sam_load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)    
    
    # Model .........


    # Reorganize ------
    model_info = pd.DataFrame()
    forecast_results = pd.DataFrame()
    
    cols = ['STOCK_SYMBOL', 'WORK_DATE', 'CLOSE']        
    forecast_results = forecast_results[cols]

    return_dict = {'MODEL_INFO':model_info,
                   'RESULTS':forecast_results}

    return return_dict




# data_begin = 20200301

# # data_begin=None
# data_end=None
# data_period=30
# forecast_end=None
# forecast_period=30

# stock_symbol=['0050', '0056']







# periods = cbyz.date_get_period(data_begin=20191201, 
#                              data_end=None, 
#                              data_period=20191231,
#                              forecast_end=None, 
#                              forecast_period=10)