#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:09 2020

@author: Aron
"""


# Worklist
# 1.Add price increse but model didn't catch
# 2.Retrieve one symbol historical data to ensure calendar



# 在台灣投資股票的交易成本包含手續費與交易稅，
# 手續費公定價格是0.1425%，買進和賣出時各要收取一次，
# 股票交易稅是0.3%，如果投資ETF交易稅是0.1%，僅在賣出時收取。


# To do action
# (1) 集成
# (2) 用迴歸，看哪一支model的成效好
# (3) 多數決
# (4) RMSE > How many model agree > RMSE (Chosen)


# rmse and profit regression


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import random


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/3_Backtest'
else:
    path = '/home/aronhack/stock_forecast/3_Backtest'
    

# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk
import stock_analysis_manager as sam



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     


# ..........
    

def backtest_predict(bt_last_begin, predict_period, interval, 
                     bt_times, data_period):
    
    
    global stock_symbol, stock_type, bt_info
    
    # Prepare For Backtest Records ......
    bt_info_raw = cbyz.date_get_seq(begin_date=bt_last_begin,
                                seq_length=bt_times,
                                unit='d', interval=-interval,
                                simplify_date=True)
    
    bt_info = bt_info_raw[['WORK_DATE']] \
            .reset_index() \
            .rename(columns={'index':'BACKTEST_ID'})
    
    bt_info['DATA_PERIOD'] = data_period
    bt_info['PREDICT_PERIOD'] = predict_period
    bt_info = bt_info.drop('WORK_DATE', axis=1)
    
    
    bt_seq = bt_info_raw['WORK_DATE'].tolist()
    
    
    # Work area ----------
    global bt_results, rmse, features, model_y
    bt_results_raw = pd.DataFrame()
    rmse = pd.DataFrame()
    features = pd.DataFrame()
    
    # Predict ......
    for i in range(0, len(bt_seq)):
        
        begin = bt_seq[i]

        results_raw = sam.master(_predict_begin=begin, 
                                 _predict_end=None, 
                                 _predict_period=predict_period,
                                 data_period=data_period, 
                                 _stock_symbol=stock_symbol,
                                 ma_values=ma_values)


        new_results = results_raw[0]
        new_results['BACKTEST_ID'] = i
        
        new_rmse = results_raw[1]
        new_rmse['BACKTEST_ID'] = i
        
        new_features = results_raw[2]
        new_features['BACKTEST_ID'] = i
        
        bt_results_raw = bt_results_raw.append(new_results)
        rmse = rmse.append(new_rmse)
        features = features.append(new_features)


    # Organize ......
    global model_y
    bt_results = bt_results_raw.reset_index(drop=True)
    model_y = cbyz.df_get_cols_except(df=bt_results, 
                                      except_cols=['STOCK_SYMBOL', 'WORK_DATE', 
                                                   'MODEL', 'BACKTEST_ID'])

    rmse = rmse \
        .sort_values(by=['MODEL', 'Y']) \
        .reset_index(drop=True)
    

# ............



def cal_profit(y_thld=2, time_thld=10, rmse_thld=0.15, 
               export_file=True, load_file=False, path=None, file_name=None):
    '''
    應用場景有可能全部作回測，而不做任何預測，因為data_end直接設為bt_last_begin
    '''
    
    global predict_period, bt_last_begin
    global bt_results, rmse, bt_main, actions, model_y
    global stock_symbol, stock_type


    print('Bug, data_begin and data_end should follow the backtest range')
    loc_begin = 20180101
    hist_data_raw = stk.get_data(data_begin=loc_begin, 
                                 data_end=bt_last_begin, 
                                 stock_type=stock_type, 
                                 stock_symbol=stock_symbol, 
                                 price_change=True,
                                 local=local)
    
    hist_data_raw = hist_data_raw[['WORK_DATE', 'STOCK_SYMBOL'] + model_y]


    # Get Last Date ....        
    calendar_end = cbyz.date_cal(bt_last_begin, predict_period, unit='d')
    date_prev = ar.get_calendar(begin_date=loc_begin, end_date=calendar_end,
                                simplify=True)
    
    date_prev = date_prev[['WORK_DATE']]
    date_prev = cbyz.df_add_shift(df=date_prev, cols='WORK_DATE', shift=1)
    date_prev = date_prev[0].dropna(subset=['WORK_DATE_PRE'], axis=0)
    date_prev = cbyz.df_conv_col_type(df=date_prev, cols='WORK_DATE_PRE', 
                                      to='int')
    
    date_prev.columns = ['WORK_DATE', 'LAST_DATE']
    date_prev = date_prev.reset_index(drop=True)
    
    
    if len(stock_symbol) > 0:
        symbol_df = pd.DataFrame({'STOCK_SYMBOL':stock_symbol})
    else:
        temp_symbol = hist_data_raw['STOCK_SYMBOL'].unique().tolist()
        symbol_df = pd.DataFrame({'STOCK_SYMBOL':temp_symbol})        
        
        
    hist_data = cbyz.df_cross_join(date_prev, symbol_df)
    hist_data = hist_data.merge(hist_data_raw, how='left', 
                                on=['WORK_DATE', 'STOCK_SYMBOL'])
    
    # Merge hist data
    hist_data = cbyz.df_shift_fill_na(df=hist_data, 
                                      loop_times=predict_period+1, 
                                      cols=model_y, group_by=['STOCK_SYMBOL'])

    # Add last price
    hist_data, _ = cbyz.df_add_shift(df=hist_data, cols=model_y, shift=1,
                                     group_by=['STOCK_SYMBOL'], suffix='_LAST', 
                                     remove_na=False)


    # Prepare columns ......
    hist_cols = [i + '_HIST' for i in model_y]
    hist_cols_dict = cbyz.li_to_dict(model_y, hist_cols)
    hist_data = hist_data.rename(columns=hist_cols_dict)

    last_cols = [i + '_LAST' for i in model_y]
    
    
    # Organize ......
    main_data = bt_results.merge(hist_data, how='left', 
                                 on=['WORK_DATE', 'STOCK_SYMBOL'])


    main_data = main_data[['BACKTEST_ID', 'STOCK_SYMBOL', 'MODEL', 
                           'WORK_DATE', 'LAST_DATE'] \
                          + model_y + hist_cols + last_cols]

    # Fill na in the prediction period.
    for i in hist_cols:
        main_data[i] = np.where(main_data['WORK_DATE'] > bt_last_begin,
                                np.nan, main_data[i])

    # Check na ......
    # 這裡有na是合理的，因為hist都是na
    chk = cbyz.df_chk_col_na(df=main_data, positive_only=True)
    
    last_cols = [i + '_LAST' for i in model_y]
    main_data = main_data.dropna(subset=last_cols)
    
    if len(chk) > len(model_y):
        print('Err01. cal_profit - main_data has na in columns.')


    bt_main, actions = \
        stk.gen_predict_action(df=main_data, rmse=rmse,
                                  date='WORK_DATE', 
                                  last_date='LAST_DATE', 
                                  y=model_y, y_last=last_cols,
                                  y_thld=y_thld, time_thld=time_thld,
                                  rmse_thld=rmse_thld, 
                                  export_file=export_file, 
                                  load_file=load_file, file_name=file_name,
                                  path=path)
    
    # Rearrange Columns ......            
    if 'CLOSE' in model_y:
        profit_cols = ['CLOSE_PROFIT_PREDICT', 'CLOSE_PROFIT_RATIO_PREDICT']
        
    cols_1 = ['BACKTEST_ID', 'STOCK_SYMBOL', 'STOCK_NAME', 
              'MODEL', 'WORK_DATE', 'LAST_DATE']

    model_y_last = [s + '_LAST' for s in model_y]
    model_y_hist = [s + '_HIST' for s in model_y]    
    cols_2 = ['RMSE_MEAN']
    
    new_cols = cols_1 + profit_cols + model_y + model_y_last \
                + model_y_hist + cols_2
                
    actions = actions[new_cols]
    
    
    # # MAPE ......
    # global mape, mape_group, mape_extreme
    # mape = pd.DataFrame()
    # mape_group = pd.DataFrame()
    # mape_extreme = pd.DataFrame()
    # mape_main = bt_main[bt_main['BACKTEST_ID']>0]
    
    # for y in model_y:
        
    #     mape_main['MAPE'] = abs(mape_main[y] \
    #                             - mape_main[y + '_HIST']) \
    #                         / mape_main[y + '_HIST']
                            
    #     mape_main['OVERESTIMATE'] = \
    #         np.where(mape_main[y] > mape_main[y + '_HIST'], 1, 0)
                 
    #     # MAPE
    #     new_mape = cbyz.summary(df=mape_main, group_by=[], cols='MAPE')
    #     new_mape['Y'] = y        
    #     mape = mape.append(new_mape)
        
        
    #     # MAPE2
    #     new_mape = cbyz.summary(df=mape_main, group_by='OVERESTIMATE', 
    #                              cols='MAPE')
    #     new_mape['Y'] = y
    #     mape_group = mape_group.append(new_mape)
        
        
    #     # MAPE Extreme
    #     new_mape = mape_main[mape_main['MAPE'] > 0.1]
    #     new_mape = cbyz.summary(df=new_mape, 
    #                             group_by=['BACKTEST_ID', 'OVERESTIMATE'], 
    #                             cols='MAPE')
    #     new_mape['Y'] = y
    #     mape_extreme = mape_extreme.append(new_mape)
        


# .................



def eval_metrics(export_file=False, upload=False):



    # MAPE ......
    global bt_main, bt_info, rmse
    global mape, mape_group, mape_extreme
    global stock_metrics_raw, stock_metrics
    
    model_y_hist = [y + '_HIST' for y in model_y]
    mape_main = bt_main.dropna(subset=model_y_hist, axis=0)
        
    mape = pd.DataFrame()
    mape_group = pd.DataFrame()
    mape_extreme = pd.DataFrame()
    stock_metrics_raw = pd.DataFrame()
    
    
    for y in model_y:
        
        mape_main.loc[:, 'MAPE'] = abs(mape_main[y] \
                                - mape_main[y + '_HIST']) \
                            / mape_main[y + '_HIST']
                            
        mape_main.loc[:, 'OVERESTIMATE'] = \
            np.where(mape_main[y] > mape_main[y + '_HIST'], 1, 0)
                 
        # MAPE Overview
        new_mape = cbyz.summary(df=mape_main, group_by=[], cols='MAPE')
        new_mape.loc[:, 'Y'] = y
        mape = mape.append(new_mape)
        
        
        # Group MAPE ......
        new_mape = cbyz.summary(df=mape_main, group_by='OVERESTIMATE', 
                                 cols='MAPE')
        new_mape.loc[:, 'Y'] = y
        mape_group = mape_group.append(new_mape)
        
        
        # Extreme MAPE ......
        new_mape = mape_main[mape_main['MAPE'] > 0.1]
        new_mape = cbyz.summary(df=new_mape, 
                                group_by=['BACKTEST_ID', 'OVERESTIMATE'], 
                                cols='MAPE')
        new_mape.loc[:, 'Y'] = y
        mape_extreme = mape_extreme.append(new_mape)


        # Stock MAPE ......
        new_metrics = mape_main[['BACKTEST_ID', 'STOCK_SYMBOL', 
                                 'WORK_DATE', 'MAPE', 'OVERESTIMATE',
                                 y, y + '_HIST']] \
            .rename(columns={y:'FORECAST_VALUE',
                             y + '_HIST':'HIST_VALUE'})

        new_metrics.loc[:, 'Y'] = y
        stock_metrics_raw = stock_metrics_raw.append(new_metrics)
    
    
    # Oraganize
    stock_metrics_raw = stock_metrics_raw \
                        .merge(bt_info, how='left', on='BACKTEST_ID') \
                        .merge(rmse, how='left', on=['Y', 'BACKTEST_ID'])


    stock_metrics_raw['MODEL_METRIC'] = 'RMSE'    
    stock_metrics_raw['FORECAST_METRIC'] = 'MAPE'
    stock_metrics_raw['STOCK_TYPE'] = 'TW'
    stock_metrics_raw.loc[:, 'EXECUTE_DATE'] = cbyz.date_get_today(simplify=False)
    
    
    stock_metrics_raw = \
        cbyz.df_date_simplify(df=stock_metrics_raw, 
                              cols=['EXECUTE_DATE', 'WORK_DATE'])    
    
    stock_metrics_raw = stock_metrics_raw \
                        .rename(columns={'WORK_DATE':'PREDICT_DATE',
                                         'RMSE':'MODEL_PRECISION',
                                         'MAPE':'FORECAST_PRECISION'}) \
                        .round({'MODEL_PRECISION':3,
                                'FORECAST_PRECISION':3})

                
    stock_metrics = stock_metrics_raw[['STOCK_TYPE', 'STOCK_SYMBOL', 
                                       'EXECUTE_DATE', 'PREDICT_DATE', 
                                       'PREDICT_PERIOD', 'DATA_PERIOD', 'Y',
                                       'MODEL_METRIC', 'MODEL_PRECISION',
                                       'FORECAST_METRIC', 'FORECAST_PRECISION', 
                                       'OVERESTIMATE']]
    
    stock_metrics['STOCK_TYPE'] = 'DEL'
    
        
    if export_file:
        time_serial = cbyz.get_time_serial(with_time=True)
        stock_metrics.to_csv(path_export + '/Metrics/stock_mape_'\
                             + time_serial + '.csv', 
                             index=False)
            
    if upload:
        ar.db_upload(data=stock_metrics, table_name='forecast_records3', 
                     local=local)



# ..........



def master(_bt_last_begin, predict_period=14, interval=360, bt_times=5, 
           data_period=5, _stock_symbol=None, _stock_type='tw',
           signal=None, budget=None, split_budget=False):
    '''
    主工作區
    Update, 增加台灣上班上課行事曆，如果是end_date剛好是休假日，直接往前推一天。
    '''
    
    
    # Parameters
    _bt_last_begin = 20210702
    # bt_last_begin = 20210211
    predict_period = 2
    interval = random.randrange(15, 40)
    bt_times = 2
    data_period = 360 * 2
    # data_period = 360 * 5
    # data_period = 360 * 7    
    _stock_symbol = [2520, 2605, 6116, 6191, 3481, 2409, 2603]
    _stock_symbol = []
    _stock_type = 'tw'
    _ma_values = [5,20]
    # _ma_values = [3,5,20,60,120]    



    # ......    
    global stock_symbol, bt_last_begin, stock_type, ma_values
    stock_symbol = _stock_symbol
    bt_last_begin = _bt_last_begin
    
    stock_type = _stock_type    
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, to_type='str')
    
    ma_values = _ma_values
    
    # full_data = sam.sam_load_data(data_begin=None, data_end=None, 
    #                               stock_type=stock_type, period=None, 
    #                               stock_symbol=stock_symbol, 
    #                               lite=False, full=True)


    # full_data = full_data['FULL_DATA']
    
    
    # 算回測precision的時候，可以低估，但不可以高估
    
    # Predict ------
    global bt_results, rmse, features
    backtest_predict(bt_last_begin=bt_last_begin, 
                     predict_period=predict_period, 
                     interval=interval,
                     bt_times=bt_times,
                     data_period=data_period)

    
    # Profit ------    
    # Update, 需把Y為close和Y為price_change的情況分開處理
    
    # y_thld=0.2
    # time_thld=predict_period
    # rmse_thld=0.1
    # export_file=True
    # load_file=False
    # path=None
    # file_name=None    
    
    
    global bt_main, actions
    cal_profit(y_thld=-100, time_thld=predict_period, rmse_thld=5,
               export_file=True, load_file=True, path=path_temp,
               file_name=None) 
    
    
    actions = actions[actions['MODEL']=='model_6']
    actions = cbyz.df_add_size(df=actions, group_by='STOCK_SYMBOL',
                               col_name='ROWS')


    time_serial = cbyz.get_time_serial(with_time=True)
    actions.to_excel(path_export + '/actions_' + time_serial + '.xlsx', 
                     index=False, encoding='utf-8-sig')


    # ------
    global mape, mape_group, mape_extreme
    global stock_metrics_raw, stock_metrics
    eval_metrics(export_file=False, upload=False)
    # eval_metrics(export_file=False, upload=True)

    
    return ''



# ..............=



# def upload_records():
#     '''
#     Delete if eval_metrics works fine.
#     '''
    
   
#     file = pd.read_csv(path_export + '/Metrics/stock_20210703_123339.csv')
#     file = pd.read_csv(path_export + '/Metrics/stock_mape_20210703_133643.csv')
    
#     file.columns = ['STOCK_TYPE', 'STOCK_SYMBOL', 
#                     'EXECUTE_DATE', 'PREDICT_DATE', 
#                     'PREDICT_PERIOD', 'DATA_PERIOD', 'Y',
#                     'FORECAST_METRIC', 'FORECAST_PRECISION', 
#                     'OVERESTIMATE']
    
    
#     file['MODEL_METRIC'] = 'RMSE'
#     file['MODEL_PRECISION'] = 0.058570158
#     file['MODEL_PRECISION'] = 0.056767662
    
    
#     file = file[['STOCK_TYPE', 'STOCK_SYMBOL', 
#                 'EXECUTE_DATE', 'PREDICT_DATE', 
#                 'PREDICT_PERIOD', 'DATA_PERIOD', 'Y',
#                 'MODEL_METRIC', 'MODEL_PRECISION',
#                 'FORECAST_METRIC', 'FORECAST_PRECISION', 
#                 'OVERESTIMATE']]
    
    
#     file = file.round({'MODEL_PRECISION':3,
#                       'FORECAST_PRECISION':3})
    
#     # ar.db_upload(data=file, table_name='forecast_records3', 
#     #               local=local, chunk=30000)
    
#     return ''



def check():
    '''
    資料驗證
    '''    
    
    # Err01
    chk = main_data[main_data['HIGH_HIST'].isna()]   
    chk
    
    return ''


def check_price():
    
    
    chk = bt_results[bt_results['STOCK_SYMBOL']=='4967']
    chk

    chk = bt_results[bt_results['STOCK_SYMBOL']=='2702']
    chk


if __name__ == '__main__':
    results = master(begin_date=20180401)





# %% Verify ......
def verify_prediction_results():

    
    # Diary
    # 0627 - 預測的標的中，實際漲跌有正有負，但負的最多掉2%。

    
    end = 20210625
    begin = cbyz.date_cal(end, -1, 'd')
    
    
    # Files ......
    file_raw = pd.read_excel(path_export + '/actions_20210624_233111.xlsx')
    
    file = cbyz.df_conv_col_type(df=file_raw, cols='STOCK_SYMBOL', to='str')
    file = file_raw[file_raw['WORK_DATE']==end]
    file = file[['STOCK_SYMBOL', 'CLOSE', 'HIGH', 'LOW']]
    file.columns = ['STOCK_SYMBOL', 'CLOSE_PREDICT', 
                    'HIGH_PREDICT', 'LOW_PREDICT']
    
    
    symbols = file_raw['STOCK_SYMBOL'].unique().tolist()
    
    
    # Market Data ......
    data_raw = stk.get_data(data_begin=begin, 
                        data_end=end, 
                        stock_type='tw', 
                        stock_symbol=[], 
                        local=local,
                        price_change=True)

    data = data_raw[(data_raw['WORK_DATE']==20210625) \
                & (data_raw['STOCK_SYMBOL'].isin(symbols))] \
            .sort_values(by='PRICE_CHANGE_RATIO', ascending=False) \
            .reset_index(drop=True)

    
    main_data = data.merge(file, how='left', on='STOCK_SYMBOL')
    
    
    

    

# %% Dev -----

def get_stock_fee():
    
    
    return ''




# %% Archive ------
