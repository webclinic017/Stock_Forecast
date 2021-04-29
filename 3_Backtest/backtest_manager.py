#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:09 2020

@author: Aron
"""


# Worklist
# 1.Add price increse but model didn't catch
# 2.Retrieve one symbol historical data to ensure calendar


# Trend and associtation

# Add date for sell signal
# Delete success



# 在台灣投資股票的交易成本包含手續費與交易稅，
# 手續費公定價格是0.1425%，買進和賣出時各要收取一次，
# 股票交易稅是0.3%，如果投資ETF交易稅是0.1%，僅在賣出時收取。


# 0213
# add full data


# Monitoring




# To do action
# (1) 集成
# (2) 用迴歸，看哪一支model的成效好
# (3) 多數決
# (4) RMSE > How many model agree > RMSE (Chosen)


# Update 1，在沒有sell_signal的情況下繼續放著，總收益會是多少？


# rmse and profit regression


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
    master_path = '/home/aronhack/stock_forecast'
    


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 master_path + '/2_Stock_Analysis',
                 master_path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk
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
    
    
    cbyz.os_create_folder(path=[path_resource, path_function, 
                             path_temp, path_export])        
    return ''



# ..............


def btm_get_stock_symbol(stock_symbol=None, top=10):
    
    global target_symbol
    
    if stock_symbol == None:
    
        target_symbol = ar.stk_get_list(stock_type='tw', 
                                        stock_info=False, 
                                        update=False,
                                        local=local)    
        
        if top > 0:
            target_symbol = target_symbol.iloc[0:top, :]
            
        target_symbol = target_symbol['STOCK_SYMBOL'].tolist()
        
    else:
        target_symbol = stock_symbol
        
    return target_symbol


# ..............


def btm_load_data(begin_date, end_date=None):
    '''
    讀取資料及重新整理
    '''   
    
    data_raw = ar.stk_get_data(data_begin=begin_date, 
                               data_end=end_date, 
                               stock_type='tw', stock_symbol=target_symbol,
                               local=local)    
    
    return data_raw




def get_stock_fee():
    
    
    return ''


# ..........
    

begin_date = 20170701
data_period=360
interval=30
volume=1000
budget=None
days=60
volume=None
budget=None
roi_base = 0.02
stop_loss=0.8
lag=7
backtest_times=5
    

stock_symbol=[2520, 2605, 6116, 6191, 3481, 2409]
stock_type='tw'
predict_period=15
backtest_times=5
signal_thld=0.03
stop_loss=0.8
lag=7
debug=False






def backtest_predict(begin, 
                     predict_period=14, data_period=360, 
                     interval=30,
                     volume=1000, budget=None, 
                     stock_symbol=[2520, 2605, 6116, 6191, 3481, 2409],
                     stock_type='tw', backtest_times=5,
                     signal_thld=0.03, stop_loss=0.8, lag=7, debug=False):
    

    # .......
    loc_time_seq = cbyz.time_get_seq(begin_date=begin,
                                     periods=backtest_times,
                                     unit='d', skip=interval,
                                     simplify_date=True)
    
    loc_time_seq = loc_time_seq['WORK_DATE'].tolist()
    
    
    # Work area ----------
    model_list = sam.get_model_list()
    buy_signal = pd.DataFrame()
    rmse = pd.DataFrame()    
    error_msg = []

    
    # Date .........
    results = pd.DataFrame()
    actions = pd.DataFrame()
    
    for i in range(0, len(loc_time_seq)):
        
        
        last_date = cbyz.date_cal(loc_time_seq[i], -1, 'd')
        
        # Last price ......
        last_price = stk.get_data(data_begin=last_date,
                                  data_end=last_date,
                                  stock_symbol=stock_symbol,
                                  stock_type=stock_type,
                                  shift=1)
        
        last_price = last_price[['STOCK_SYMBOL', 'CLOSE']] \
                        .rename(columns={'CLOSE':'LAST_CLOSE'})
        
        # Predict ......
        results_raw, rmse = sam.master(predict_begin=begin, 
                                       predict_end=None, 
                                       predict_period=predict_period,
                                       data_period=data_period, 
                                       stock_symbol=stock_symbol, 
                                       today=None, hold_stocks=None, 
                                       limit=90)
        
         # Merge Data ......   
        results = results_raw.merge(last_price, on=['STOCK_SYMBOL'])
        results['INCREASE'] = (results['CLOSE'] - results['LAST_CLOSE']) \
                                / results['LAST_CLOSE']
        
        
        # Decision ......
        results.loc[results['INCREASE']>=signal_thld, 'BUY_SIGNAL'] = 1
        results = cbyz.df_conv_na(df=results, cols='BUY_SIGNAL', value=0)
        
        
        actions = results[results['BUY_SIGNAL']==1] \
                    .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                    .drop_duplicates(subset='STOCK_SYMBOL') \
                    .reset_index(drop=True)    
    

    return results, actions


def backtest_verify(begin_date, actions, stock_symbol, stock_type='tw', 
                    stop_loss=0.8):
    
    
    # Historic Data
    hist_data = stk.get_data(data_begin=begin_date, data_end=None,
                             stock_symbol=stock_symbol,
                             stock_type=stock_type,
                             shift=0)
    
    
    
    for i in range(len(actions)):

        cur_date = actions.loc[i, 'WORK_DATE']
        cur_data = hist_data[hist_data['WORK_DATE']>cur_date]
        cur_data['BUY_PRICE'] = actions.loc[i, 'WORK_DATE']

        
        print(hist_data[i])
        
        
    
    
    
    
    
    
    # Add dynamic ROI -------
    predict_roi_pre = cbyz.df_add_rank(buy_signal, 
                             value='WORK_DATE',
                             group_key=['STOCK_SYMBOL', 'BACKTEST_ID'])
          

    predict_roi = pd.DataFrame()
    
    
    for i in range(0, len(loc_time_seq)):
    
        df_lv1 = predict_roi_pre[predict_roi_pre['BACKTEST_ID']==i]
        unique_symbol = df_lv1['STOCK_SYMBOL'].unique()
        
        
        for j in range(0, len(unique_symbol)):
            
            df_lv2 = df_lv1[df_lv1['STOCK_SYMBOL']==unique_symbol[j]] \
                        .reset_index(drop=True)
            
            for k in range(0, len(df_lv2)):
                
                if k == 0:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'BUY_PRICE']
                    continue
                    
                if df_lv2.loc[k, 'FORECAST_CLOSE'] >= df_lv2.loc[k-1, 'MAX_PRICE']:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'FORECAST_CLOSE']
                else:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k-1, 'MAX_PRICE']
                            
            predict_roi = predict_roi.append(df_lv2)
    
    # ............
    predict_results = predict_roi.copy()
    predict_results.loc[predict_results['MAX_PRICE'] * stop_loss \
                        >= predict_results['FORECAST_CLOSE'], \
                        'SELL_SIGNAL'] = True
    
    predict_results = cbyz.df_conv_na(predict_results,
                                cols='SELL_SIGNAL',
                                value=False)
 
    predict_results['GRP_SELL_SIGNAL'] = predict_results \
        .groupby(['STOCK_SYMBOL', 'BACKTEST_ID'])['SELL_SIGNAL'] \
        .transform(max)
    
    
    
    predict_results.loc[(predict_results['GRP_SELL_SIGNAL']==True) \
                   & (predict_results['SELL_SIGNAL']==False), \
                   'REMOVE'] = True
    
    
    # Remove for GRP_SELL_SIGNAL == NA ......
    # Bug, 移除在period內完全沒碰到停損點的還沒判斷        
    predict_results['MAX_RANK'] = predict_results \
                    .groupby(['STOCK_SYMBOL', 'BACKTEST_ID'])['RANK'] \
                    .transform(max)

    predict_results.loc[(predict_results['GRP_SELL_SIGNAL']==False) \
                   & (predict_results['RANK']<predict_results['MAX_RANK']), \
                   'REMOVE'] = True
      

    
    return ''





def backtest_master(predict_begin=20200801, predict_period=15,
                    interval=30,
                volume=1000, budget=None, 
                stock_symbol=[2520, 2605, 6116, 6191, 3481, 2409],
                stock_type='tw',
                backtest_times=14,
                signal_thld=0.03, stop_loss=0.8, lag=7, debug=False):
    
    
    results, actions = \
        backtest_predict(begin=20190801, predict_period=15,
                         interval=30, data_period=360, 
                        volume=1000, budget=None, 
                         stock_symbol=[2520, 2605, 6116, 6191, 3481, 2409],
                         stock_type='tw',
                         backtest_times=5,
                         signal_thld=0.03, lag=7, debug=False)    

    
    
    backtest_verify(begin_date=begin_date, actions=actions, 
                    stock_symbol=stock_symbol, stop_loss=stop_loss)
    
    
    
    
    return ''



def btm_predict_backup20210428(begin_date, data_period=360, interval=30,
                volume=1000, budget=None, 
                stock_symbol=[2520, 2605, 6116, 6191, 3481, 2409],
                stock_type='tw',
                predict_period=15, backtest_times=5,
                signal_thld=0.03, stop_loss=0.8, lag=7, debug=False):
    
    '''
    interval : int. Backest interval
    
    '''
    

    # .......
    loc_time_seq = cbyz.time_get_seq(begin_date=begin_date,
                                     periods=backtest_times,
                                     unit='d', skip=interval,
                                     simplify_date=True)
    
    loc_time_seq = loc_time_seq['WORK_DATE'].tolist()

    
    # Work area ----------
    model_list = sam.get_model_list()
    buy_signal = pd.DataFrame()
    rmse = pd.DataFrame()    
    error_msg = []
    
    
    # Date .........
    for i in range(0, len(loc_time_seq)):
        
        
        last_date = cbyz.date_cal(loc_time_seq[i], -1, 'd')
        
        # Last price ......
        last_price = stk.get_data(data_begin=last_date,
                                  data_end=last_date,
                                  stock_symbol=stock_symbol,
                                  stock_type=stock_type,
                                  shift=1)
        
        last_price = last_price[['STOCK_SYMBOL', 'CLOSE']] \
                        .rename(columns={'CLOSE':'LAST_CLOSE'})
        
        # Predict ......
        results_raw, rmse = sam.master(data_period=150, predict_end=None, 
                                       predict_period=15,
                                       stock_symbol=stock_symbol, 
                                       today=None, hold_stocks=None, 
                                       limit=90)
        
         # Merge Data ......   
        results = results_raw.merge(last_price, on=['STOCK_SYMBOL'])
        results['INCREASE'] = (results['CLOSE'] - results['LAST_CLOSE']) \
                                / results['LAST_CLOSE']
        
        
        # Decision ......
        results.loc[results['INCREASE']>=signal_thld, 'BUY_SIGNAL'] = 1
        results = cbyz.df_conv_na(df=results, cols='BUY_SIGNAL', value=0)
        
        
        actions = results \
                    .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                    .drop_duplicates(subset='STOCK_SYMBOL') \
                    .reset_index(drop=True)
        
        
        下面還沒改

        # Buy Price ......
        # Bug, 檢查這一段邏輯有沒有錯
        # BUG，BUY_PRICE可能有0
        
        # Bug, end_date可能沒有開盤，導致buy_price為空值
        # real_data = btm_load_data(begin_date=data_begin,
        #                           end_date=data_end)
        
        # real_data = real_data.drop(['HIGH', 'LOW'], axis=1)            
        buy_price = model_data[model_data['WORK_DATE']==data_begin] \
                    .rename(columns={'CLOSE':'BUY_PRICE'}) 
            
            
        buy_price = buy_price[['STOCK_SYMBOL', 'BUY_PRICE']] \
                    .reset_index(drop=True)
    
    
        if len(buy_price) == 0:
            print(str(data_begin) + ' without data.')
            continue


        
    
    if len(buy_signal) == 0:
        print('bt_buy_signal return 0 row.')
        return pd.DataFrame()
    
    
    buy_signal = buy_signal.rename(columns={'CLOSE':'FORECAST_CLOSE'})
    rmse = rmse.reset_index(drop=True)
    
    
    # Add dynamic ROI -------
    predict_roi_pre = cbyz.df_add_rank(buy_signal, 
                               value='WORK_DATE',
                               group_key=['STOCK_SYMBOL', 'BACKTEST_ID'])
            

    predict_roi = pd.DataFrame()
    
    
    for i in range(0, len(loc_time_seq)):
    
        df_lv1 = predict_roi_pre[predict_roi_pre['BACKTEST_ID']==i]
        unique_symbol = df_lv1['STOCK_SYMBOL'].unique()
        
        
        for j in range(0, len(unique_symbol)):
            
            df_lv2 = df_lv1[df_lv1['STOCK_SYMBOL']==unique_symbol[j]] \
                        .reset_index(drop=True)
            
            for k in range(0, len(df_lv2)):
                
                if k == 0:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'BUY_PRICE']
                    continue
                    
                if df_lv2.loc[k, 'FORECAST_CLOSE'] >= df_lv2.loc[k-1, 'MAX_PRICE']:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'FORECAST_CLOSE']
                else:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k-1, 'MAX_PRICE']
                            
            predict_roi = predict_roi.append(df_lv2)
    
    # ............
    predict_results = predict_roi.copy()
    predict_results.loc[predict_results['MAX_PRICE'] * stop_loss \
                        >= predict_results['FORECAST_CLOSE'], \
                        'SELL_SIGNAL'] = True
    
    predict_results = cbyz.df_conv_na(predict_results,
                                cols='SELL_SIGNAL',
                                value=False)
 
    predict_results['GRP_SELL_SIGNAL'] = predict_results \
        .groupby(['STOCK_SYMBOL', 'BACKTEST_ID'])['SELL_SIGNAL'] \
        .transform(max)
    
    
    
    predict_results.loc[(predict_results['GRP_SELL_SIGNAL']==True) \
                   & (predict_results['SELL_SIGNAL']==False), \
                   'REMOVE'] = True
    
    
    # Remove for GRP_SELL_SIGNAL == NA ......
    # Bug, 移除在period內完全沒碰到停損點的還沒判斷        
    predict_results['MAX_RANK'] = predict_results \
                    .groupby(['STOCK_SYMBOL', 'BACKTEST_ID'])['RANK'] \
                    .transform(max)

    predict_results.loc[(predict_results['GRP_SELL_SIGNAL']==False) \
                   & (predict_results['RANK']<predict_results['MAX_RANK']), \
                   'REMOVE'] = True
      
        
    if debug == False:
        predict_results = predict_results[predict_results['REMOVE'].isna()] \
                            .drop(['REMOVE', 'MAX_RANK'], axis=1)   
    
    predict_results = predict_results.reset_index(drop=True)


    predict_dict = {'RESULTS':predict_results, \
                    'RMSE':rmse}
    
    return predict_dict



def btm_predict_backup_20210426(begin_date, data_period=360, interval=30,
                volume=1000, budget=None, 
                predict_period=15, backtest_times=5,
                roi_base=0.03, stop_loss=0.8, lag=7, debug=False):
    
    '''
    interval : int. Backest interval
    
    '''
    

    # .......
    loc_time_seq = cbyz.time_get_seq(begin_date=begin_date,
                                     periods=backtest_times,
                                     unit='d', skip=interval,
                                     simplify_date=True)
    
    loc_time_seq = loc_time_seq['WORK_DATE'].tolist()

    
    # Work area ----------
    model_list = sam.get_model_list()
    buy_signal = pd.DataFrame()
    rmse = pd.DataFrame()    
    error_msg = []
    
    
    # Date .........
    for i in range(0, len(loc_time_seq)):
    

        date_period = cbyz.date_get_period(data_begin=loc_time_seq[i],
                                           data_end=None, 
                                           data_period=data_period,
                                           predict_end=None, 
                                           predict_period=predict_period)        
        

        data_begin_pre = date_period['DATA_BEGIN']
        data_end_pre = date_period['DATA_END']
        predict_begin = date_period['PREDICT_BEGIN']
        predict_end = date_period['PREDICT_END']



        # (1) data_begin and data_end may be changed here.        
        # Update，增加df_normalize
        
        data_raw = sam.get_model_data(data_begin=data_begin_pre,
                                      data_end=data_end_pre,
                                      data_period=None, 
                                      predict_end=predict_end, 
                                      predict_period=predict_period,
                                      stock_symbol=target_symbol,
                                      lag=lag)
        
        data_begin = data_raw['DATA_BEGIN']
        data_end = data_raw['DATA_END']        
        
        
        if isinstance(data_raw, str) :
            error_msg.append(data_raw)
            continue
        
        
        # Update, set to global variables?
        model_data = data_raw['MODEL_DATA']
        predict_data = data_raw['PRECIDT_DATA']
        predict_date = data_raw['PRECIDT_DATE']
        model_x = data_raw['MODEL_X']
        model_y = data_raw['MODEL_Y']            


        # Buy Price ......
        # Bug, 檢查這一段邏輯有沒有錯
        # BUG，BUY_PRICE可能有0
        
        # Bug, end_date可能沒有開盤，導致buy_price為空值
        # real_data = btm_load_data(begin_date=data_begin,
        #                           end_date=data_end)
        
        # real_data = real_data.drop(['HIGH', 'LOW'], axis=1)            
        buy_price = model_data[model_data['WORK_DATE']==data_begin] \
                    .rename(columns={'CLOSE':'BUY_PRICE'}) 
            
            
        buy_price = buy_price[['STOCK_SYMBOL', 'BUY_PRICE']] \
                    .reset_index(drop=True)
    
    
        if len(buy_price) == 0:
            print(str(data_begin) + ' without data.')
            continue


        
        # Model ......
        for j in range(0, len(model_list)):

            cur_model = model_list[j]
            
            # Model results .......
            # (1) Update, should deal with multiple signal issues.
            #     Currently, only consider the first signal.
            # (2) predict_data doesn't use.
            
            # global model_results_raw
            model_results_raw = cur_model(stock_symbol=stock_symbol,
                                          model_data=model_data, 
                                          predict_date=predict_date,
                                          model_x=model_x, model_y=model_y,
                                          remove_none=True)      
            
            
            if len(model_results_raw['RESULTS']) == 0:
                continue
            
            
            model_name = cur_model.__name__
            
            
            # Buy Signal
            temp_results = model_results_raw['RESULTS']
            temp_results = temp_results.merge(buy_price, 
                                                how='left',
                                                on='STOCK_SYMBOL')
            
            
            temp_results['ROI'] = (temp_results['CLOSE'] \
                                    - temp_results['BUY_PRICE']) \
                                    / temp_results['BUY_PRICE']


            temp_results = temp_results[temp_results['ROI'] >= roi_base]
            
            
            temp_results['MODEL'] = model_name
            temp_results['DATA_BEGIN'] = data_begin
            temp_results['DATA_END'] = data_end
            temp_results['PREDICT_BEGIN'] = predict_begin
            temp_results['PREDICT_END'] = predict_end
            temp_results['BACKTEST_ID'] = i
            
            buy_signal = buy_signal.append(temp_results)        
            
            
            # RMSE ......
            new_rmse = model_results_raw['RMSE']
            new_rmse['MODEL'] = model_name
            new_rmse['BACKTEST_ID'] = i
            rmse = rmse.append(new_rmse)
        
        
    
    if len(buy_signal) == 0:
        print('bt_buy_signal return 0 row.')
        return pd.DataFrame()
    
    
    buy_signal = buy_signal.rename(columns={'CLOSE':'FORECAST_CLOSE'})
    rmse = rmse.reset_index(drop=True)
    
    
    # Add dynamic ROI -------
    predict_roi_pre = cbyz.df_add_rank(buy_signal, 
                               value='WORK_DATE',
                               group_key=['STOCK_SYMBOL', 'BACKTEST_ID'])
            

    predict_roi = pd.DataFrame()
    
    
    for i in range(0, len(loc_time_seq)):
    
        df_lv1 = predict_roi_pre[predict_roi_pre['BACKTEST_ID']==i]
        unique_symbol = df_lv1['STOCK_SYMBOL'].unique()
        
        
        for j in range(0, len(unique_symbol)):
            
            df_lv2 = df_lv1[df_lv1['STOCK_SYMBOL']==unique_symbol[j]] \
                        .reset_index(drop=True)
            
            for k in range(0, len(df_lv2)):
                
                if k == 0:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'BUY_PRICE']
                    continue
                    
                if df_lv2.loc[k, 'FORECAST_CLOSE'] >= df_lv2.loc[k-1, 'MAX_PRICE']:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'FORECAST_CLOSE']
                else:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k-1, 'MAX_PRICE']
                            
            predict_roi = predict_roi.append(df_lv2)
    
    # ............
    predict_results = predict_roi.copy()
    predict_results.loc[predict_results['MAX_PRICE'] * stop_loss \
                        >= predict_results['FORECAST_CLOSE'], \
                        'SELL_SIGNAL'] = True
    
    predict_results = cbyz.df_conv_na(predict_results,
                                cols='SELL_SIGNAL',
                                value=False)
 
    predict_results['GRP_SELL_SIGNAL'] = predict_results \
        .groupby(['STOCK_SYMBOL', 'BACKTEST_ID'])['SELL_SIGNAL'] \
        .transform(max)
    
    
    
    predict_results.loc[(predict_results['GRP_SELL_SIGNAL']==True) \
                   & (predict_results['SELL_SIGNAL']==False), \
                   'REMOVE'] = True
    
    
    # Remove for GRP_SELL_SIGNAL == NA ......
    # Bug, 移除在period內完全沒碰到停損點的還沒判斷        
    predict_results['MAX_RANK'] = predict_results \
                    .groupby(['STOCK_SYMBOL', 'BACKTEST_ID'])['RANK'] \
                    .transform(max)

    predict_results.loc[(predict_results['GRP_SELL_SIGNAL']==False) \
                   & (predict_results['RANK']<predict_results['MAX_RANK']), \
                   'REMOVE'] = True
      
        
    if debug == False:
        predict_results = predict_results[predict_results['REMOVE'].isna()] \
                            .drop(['REMOVE', 'MAX_RANK'], axis=1)   
    
    predict_results = predict_results.reset_index(drop=True)


    predict_dict = {'RESULTS':predict_results, \
                    'RMSE':rmse}
    
    return predict_dict




def btm_add_hist_data(predict_results):
    
        
    # Organize results ------
    loc_forecast = predict_results.copy()

    loc_forecast = loc_forecast \
                    .rename(columns={'WORK_DATE':'BUY_DATE'})

    # Add Historical Data......
    hist_data_info = loc_forecast[['STOCK_SYMBOL', 'PREDICT_BEGIN',
                                   'PREDICT_END', 'FORECAST_CLOSE']] \
                    .reset_index(drop=True)

    
    hist_data_period = hist_data_info[['PREDICT_BEGIN', 'PREDICT_END']] \
                        .drop_duplicates() \
                        .reset_index(drop=True)
    
    
    hist_data_pre = pd.DataFrame()
        
    for i in range(0, len(hist_data_period)):

        temp_begin = hist_data_period.loc[i, 'PREDICT_BEGIN']
        temp_end = hist_data_period.loc[i, 'PREDICT_END']       
        
        # Symbol ...
        temp_symbol = hist_data_info[
            (hist_data_info['PREDICT_BEGIN']==temp_begin) \
            & (hist_data_info['PREDICT_END']==temp_end)]

        temp_symbol = temp_symbol['STOCK_SYMBOL'].tolist()  
            
            
        new_data = ar.stk_get_data(data_begin=temp_begin, 
                                   data_end=temp_end, 
                                   stock_type='tw', 
                                   stock_symbol=temp_symbol, 
                                   local=local)           
        
        new_data['PREDICT_BEGIN'] = temp_begin
        new_data['PREDICT_END'] = temp_end
        
        hist_data_pre = hist_data_pre.append(new_data)


    hist_data_pre = hist_data_pre.drop(['HIGH', 'LOW'], axis=1) \
                    .reset_index(drop=True)

    # hist_data_pre = cbyz.df_ymd(hist_data_pre, cols='WORK_DATE')

        
    # Combine data ......
    hist_data_full = hist_data_pre.merge(hist_data_info,
                                         how='left',
                                         on=['PREDICT_BEGIN', 'PREDICT_END', 
                                             'STOCK_SYMBOL']) 

        
    hist_data = hist_data_full[hist_data_full['FORECAST_CLOSE'] \
                          <= hist_data_full['CLOSE']] \
                .drop_duplicates(subset=['PREDICT_BEGIN', 'PREDICT_END', 
                                         'STOCK_SYMBOL']) \
                .reset_index(drop=True)
            
      
    # Join forecast and historical data ......
    # Bug, 這裡的資料抓出來可能會有na          


    backtest_results = loc_forecast \
                        .merge(hist_data, how='left', 
                               on=['STOCK_SYMBOL', 'PREDICT_BEGIN',
                                   'PREDICT_END', 'FORECAST_CLOSE'])
            
    
    backtest_results.loc[~backtest_results['CLOSE'].isna(), 'SUCCESS'] = True 
    backtest_results.loc[backtest_results['CLOSE'].isna(), 'SUCCESS'] = False     
        
        
    backtest_results = cbyz.df_ymd(backtest_results, 
                                cols=['BUY_DATE', 'PREDICT_BEGIN',
                                      'PREDICT_END'])    
    

        
    # Combine data ......
    hist_data_full = hist_data_pre.merge(hist_data_info,
                                         how='left',
                                         on=['PREDICT_BEGIN', 'PREDICT_END', 
                                             'STOCK_SYMBOL']) 

        
    hist_data = hist_data_full[hist_data_full['FORECAST_CLOSE'] \
                          <= hist_data_full['CLOSE']] \
                .drop_duplicates(subset=['PREDICT_BEGIN', 'PREDICT_END', 
                                         'STOCK_SYMBOL']) \
                .reset_index(drop=True)
            
      
    # Join forecast and historical data ......
    # Bug, 這裡的資料抓出來可能會有na          


    backtest_results = loc_forecast \
                        .merge(hist_data,
                               how='left',
                               on=['STOCK_SYMBOL', 'PREDICT_BEGIN',
                                   'PREDICT_END', 'FORECAST_CLOSE'])
            
    
    backtest_results.loc[~backtest_results['CLOSE'].isna(), 'SUCCESS'] = True 
    backtest_results.loc[backtest_results['CLOSE'].isna(), 'SUCCESS'] = False     
        
        
    backtest_results = cbyz.df_ymd(backtest_results, 
                                cols=['BUY_DATE', 'PREDICT_BEGIN',
                                      'PREDICT_END'])    
    

        
    # Combine data ......
    hist_data_full = hist_data_pre.merge(hist_data_info,
                                         how='left',
                                         on=['PREDICT_BEGIN', 'PREDICT_END', 
                                             'STOCK_SYMBOL']) 

        
    hist_data = hist_data_full[hist_data_full['FORECAST_CLOSE'] \
                          <= hist_data_full['CLOSE']] \
                .drop_duplicates(subset=['PREDICT_BEGIN', 'PREDICT_END', 
                                         'STOCK_SYMBOL']) \
                .reset_index(drop=True)
            
      
    # Join forecast and historical data ......
    # Bug, 這裡的資料抓出來可能會有na          


    backtest_results = loc_forecast \
                        .merge(hist_data,
                               how='left',
                               on=['STOCK_SYMBOL', 'PREDICT_BEGIN',
                                   'PREDICT_END', 'FORECAST_CLOSE'])
            
    
    backtest_results.loc[~backtest_results['CLOSE'].isna(), 'SUCCESS'] = True 
    backtest_results.loc[backtest_results['CLOSE'].isna(), 'SUCCESS'] = False     
        
        
    backtest_results = cbyz.df_ymd(backtest_results, 
                                cols=['BUY_DATE', 'PREDICT_BEGIN',
                                      'PREDICT_END'])    
    

        
    # Combine data ......
    hist_data_full = hist_data_pre.merge(hist_data_info,
                                         how='left',
                                         on=['PREDICT_BEGIN', 'PREDICT_END', 
                                             'STOCK_SYMBOL']) 

        
    hist_data = hist_data_full[hist_data_full['FORECAST_CLOSE'] \
                          <= hist_data_full['CLOSE']] \
                .drop_duplicates(subset=['PREDICT_BEGIN', 'PREDICT_END', 
                                         'STOCK_SYMBOL']) \
                .reset_index(drop=True)
            
      
    # Join forecast and historical data ......
    # Bug, 這裡的資料抓出來可能會有na          


    backtest_results = loc_forecast \
                        .merge(hist_data,
                               how='left',
                               on=['STOCK_SYMBOL', 'PREDICT_BEGIN',
                                   'PREDICT_END', 'FORECAST_CLOSE'])
            
    
    backtest_results.loc[~backtest_results['CLOSE'].isna(), 'SUCCESS'] = True 
    backtest_results.loc[backtest_results['CLOSE'].isna(), 'SUCCESS'] = False     
        
        
    backtest_results = cbyz.df_ymd(backtest_results, 
                                cols=['BUY_DATE', 'PREDICT_BEGIN',
                                      'PREDICT_END'])    
    

        
    # Combine data ......
    hist_data_full = hist_data_pre.merge(hist_data_info,
                                         how='left',
                                         on=['PREDICT_BEGIN', 'PREDICT_END', 
                                             'STOCK_SYMBOL']) 

        
    hist_data = hist_data_full[hist_data_full['FORECAST_CLOSE'] \
                          <= hist_data_full['CLOSE']] \
                .drop_duplicates(subset=['PREDICT_BEGIN', 'PREDICT_END', 
                                         'STOCK_SYMBOL']) \
                .reset_index(drop=True)
            
      
    # Join forecast and historical data ......
    # Bug, 這裡的資料抓出來可能會有na          


    backtest_results = loc_forecast \
                        .merge(hist_data,
                               how='left',
                               on=['STOCK_SYMBOL', 'PREDICT_BEGIN',
                                   'PREDICT_END', 'FORECAST_CLOSE'])
            
    
    backtest_results.loc[~backtest_results['CLOSE'].isna(), 'SUCCESS'] = True 
    backtest_results.loc[backtest_results['CLOSE'].isna(), 'SUCCESS'] = False     
        
        
    backtest_results = cbyz.df_ymd(backtest_results, 
                                cols=['BUY_DATE', 'PREDICT_BEGIN',
                                      'PREDICT_END'])    
    
    
    # Reorganize ......
    backtest_results = backtest_results \
                    .sort_values(by=['STOCK_SYMBOL', 'BUY_DATE']) \
                    .reset_index(drop=True)
    
    
    return backtest_results


# .............
    

def btm_gen_actions(predict_results, rmse):
    

                
    loc_rmse = rmse.copy()
    
    loc_rmse['RMSE_MEAN'] = loc_rmse \
                                .groupby(['STOCK_SYMBOL', 'MODEL'])['RMSE'] \
                                .transform('mean')
    
    
    
    loc_rmse['RMSE_MEAN_MIN'] = loc_rmse \
                                .groupby(['STOCK_SYMBOL', 'MODEL'])['RMSE'] \
                                .transform('min')


    loc_rmse['RMSE_MEAN_MAX'] = loc_rmse \
                                .groupby(['STOCK_SYMBOL', 'MODEL'])['RMSE'] \
                                .transform('max')


    
    # loc_rmse.loc[loc_rmse['RMSE_MEAN']==loc_rmse['RMSE_MEAN_MIN'],
    #     'ACTION'] = True 


    # Check, 這個方式是否合理
    loc_rmse = loc_rmse \
                .drop_duplicates(subset=['STOCK_SYMBOL', 'MODEL']) \
                .sort_values(by=['STOCK_SYMBOL', 'MODEL']) \
                .reset_index(drop=True)
                
    # ......
    
    loc_main = predict_results.merge(loc_rmse, how='left',
                            on=['MODEL', 'STOCK_SYMBOL'])
    
    # loc_main = loc_main[loc_main['ACTION']==True]
    
    
    print('BUG,BUY_PRICE可能有0')
    
    
    return_dict = {'SUMMARY':loc_rmse,
                   'RESULTS':loc_main}
    
    return return_dict


# ...............


def master(begin_date=20190401, periods=5, stock_symbol=None, stock_type='tw',
           signal=None, budget=None, split_budget=False):
    '''
    主工作區
    Update, 增加台灣上班上課行事曆，如果是end_date剛好是休假日，直接往前推一天。
    '''
    
    # fee = get_stock_fee()

    
    # Variables    
    # (1) Fix missing date issues
    # begin_date = 20170401
    # stock_type='tw'
    
    
    
    # target_symbol
    # (1) Bug, top=150時會出錯
    # stock_symbol = ['0050', '0056']
    # stock_symbol = ['2301', '2474', '1714', '2385']
    
    # global stock_symbol
    # stock_symbol = [2520, 2605, 6116, 6191, 3481, 2409]
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, to_type='str')
    
    
    btm_get_stock_symbol(stock_symbol=stock_symbol,
                         top=100) 


    
    full_data = sam.sam_load_data(data_begin=None, data_end=None, 
                                  stock_type=stock_type, period=None, 
                                  stock_symbol=stock_symbol, 
                                  lite=False, full=True)

    full_data = full_data['FULL_DATA']
    
    
    # Predict ------
    # (1) Bug, predict_main裡面可能會有duplicated rows by stock_symbole and 
    # model and backtest_id
    predict_main_raw = btm_predict(begin_date=begin_date,
                                   data_period=365, volume=1000, 
                                   budget=None, predict_period=14, 
                                   backtest_times=10, debug=False)
    
    predict_main = predict_main_raw['RESULTS']
    predict_rmse = predict_main_raw['RMSE']
    

    
    # Backtest Report ------
    backtest_main = btm_add_hist_data(predict_results=predict_main)

    

    # Action Insights ------
    actions = btm_gen_actions(predict_results=predict_main,
                              rmse=predict_rmse)
    

    # Update 2
    # 00:28 Lien 連祥宇 我剛剛想了一下 目前有個小問題。公式可能需要設一下時間差。台股一天漲幅
    # 最大10啪 如果漲停 按你的公式 他會回跌2趴的時候出場。可是如果只有漲2趴 回跌0.4趴的時候
    # 你的公式就會出場賣出 可是一般同日買賣 0.4啪算是很平常的小波動。又如當日2啪跌到1啪
    # 是很正常的波動範圍 可是2啪跌到1啪已經是回檔5成了
    # 00:29 Lien 連祥宇 所以我在想是否兩最高價之間需要設立時間差？ 
    # 如每日計算一次之類的（我個人覺得每日還算太頻繁）否著你會過度交易 一天進出買賣100次之類的 手續費會直接讓你賠大錢
    
    
    # Update 3, how to generate action?
    # Update 4, record rate of win
    
    
    # Predictions -----
    predict_date = cbyz.date_gat_calendar(begin_date=20210406, 
                                          end_date=20210420)
    
    predict_date = predict_date[['WORK_DATE']]
    predict_date['PREDICT'] = True
    
    predict_date_list = predict_date['WORK_DATE'].tolist()

    
    
    predict_full_data = full_data.merge(predict_date, how='outer',
                                  on='WORK_DATE')
    
# predict_date = predict_date['WORK_DATE'].tolist()
    
    
    prediction = sam.model_2(model_data=predict_full_data, 
                             predict_data=predict_data, 
                             predict_date=predict_date_list,
                             model_x=model_x, model_y=model_y, 
                             stock_symbol=stock_symbol, remove_none=True)
        
    
    return ''






# ..............=


def check():
    '''
    資料驗證
    '''    
    return ''



if __name__ == '__main__':
    results = master(begin_date=20180401)



    
    
# begin_date = 20170102
# days=60
# volume=None
# budget=None
# roi_base = 0.02    
# data_period=90
# interval=30
# volume=1000
# budget=None
# predict_period=15
# backtest_times=5
# roi_base=0.03
# stop_loss=0.9    
# stock_type = 'tw'
    
    
    

