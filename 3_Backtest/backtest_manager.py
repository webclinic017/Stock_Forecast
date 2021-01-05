#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:09 2020

@author: Aron
"""


# Worklist
# 1.Add price increse but model didn't catch
# 2.Retrieve one symbol historical data to ensure calendar




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




def btm_load_data(begin_date, end_date=None):
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
    

# begin_date = 20180701
# days=60
# volume=None
# budget=None
# roi_base = 0.02
    


def forecast_single(begin_date, stock_symbol=['0050', '0056'],
                    model_data_period=90, volume=1000, budget=None, 
                    forecast_period=15, backtest_times=5,
                    roi_base=0.03, stop_loss=0.8):
    
    
    # .......
    
    
    BUGGGG, 這支function有問題，導致每次出來的結果都一樣
    loc_time_seq = cbyz.get_time_seq(begin_date=begin_date,
                                 periods=backtest_times,
                                 unit='d',
                                 simplify_date=True)
    
    loc_time_seq = loc_time_seq['WORK_DATE'].tolist()

    
    # Work area ----------
    model_list = sam.get_model_list()
    buy_signal = pd.DataFrame()
    
    
    # Date .........
    for i in range(0, len(loc_time_seq)):
        
        loop_begin_date = loc_time_seq[i]
        loop_end_date = cbyz.date_cal(loop_begin_date, 
                                 amount=model_data_period, 
                                 unit='d')
        
        # ......
        # Bug, end_date可能沒有開盤，導致buy_price為空值
        real_data = btm_load_data(begin_date=loop_begin_date,
                              end_date=loop_end_date)
        
        real_data = real_data.drop(['HIGH', 'LOW'], axis=1)        
        
    
        # Buy Price
        # Bug, 檢查這一段邏輯有沒有錯
        buy_price = real_data[real_data['WORK_DATE']==loop_begin_date] \
            .rename(columns={'CLOSE':'BUY_PRICE'}) 
            
        buy_price = buy_price[['STOCK_SYMBOL', 'BUY_PRICE']] \
                    .reset_index(drop=True)
    
    
        if len(buy_price) == 0:
            
            print(str(loc_time_seq[i]) + ' without data.')
            continue
    
        # Model ......
        for j in range(0, len(model_list)):

            cur_model = model_list[j]
            
            # Model results .......
            # (1) Update, should deal with multiple signal issues.
            #     Currently, only consider the first signal.
            # (2) Add data
            
            # global model_results_raw
            model_results_raw = cur_model(data_begin=loop_begin_date,
                                          data_end=loop_end_date,
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


            temp_results = temp_results[temp_results['ROI'] >= roi_base]
            # \
            #     .drop_duplicates(subset='STOCK_SYMBOL')
            
            
            temp_results['MODEL'] = cur_model.__name__
            temp_results['FORECAST_BEGIN'] = loop_begin_date
            temp_results['FORECAST_END'] = loop_end_date
            temp_results['BACKTEST_ID'] = i
            
            buy_signal = buy_signal.append(temp_results)
            
            
    
    if len(buy_signal) == 0:
        print('bt_buy_signal return 0 row.')
        return pd.DataFrame()
    
    
    # Add dynamic ROI -------
    # dynamic_roi = backtest_results_pre.copy()
    
    forecast_roi_pre = cbyz.df_add_rank(buy_signal, 
                               value='WORK_DATE',
                               group_key=['STOCK_SYMBOL', 'BACKTEST_ID'])
            

    forecast_roi = pd.DataFrame()
    
    
    for i in range(0, len(loc_time_seq)):
    
        df_lv1 = forecast_roi_pre[forecast_roi_pre['BACKTEST_ID']==i]
        
        unique_symbol = df_lv1['STOCK_SYMBOL'].unique()
        
        
        for j in range(0, len(unique_symbol)):
            
            df_lv2 = df_lv1[df_lv1['STOCK_SYMBOL']==unique_symbol[j]] \
                        .reset_index(drop=True)
            
            for k in range(0, len(df_lv2)):
                
                if k == 0:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'BUY_PRICE']
                    continue
                    
                if df_lv2.loc[k, 'CLOSE'] >= df_lv2.loc[k-1, 'MAX_PRICE']:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'CLOSE']
                else:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k-1, 'MAX_PRICE']
                            
            
            forecast_roi = forecast_roi.append(df_lv2)

    
    # ............
    global forecast_results
    forecast_results = forecast_roi.copy()
    forecast_results.loc[forecast_results['MAX_PRICE'] * stop_loss
                         >= forecast_results['CLOSE'], 'SELL_SIGNAL'] = True
    
    forecast_results = cbyz.df_conv_na(forecast_results,
                                cols='SELL_SIGNAL',
                                value=False)
 
    forecast_results['GRP_SELL_SIGNAL'] = forecast_results \
        .groupby(['STOCK_SYMBOL', 'BACKTEST_ID'])['SELL_SIGNAL'] \
        .transform(max)
    
    
    
    forecast_results.loc[(forecast_results['GRP_SELL_SIGNAL']==True) \
                   & (forecast_results['SELL_SIGNAL']==False), 
                   'REMOVE'] = True
    
    
    # Remove for GRP_SELL_SIGNAL == NA ......
    # 移除在period內完全沒碰到停損點的還沒判斷        
    forecast_results['MAX_RANK'] = forecast_results \
                    .groupby(['STOCK_SYMBOL', 'BACKTEST_ID'])['RANK'] \
                    .transform(max)

    forecast_results.loc[(forecast_results['GRP_SELL_SIGNAL']==False) \
                   & (forecast_results['RANK']<forecast_results['MAX_RANK']),
                   'REMOVE'] = True
      
    
    forecast_results = forecast_results[forecast_results['REMOVE'].isna()] \
                .drop(['REMOVE', 'MAX_RANK'], axis=1) \
                .reset_index(drop=True)

    
    return forecast_results




def add_historical_data():
    

        
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
    return ''





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
    
    
    # time_seq = cbyz.get_time_seq(begin_date=begin_date,
    #                   periods=periods,
    #                   unit='m', 
    #                   simplify_date=True)
   
    # # Backtest ----------
    # global backtest_results
    # backtest_results = pd.DataFrame()    
    
    # for i in range(0, len(time_seq)):
        
    #     single = forecast_single(begin_date=time_seq.loc[i, 'WORK_DATE'],
    #                               model_data_period=60, volume=1000, budget=None, 
    #                               forecast_period=30, backtest_times=5)
        
    #     backtest_results = backtest_results.append(single)

        
    # backtest_results = backtest_results \
    #                     .reset_index(drop=True)
                        
    backtest_results = forecast_single(begin_date=begin_date,
                                  model_data_period=90, volume=1000, budget=None, 
                                  forecast_period=15, backtest_times=5)                        
    
    
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
# roi_base = 0.02
# stock_symbol=['0050', '0056']
# model_data_period=60


# volume=1000
# forecast_period=30
# backtest_times=5
# roi_base=0.015
    
    
# begin_date = 20190102
# days=60
# volume=None
# budget=None
# roi_base = 0.02    