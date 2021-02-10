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
    

# begin_date = 20180701
# days=60
# volume=None
# budget=None
# roi_base = 0.02
    


def btm_predict(begin_date, model_data_period=85, volume=1000, budget=None, 
                predict_period=15, backtest_times=5,
                roi_base=0.03, stop_loss=0.9):
    


    
    # .......
    
    loc_time_seq = cbyz.time_get_seq(begin_date=begin_date,
                                     periods=backtest_times,
                                     unit='d', skip=model_data_period,
                                     simplify_date=True)
    
    loc_time_seq = loc_time_seq['WORK_DATE'].tolist()


    # ......
    
    



    
    # Work area ----------
    model_list = sam.get_model_list()
    buy_signal = pd.DataFrame()
    error_msg = []
    
    
    # Date .........
    for i in range(0, len(loc_time_seq)):
    

        date_period = cbyz.date_get_period(data_begin=loc_time_seq[i], 
                                           data_end=None, 
                                           data_period=model_data_period,
                                           predict_end=None, 
                                           predict_period=predict_period)        
        

        data_begin = date_period['DATA_BEGIN']
        data_end = date_period['DATA_END']
        predict_begin = date_period['PREDICT_BEGIN']
        predict_end = date_period['PREDICT_END']

        
        # Update, transfer this model out side of loop.
        data_raw = sam.get_model_data(data_begin=data_begin,
                                      data_end=data_end,
                                      data_period=None, 
                                      predict_end=predict_end, 
                                      predict_period=predict_period,
                                      stock_symbol=target_symbol)
        
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
        
        # Bug, end_date可能沒有開盤，導致buy_price為空值
        # real_data = btm_load_data(begin_date=data_begin,
        #                           end_date=data_end)
        
        # real_data = real_data.drop(['HIGH', 'LOW'], axis=1)            
        buy_price = model_data[model_data['WORK_DATE']==data_begin] \
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
            model_results_raw = cur_model(model_data=model_data, 
                                          predict_data=predict_data,
                                          predict_date=predict_date,
                                          model_x=model_x, model_y=model_y,
                                          remove_none=True)                
            
            
            
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
            temp_results['PREDICT_BEGIN'] = predict_begin
            temp_results['PREDICT_END'] = predict_end
            temp_results['BACKTEST_ID'] = i
            
            buy_signal = buy_signal.append(temp_results)        
        
        
        
        
        
        
        
        










        
        # # ......
        # # Bug, end_date可能沒有開盤，導致buy_price為空值
        # real_data = btm_load_data(begin_date=data_begin,
        #                           end_date=data_end)
        
        # real_data = real_data.drop(['HIGH', 'LOW'], axis=1)        
        
    
        # # Buy Price
        # # Bug, 檢查這一段邏輯有沒有錯
        # buy_price = real_data[real_data['WORK_DATE']==data_begin] \
        #             .rename(columns={'CLOSE':'BUY_PRICE'}) 
            
            
        # buy_price = buy_price[['STOCK_SYMBOL', 'BUY_PRICE']] \
        #             .reset_index(drop=True)
    
    
        # if len(buy_price) == 0:
            
        #     print(str(loc_time_seq[i]) + ' without data.')
        #     continue

    
        # # Model ......
        # for j in range(0, len(model_list)):

        #     cur_model = model_list[j]
            
        #     # Model results .......
        #     # (1) Update, should deal with multiple signal issues.
        #     #     Currently, only consider the first signal.
        #     # (2) Add data
            
        #     # global model_results_raw
        #     model_results_raw = cur_model(data_begin=data_begin,
        #                                   data_end=data_end,
        #                                   stock_symbol=target_symbol,
        #                                   predict_period=predict_period,
        #                                   remove_none=False)                
            
            
            
        #     temp_results = model_results_raw['RESULTS']
        #     temp_results = temp_results.merge(buy_price, 
        #                                         how='left',
        #                                         on='STOCK_SYMBOL')
            
            
        #     temp_results['ROI'] = (temp_results['CLOSE'] \
        #                             - temp_results['BUY_PRICE']) \
        #                             / temp_results['BUY_PRICE']


        #     temp_results = temp_results[temp_results['ROI'] >= roi_base]
        #     # \
        #     #     .drop_duplicates(subset='STOCK_SYMBOL')
            
            
        #     temp_results['MODEL'] = cur_model.__name__
        #     temp_results['PREDICT_BEGIN'] = predict_begin
        #     temp_results['PREDICT_END'] = predict_end
        #     temp_results['BACKTEST_ID'] = i
            
        #     buy_signal = buy_signal.append(temp_results)
            
            
            
            
            
            
            
            
            
    
    if len(buy_signal) == 0:
        print('bt_buy_signal return 0 row.')
        return pd.DataFrame()
    
    
    buy_signal = buy_signal.rename(columns={'CLOSE':'FORECAST_CLOSE'})
    
    
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
                    
                if df_lv2.loc[k, 'FORECAST_CLOSE'] >= df_lv2.loc[k-1, 'MAX_PRICE']:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'FORECAST_CLOSE']
                else:
                    df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k-1, 'MAX_PRICE']
                            
            forecast_roi = forecast_roi.append(df_lv2)
    
    # ............
    global forecast_results
    forecast_results = forecast_roi.copy()
    forecast_results.loc[forecast_results['MAX_PRICE'] * stop_loss
                         >= forecast_results['FORECAST_CLOSE'],
                         'SELL_SIGNAL'] = True
    
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

    
    return ''




def btm_add_hist_data():
    
        
    # Organize results ------
    loc_forecast = forecast_results.copy()

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
            
            
        new_data = ar.stk_get_data(begin_date=temp_begin, 
                                   end_date=temp_end, 
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
                
    global backtest_main
    backtest_main = loc_forecast \
                    .merge(hist_data,
                           how='left',
                           on=['STOCK_SYMBOL', 'PREDICT_BEGIN',
                               'PREDICT_END', 'FORECAST_CLOSE'])
        
    
    backtest_main.loc[~backtest_main['CLOSE'].isna(), 'SUCCESS'] = True 
    backtest_main.loc[backtest_main['CLOSE'].isna(), 'SUCCESS'] = False     
        
        
    backtest_main = cbyz.df_ymd(backtest_main, 
                                cols=['BUY_DATE', 'PREDICT_BEGIN',
                                      'PREDICT_END'])    
    
    
    return ''




def master(begin_date=20190401, periods=5, stock_symbol=None,
           signal=None, budget=None, split_budget=False):
    '''
    主工作區
    Update, 增加台灣上班上課行事曆，如果是end_date剛好是休假日，直接往前推一天。
    '''
    
    # fee = get_stock_fee()

    
    # Variables    
    # (1) Fix missing date issues
    # begin_date = 20190401
    
    
    
    
    # target_symbol
    # (1) Bug, top=150時會出錯
    # stock_symbol = ['0050', '0056']
    stock_symbol = ['2301', '2474', '1714', '2385']
    
    
    btm_get_stock_symbol(stock_symbol=stock_symbol,
                         top=100) 

   
    # forecast_results
    btm_predict(begin_date=begin_date,
                 model_data_period=180, volume=1000, 
                 budget=None, predict_period=14, 
                 backtest_times=5)
    
    
    # backtest_main
    btm_add_hist_data()
    
    return ''


# ..............


def check():
    '''
    資料驗證
    '''    
    return ''



if __name__ == '__main__':
    results = master(begin_date=20180401)



# periods=5
# signal=None
# budget=None
# split_budget=False
# days=60
# roi_base = 0.02
# stock_symbol=['0050', '0056']
# model_data_period=60


# volume=1000
# predict_period=30
# backtest_times=5
# roi_base=0.015
    
    
# begin_date = 20170102
# days=60
# volume=None
# budget=None
# roi_base = 0.02    
    
    



