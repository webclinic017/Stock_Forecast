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
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/3_Backtest'
    master_path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast'
else:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'
    master_path = '/home/aronhack/stock_forecast'
    


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 master_path + '/2_Stock_Analysis',
                 master_path + '/Function', 
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ']


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
    
    
    global stock_symbol, stock_type
    bt_seq = cbyz.date_get_seq(begin_date=bt_last_begin,
                               seq_length=bt_times,
                               unit='d', interval=-interval,
                               simplify_date=True)
    
    bt_seq_list = bt_seq['WORK_DATE'].tolist()
    
    
    # Work area ----------
    global bt_results, bt_rmse, model_y
    bt_results_raw = pd.DataFrame()
    bt_rmse = pd.DataFrame()
    
    
    # Predict ......
    for i in range(0, len(bt_seq_list)):
        
        begin = bt_seq_list[i]
        
        results_raw = sam.master(_predict_begin=begin, 
                                 _predict_end=None, 
                                 _predict_period=predict_period,
                                 data_period=data_period, 
                                 _stock_symbol=stock_symbol)


        new_results = results_raw[0]
        new_results['BACKTEST_ID'] = i
        
        new_rmse = results_raw[1]
        new_rmse['BACKTEST_ID'] = i
        
        bt_results_raw = bt_results_raw.append(new_results)
        bt_rmse = bt_rmse.append(new_rmse)


    # Organize ......
    global model_y
    bt_results = bt_results_raw.reset_index(drop=True)
    model_y = cbyz.df_get_cols_except(df=bt_results, 
                                      except_cols=['STOCK_SYMBOL', 'WORK_DATE', 
                                                   'MODEL', 'BACKTEST_ID'])

    bt_rmse = bt_rmse \
        .sort_values(by=['MODEL', 'Y']) \
        .reset_index(drop=True)
    

# ............



def backtest_verify(begin_date, actions, stock_symbol, stock_type='tw', 
                    stop_loss=0.8, buffer_days=7):
    
    
    # Historic Data
    hist_data = stk.get_data(data_begin=begin_date, data_end=None,
                             stock_symbol=stock_symbol,
                             stock_type=stock_type,
                             shift=0)
    
    
    bt_results = pd.DataFrame()
    for i in range(len(actions)):

        action_date = actions.loc[i, 'WORK_DATE']
        action_symbol = actions.loc[i, 'STOCK_SYMBOL']
        
        
        
        cur_data = hist_data[(hist_data['WORK_DATE']>action_date) \
                             & (hist_data['STOCK_SYMBOL']==action_symbol)] \
                        .reset_index(drop=True)

        cur_data['SELL_SIGNAL'] = False
                        
            
        # Buy Price
        buy_price =  hist_data[(hist_data['WORK_DATE']==action_date) \
                             & (hist_data['STOCK_SYMBOL']==action_symbol)] \
                        .reset_index(drop=True)
    
        buy_price = buy_price.loc[0, 'CLOSE']
        
        # Highest Prices
        highest_price = buy_price
    

        # Wrap This as function
        for j in range(len(cur_data)):

            
            cur_price = cur_data.loc[i, 'CLOSE']

            
            if cur_price > highest_price:
                highest_price = cur_price
            
            
            # Get Sell Signal
            if (j > buffer_days) \
                and ((highest_price - cur_price) \
                     >= (highest_price - buy_price) * (1 - stop_loss)):
                    
                    
                new_results = cur_data.loc[j, :]
                new_results['BUY_PRICE'] = buy_price
                new_results['HIGHEST_PRICE'] = highest_price
                new_results['SELL_DATE'] = cur_data.loc[j, 'WORK_DATE']
                new_results['BUY_DATE'] = action_date
                new_results['SELL_SIGNAL'] = True
                
                bt_results = bt_results.append(new_results)
                break
            
            # Not Get Sell Signal
            if j == len(cur_data) - 1:
                
                new_results = cur_data.loc[j, :]
                new_results['BUY_PRICE'] = buy_price
                new_results['HIGHEST_PRICE'] = highest_price
                new_results['SELL_DATE'] = np.nan
                new_results['BUY_DATE'] = action_date
                new_results['SELL_SIGNAL'] = False
            
                bt_results = bt_results.append(new_results)


    # I don't know why SELL_SIGNAL in bt_results will convert to 0/1 from 
    # True/False automatically.
    bt_results = bt_results \
        .rename(columns={'WORK_DATE':'LAST_DATE'}) \
        .reset_index(drop=True)
        
        
    bt_results['PROFIT'] = bt_results['CLOSE'] - bt_results['BUY_PRICE']     
    bt_results['ROI'] = bt_results['PROFIT'] / bt_results['BUY_PRICE']
        
        
    bt_results = bt_results[['STOCK_SYMBOL', 'BUY_DATE', 'BUY_PRICE',
                             'SELL_SIGNAL', 'SELL_DATE', 'HIGHEST_PRICE',
                             'LAST_DATE', 'HIGH', 'CLOSE', 'LOW', 'VOLUME',
                             'PROFIT', 'ROI']]

    
    return bt_results




def cal_profit(price_thld=2, time_thld=10, rmse_thld=0.15):
    
    
    global predict_period, bt_last_begin
    global bt_results, bt_rmse, bt_main, actions, model_y
    global stock_symbol, stock_type


    print('Bug, data_begin and data_end should follow the backtest range')
    loc_begin = 20160101
    loc_end = cbyz.date_cal(bt_last_begin, -1, 'd')
    hist_data_raw = stk.get_data(data_begin=loc_begin, 
                                 data_end=loc_end, 
                                 stock_type=stock_type, 
                                 stock_symbol=stock_symbol, 
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
    
    symbol_df = pd.DataFrame({'STOCK_SYMBOL':stock_symbol})
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

    # Add predict rmse


    # Prepare columns ......
    hist_cols = [i + '_HIST' for i in model_y]
    hist_cols_dict = cbyz.li_to_dict(model_y, hist_cols)
    hist_data = hist_data.rename(columns=hist_cols_dict)


    last_cols = [i + '_LAST' for i in model_y]
    last_cols_dict = cbyz.li_to_dict(model_y, last_cols)
    
    
    # Organize ......
    main_data = bt_results.merge(hist_data, how='left', 
                              on=['WORK_DATE', 'STOCK_SYMBOL'])


    main_data = main_data[['BACKTEST_ID', 'STOCK_SYMBOL', 'MODEL', 
                           'WORK_DATE', 'LAST_DATE'] \
                          + model_y + hist_cols + last_cols]

    # Fill na in the prediction period.
    for i in hist_cols:
        main_data[i] = np.where(main_data['WORK_DATE'] >= bt_last_begin,
                                np.nan, main_data[i])

    # Check na ......
    chk = cbyz.df_chk_col_na(df=main_data, positive_only=True)
    
    if len(chk) > len(model_y):
        print('Err01. cal_profit - main_data has na in columns.')


    bt_main, actions = \
        stk.gen_predict_action(df=main_data, rmse=bt_rmse,
                                  date='WORK_DATE', 
                                  last_date='LAST_DATE', 
                                  y=model_y, y_last=last_cols,
                                  price_thld=price_thld, time_thld=time_thld,
                                  rmse_thld=rmse_thld)

    # df = main_data.copy()
    # rmse = bt_rmse.copy()
    # date = 'WORK_DATE' 
    # last_date = 'LAST_DATE'
    # y = model_y
    # y_last = last_cols
        
    # price_thld = 1
    # time_thld = predict_period
    # rmse_thld = 0.15    


    # return bt_main, actions





def master(_bt_last_begin, predict_period=14, interval=360, bt_times=5, 
           data_period=5, _stock_symbol=None, _stock_type='tw',
           signal=None, budget=None, split_budget=False):
    '''
    主工作區
    Update, 增加台灣上班上課行事曆，如果是end_date剛好是休假日，直接往前推一天。
    '''
    
    
    
    # Parameters
    _bt_last_begin = 20210626
    # bt_last_begin = 20210211
    predict_period = 5
    interval = 60
    bt_times = 1
    data_period = 360 * 7
    _stock_symbol = [2520, 2605, 6116, 6191, 3481, 2409, 2603]
    _stock_type = 'tw'


    path_sam = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/2_Stock_Analysis/Export'
    target_symbols = pd.read_csv(path_sam \
                                 + '/target_symbols_20210627_230214.csv')

    _stock_symbol = target_symbols['STOCK_SYMBOL'].tolist()    

    
    global stock_symbol, bt_last_begin
    stock_symbol = _stock_symbol
    bt_last_begin = _bt_last_begin
    
    stock_type = _stock_type    
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, to_type='str')

    
    # full_data = sam.sam_load_data(data_begin=None, data_end=None, 
    #                               stock_type=stock_type, period=None, 
    #                               stock_symbol=stock_symbol, 
    #                               lite=False, full=True)


    # full_data = full_data['FULL_DATA']
    
    
    # Predict ------
    global bt_results, bt_rmse
    backtest_predict(bt_last_begin=bt_last_begin, 
                     predict_period=predict_period, 
                     interval=interval,
                     bt_times=bt_times,
                     data_period=data_period)

    
    # Profit ------    
    # price_thld = 2
    # time_thld = 10
    # rmse_thld = 0.15
    
    global bt_main, actions    
    cal_profit(price_thld=1, time_thld=predict_period, rmse_thld=0.10)
    actions = actions[actions['MODEL']=='model_6']
    actions = actions.drop('ROWS', axis=1)
    actions = cbyz.df_add_size(df=actions, group_by='STOCK_SYMBOL',
                               col_name='ROWS')

    time_serial = cbyz.get_time_serial(with_time=True)
    actions.to_excel(path_export + '/actions_' + time_serial + '.xlsx', 
                     index=False, encoding='utf-8-sig')

    
    # # Predictions -----
    # predict_date = cbyz.date_gat_calendar(begin_date=20210406, 
    #                                       end_date=20210420)
    
    # predict_date = predict_date[['WORK_DATE']]
    # predict_date['PREDICT'] = True
    
    # predict_date_list = predict_date['WORK_DATE'].tolist()

    
    
    # predict_full_data = full_data.merge(predict_date, how='outer',
    #                               on='WORK_DATE')
    

        
    
    return ''


# ..............=


def check():
    '''
    資料驗證
    '''    
    
    # Err01
    chk = main_data[main_data['HIGH_HIST'].isna()]   
    chk
    
    return ''



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

def backtest_master(predict_begin=20200801, predict_period=15,
                    interval=30,
                volume=1000, budget=None, 
                stock_symbol=[2520, 2605, 6116, 6191, 3481, 2409],
                stock_type='tw',
                backtest_times=14,
                signal_thld=0.03, stop_loss=0.8, buffer_days=7, lag=7, debug=False):
    
    
    pred_results, actions = \
        backtest_predict(begin=20190801, predict_period=15,
                         interval=30, data_period=360, 
                        volume=1000, budget=None, 
                         stock_symbol=[2520, 2605, 6116, 6191, 3481, 2409],
                         stock_type='tw',
                         backtest_times=5,
                         signal_thld=0.03, lag=7, debug=False)    

    
    
    bt_results = backtest_verify(begin_date=begin_date, actions=actions, 
                                 stock_symbol=stock_symbol, 
                                 stop_loss=stop_loss,
                                 buffer_days=buffer_days)
    
    
    
    return pred_results, actions, bt_results



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
    loc_time_seq = cbyz.date_get_seq(begin_date=begin_date,
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
    loc_time_seq = cbyz.date_get_seq(begin_date=begin_date,
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


# ...............




