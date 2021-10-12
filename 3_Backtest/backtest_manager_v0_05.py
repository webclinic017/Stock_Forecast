#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

History

20210703 - Replaced MA with WMA in SAM

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

host = 0
stock_type = 'tw'

# Path .....
if host == 0:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/3_Backtest'
    path_sam = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/2_Stock_Analysis'

elif host == 2:
    path = '/home/jupyter/3_Backtest'
    path_sam = '/home/jupyter/2_Stock_Analysis'    

# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path + '/Function',
                 path_sam]


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk
# import stock_analysis_manager_v1_02 as sam
# import stock_analysis_manager_v1_03 as sam
# import stock_analysis_manager_v1_04 as sam
# import stock_analysis_manager_v1_05 as sam
import stock_analysis_manager_v1_06 as sam



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     



def set_calendar(_bt_last_begin, predict_period):

    global calendar, bt_last_begin, bt_last_end    
    bt_last_begin = _bt_last_begin
    calendar_end = cbyz.date_cal(bt_last_begin, predict_period + 20, 'd')
    
    calendar = stk.get_market_calendar(begin_date=20140101, 
                                       end_date=calendar_end, 
                                       stock_type=stock_type)
    
    calendar['TRADE'] = np.where(calendar['TRADE_DATE']==0, 0, 1)
    
    
    calendar, _ = cbyz.df_add_shift(df=calendar, cols='WORK_DATE',
                                    shift=predict_period - 1, 
                                    group_by=['TRADE'], 
                                    suffix='_LAST', remove_na=False)
    
    bt_last_end = calendar[calendar['WORK_DATE_LAST']==bt_last_begin]
    bt_last_end = int(bt_last_end['WORK_DATE'])    



# ..........
    

def backtest_predict(bt_last_begin, predict_period, interval, 
                     bt_times, data_period, load_model=False, cv=2,
                     fast=False):
    
    
    global stock_symbol, stock_type, bt_info
    
    # Prepare For Backtest Records ......
    print('backtest_predict - 這裡有bug，應該用global calendar')
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
    global bt_results, precision, features, model_y, volume_thld
    
    bt_results_raw = pd.DataFrame()
    precision = pd.DataFrame()
    features = pd.DataFrame()
    
    # Predict ......
    for i in range(0, len(bt_seq)):
        
        begin = bt_seq[i]

        results_raw = sam.master(_predict_begin=begin,
                                 _predict_end=None, 
                                 _predict_period=predict_period,
                                 _data_period=data_period, 
                                 _stock_symbol=stock_symbol,
                                 _ma_values=ma_values,
                                 _volume_thld=volume_thld,
                                 load_model=load_model,
                                 cv=cv, fast=fast)


        new_result = results_raw[0]
        new_result['BACKTEST_ID'] = i
        
        new_precision = results_raw[1]
        new_precision['BACKTEST_ID'] = i
        
        # new_features = results_raw[2]
        # new_features['BACKTEST_ID'] = i
        
        bt_results_raw = bt_results_raw.append(new_result)
        precision = precision.append(new_precision)
        # features = features.append(new_features)


    # Organize ......
    global model_y
    bt_results = bt_results_raw.reset_index(drop=True)
    model_y = cbyz.df_get_cols_except(df=bt_results, 
                                      except_cols=['STOCK_SYMBOL', 'WORK_DATE', 
                                                   'BACKTEST_ID'])

# ............



def cal_profit(y_thld=2, time_thld=10, prec_thld=0.15, execute_begin=None,
               export_file=True, load_file=False, path=None, file_name=None,
               upload_metrics=False):
    '''
    應用場景有可能全部作回測，而不做任何預測，因為data_end直接設為bt_last_begin
    '''
    
    global predict_period
    global interval, bt_times 
    global bt_results, rmse, bt_main, actions, model_y
    global stock_symbol, stock_type
    global bt_last_begin, bt_last_end    
    global calendar


    # Prepare columns ......
    
    model_y_last = [i + '_LAST' for i in model_y]
    model_y_hist = [s + '_HIST' for s in model_y]
    
    # Y And OHLC
    ohlc = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 
            'OPEN_CHANGE_RATIO', 'HIGH_CHANGE_RATIO',
            'LOW_CHANGE_RATIO', 'CLOSE_CHANGE_RATIO']

    ohlc_hist = [i + '_HIST' for i in ohlc]
    ohlc_hist_dict = cbyz.li_to_dict(ohlc, ohlc_hist)    

    ohlc_last = [i + '_LAST' for i in ohlc]
    
    
    # close = 'CLOSE'
    close = 'CLOSE_CHANGE_RATIO'
    

    # Period ......
    # predict_period * 2是為了保險起見
    bt_first_begin = cbyz.date_cal(bt_last_begin, 
                                    -interval * bt_times - predict_period * 2, 
                                    'd')
    
    forecast_calendar = ar.get_calendar(begin_date=bt_last_begin, 
                                        end_date=bt_last_end,
                                        simplify=True)    
    
    forecast_calendar = forecast_calendar[['WORK_DATE']]
    forecast_calendar = forecast_calendar['WORK_DATE'].tolist()
    

    # Hist Data ......
    hist_data_raw = stk.get_data(data_begin=bt_first_begin, 
                                 data_end=bt_last_end, 
                                 stock_type=stock_type, 
                                 stock_symbol=stock_symbol, 
                                 price_change=True,
                                 tej=True)
    
    temp_cols = ['WORK_DATE', 'STOCK_SYMBOL'] + ohlc
    hist_data_raw = hist_data_raw[temp_cols]
    
 
    # Get Last Date ....        
    print('Optimize - 這裡的work_date從2014開始')
    date_last = calendar[calendar['TRADE_DATE']>0]
    date_last = cbyz.df_add_shift(df=date_last, cols='WORK_DATE', shift=1)
    
    date_last = date_last[0] \
                .dropna(subset=['WORK_DATE_PRE'], axis=0) \
                .rename(columns={'WORK_DATE_PRE':'LAST_DATE'})
                
    date_last = cbyz.df_conv_col_type(df=date_last, cols='LAST_DATE', 
                                      to='int')
    
    # ......
    if len(stock_symbol) > 0:
        symbol_df = pd.DataFrame({'STOCK_SYMBOL':stock_symbol})
    else:
        temp_symbol = hist_data_raw['STOCK_SYMBOL'].unique().tolist()
        symbol_df = pd.DataFrame({'STOCK_SYMBOL':temp_symbol})        

        
    # Merge hist data ......
    main_data_pre = cbyz.df_cross_join(date_last, symbol_df)
    main_data_pre = main_data_pre \
                .merge(hist_data_raw, how='left', on=['WORK_DATE', 'STOCK_SYMBOL'])
    
    # Add last price
    main_data_pre, _ = cbyz.df_add_shift(df=main_data_pre, cols=ohlc, 
                                         shift=1, group_by=['STOCK_SYMBOL'], 
                                         suffix='_LAST', remove_na=False)

    main_data_pre = main_data_pre.rename(columns=ohlc_hist_dict)


    # Organize ......
    main_data = bt_results.merge(main_data_pre, how='left', 
                                 on=['WORK_DATE', 'STOCK_SYMBOL'])

    main_data = main_data[['BACKTEST_ID', 'STOCK_SYMBOL', 
                           'WORK_DATE', 'LAST_DATE'] \
                          + model_y + ohlc_hist + ohlc_last]


    # 把LAST全部補上最後一個交易日的資料
    # 因為回測的時間有可能是假日，所以這裡的LAST可能會有NA
    main_data = cbyz.df_shift_fill_na(df=main_data, 
                                      loop_times=predict_period+1, 
                                      cols=ohlc_last, 
                                      group_by=['STOCK_SYMBOL', 'BACKTEST_ID'])

    # Check na ......
    # 這裡有na是合理的，因為hist可能都是na
    chk = cbyz.df_chk_col_na(df=main_data)
    # main_data = main_data.dropna(subset=last_cols)
    
    if len(chk) > len(model_y):
        print('Err01. cal_profit - main_data has na in columns.')
        
        
    if len(main_data) == 0:
        print(('Error 1. main_data is empty. Check the market data has been'
               ' updated or not, it may be the reason cause last price na.'))
        

    # Generate Actions ......
    global precision
    
    bt_main, actions = \
        stk.gen_predict_action(df=main_data, precision=precision,
                                  date='WORK_DATE', 
                                  last_date='LAST_DATE', 
                                  y=model_y, y_last=model_y_last,
                                  y_thld=y_thld, time_thld=time_thld,
                                  prec_thld=prec_thld)
        
    # Evaluate Precision ......
    eval_metrics(export_file=False, upload=upload_metrics)            
    
    
    # Forecast Records ......
    print('Bug - get_forecast_records中的Action Score根本沒用到')
    records = stk.get_forecast_records(forecast_begin=None, 
                                       forecast_end=None, 
                                       execute_begin=execute_begin, 
                                       execute_end=None, 
                                       y=['CLOSE_CHANGE_RATIO'], 
                                       summary=True)
    
    if len(records)  > 0:
        records = records \
            .rename(columns={'FORECAST_PRECISION_MEDIAN':'RECORD_PRECISION_MEDIAN',
                             'FORECAST_PRECISION_STD':'RECORD_PRECISION_STD'})
            
    # Add name ......
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    stock_info = stock_info[['STOCK_SYMBOL', 'STOCK_NAME', 'INDUSTRY']]
    actions = actions.merge(stock_info, how='left', on='STOCK_SYMBOL')      


    # Hold 
    global hold
    hold = [str(i) for i in hold]
    actions['HOLD'] = np.where(actions['STOCK_SYMBOL'].isin(hold), 1, 0)
    
    actions.loc[actions.index, 'BUY_SIGNAL'] = np.nan 
    
    
    
    # Add OHLC ......
    action_cols = actions.columns
    if 'OPEN_CHANGE_RATIO' in action_cols and 'OPEN' not in action_cols:
        actions['OPEN'] = actions['OPEN_LAST'] \
                            * (1 + actions['OPEN_CHANGE_RATIO'])
    
    if 'HIGH_CHANGE_RATIO' in action_cols and 'HIGH' not in action_cols:
        actions['HIGH'] = actions['HIGH_LAST'] \
                            * (1 + actions['HIGH_CHANGE_RATIO'])
        
    if 'LOW_CHANGE_RATIO' in action_cols and 'LOW' not in action_cols:
        actions['LOW'] = actions['LOW_LAST'] \
                            * (1 + actions['LOW_CHANGE_RATIO'])

    if 'CLOSE_CHANGE_RATIO' in action_cols and 'CLOSE' not in action_cols:
        actions['CLOSE'] = actions['CLOSE_LAST'] \
                            * (1 + actions['CLOSE_CHANGE_RATIO'])



    # Rearrange Columns ......            
    # if 'CLOSE' in model_y:
    #     profit_cols = ['CLOSE_PROFIT_PREDICT', 
    #                    'CLOSE_PROFIT_RATIO_PREDICT']
    # else:
    #     profit_cols = []
        
    profit_cols = [close + '_PROFIT_PREDICT', 
                   close + '_PROFIT_RATIO_PREDICT']
        
        
    profit_cols = profit_cols \
        + ['RECORD_PRECISION_MEDIAN', 'RECORD_PRECISION_STD', 
           'DIFF_MEDIAN', 'DIFF_STD']
        
    
    cols_1 = ['BACKTEST_ID', 'STOCK_SYMBOL', 'STOCK_NAME', 'INDUSTRY',
              'BUY_SIGNAL', 'HOLD', 'WORK_DATE', 'LAST_DATE']

    cols_2 = ['PRECISION_'+ s for s in model_y]    
    
    new_cols = cols_1 + profit_cols + ohlc + ohlc_last \
                + model_y_hist + cols_2


    # Merge Data ......
    if len(records) > 0:
        actions = actions.merge(records, how='left', on=['STOCK_SYMBOL'])
    
        actions.loc[:, 'DIFF_MEDIAN'] = \
            actions[close + '_PROFIT_RATIO_PREDICT'] \
            - actions['RECORD_PRECISION_MEDIAN']
    
        actions.loc[:, 'DIFF_STD'] = \
            actions[close + '_PROFIT_RATIO_PREDICT'] \
            - actions['RECORD_PRECISION_STD']
    
    else:
        actions['RECORD_PRECISION_STD'] = np.nan
        actions['RECORD_PRECISION_MEDIAN'] = np.nan
        actions['DIFF_MEDIAN'] = np.nan
        actions['DIFF_STD'] = np.nan
        
    
    actions = actions[new_cols]

        
    # Buy Signal ......
    # 是不是可以移到gen_predict_action
    
    # Decrease On First Day ...
    cond1 = actions[(actions['WORK_DATE']==bt_last_begin) \
                   & (actions[close + '_PROFIT_RATIO_PREDICT']<0)]
    cond1 = cond1['STOCK_SYMBOL'].unique().tolist()
    
    # Estimated Profit ...
    cond2 = actions[(actions['WORK_DATE']>bt_last_begin) \
                   & (actions[close + '_PROFIT_RATIO_PREDICT']>=y_thld)]
    cond2 = cond2['STOCK_SYMBOL'].unique().tolist()    
    
    # Max Error ...
    cond3 = actions[actions['DIFF_MEDIAN']<prec_thld]
    cond3 = cond3['STOCK_SYMBOL'].unique().tolist()       
    
    
    buy_signal_symbols = cbyz.li_intersect(cond1, cond2, cond3)
    actions['BUY_SIGNAL'] = \
        np.where(actions['STOCK_SYMBOL'].isin(buy_signal_symbols), 1, 0)

    


# .................



def eval_metrics(export_file=False, upload=False):


    # MAPE ......
    global bt_main, bt_info, rmse
    global mape, mape_group, mape_extreme
    global stock_metrics_raw, stock_metrics
    
    model_y_hist = [y + '_HIST' for y in model_y]
    mape_main = bt_main.dropna(subset=model_y_hist, axis=0)
        
    
    if len(mape_main) == 0:
        return ''
    
    # ......
    mape = pd.DataFrame()
    mape_group = pd.DataFrame()
    mape_extreme = pd.DataFrame()
    stock_metrics_raw = pd.DataFrame()
    
    
    for i in range(len(model_y)):
        
        y = model_y[i]
        
        
        if 'CHANGE_RATIO' in y:
            mape_main.loc[:, 'MAPE'] = mape_main[y] - mape_main[y + '_HIST']
        else:
            mape_main.loc[:, 'MAPE'] = (mape_main[y] \
                                    - mape_main[y + '_HIST']) \
                                / mape_main[y + '_HIST']            
                            
                            
        mape_main.loc[:, 'OVERESTIMATE'] = \
            np.where(mape_main[y] > mape_main[y + '_HIST'], 1, 0)

        # Absolute ......
        mape_main_abs = mape_main.copy()
        mape_main_abs.loc[mape_main_abs.index, 'MAPE'] = abs(mape_main_abs['MAPE'])
                 
        
        # MAPE Overview
        new_mape = cbyz.df_summary(df=mape_main_abs, group_by=[], cols=['MAPE'])
        new_mape.loc[:, 'Y'] = y
        mape = mape.append(new_mape)
        
        
        # Group MAPE ......
        new_mape = cbyz.df_summary(df=mape_main_abs, group_by=['OVERESTIMATE'], 
                                 cols=['MAPE'])
        new_mape.loc[:, 'Y'] = y
        mape_group = mape_group.append(new_mape)
        
        
        # Extreme MAPE ......
        new_mape = mape_main_abs[mape_main_abs['MAPE'] > 0.1]
        new_mape = cbyz.df_summary(df=new_mape, 
                                group_by=['BACKTEST_ID', 'OVERESTIMATE'], 
                                cols=['MAPE'])
        
        if len(new_mape) > 0:
            new_mape.loc[:, 'Y'] = y
            mape_extreme = mape_extreme.append(new_mape)


        # Stock MAPE ......
        new_metrics = mape_main_abs[['BACKTEST_ID', 'STOCK_SYMBOL', 
                                 'WORK_DATE', 'MAPE', 'OVERESTIMATE',
                                 y, y + '_HIST']] \
            .rename(columns={y:'FORECAST_VALUE',
                             y + '_HIST':'HIST_VALUE'})

        new_metrics.loc[:, 'Y'] = y
        stock_metrics_raw = stock_metrics_raw.append(new_metrics)
    
    
    # Oraganize
    stock_metrics_raw = stock_metrics_raw \
                        .merge(bt_info, how='left', on='BACKTEST_ID') \
                        .merge(precision, how='left', on=['Y', 'BACKTEST_ID'])

    stock_metrics_raw['VERSION'] = sam.version
    stock_metrics_raw['MODEL_METRIC'] = 'RMSE'    
    stock_metrics_raw['FORECAST_METRIC'] = 'MAPE'
    stock_metrics_raw['STOCK_TYPE'] = 'TW'
    
    
    execute_id = cbyz.get_time_serial(with_time=True)
    execute_id = execute_id[2:13].replace('_', '')
    execute_id = int(execute_id)
    stock_metrics_raw.loc[:, 'EXECUTE_SERIAL'] = execute_id
    
    
    stock_metrics_raw = \
        cbyz.df_date_simplify(df=stock_metrics_raw, 
                              cols=['EXECUTE_SERIAL', 'WORK_DATE'])    
    
    stock_metrics_raw = stock_metrics_raw \
                        .rename(columns={'WORK_DATE':'FORECAST_DATE',
                                         'PRECISION':'MODEL_PRECISION',
                                         'MAPE':'FORECAST_PRECISION'}) \
                        .round({'MODEL_PRECISION':3,
                                'FORECAST_PRECISION':3})

                
    stock_metrics = stock_metrics_raw[['VERSION', 'STOCK_TYPE', 'STOCK_SYMBOL', 
                                       'EXECUTE_SERIAL', 'FORECAST_DATE', 
                                       'PREDICT_PERIOD', 'DATA_PERIOD', 'Y',
                                       'MODEL_METRIC', 'MODEL_PRECISION',
                                       'FORECAST_METRIC', 'FORECAST_PRECISION', 
                                       'OVERESTIMATE']]
    
        
    if export_file:
        time_serial = cbyz.get_time_serial(with_time=True)
        stock_metrics.to_csv(path_export + '/Metrics/stock_mape_'\
                             + time_serial + '.csv', 
                             index=False)
            
    if upload:
        ar.db_upload(data=stock_metrics, table_name='forecast_records')


# ..........


def master(_bt_last_begin, predict_period=14, interval=360, bt_times=5, 
           data_period=5, _stock_symbol=None, _stock_type='tw',
           signal=None, budget=None, split_budget=False):
    '''
    主工作區
    Update, 增加台灣上班上課行事曆，如果是end_date剛好是休假日，直接往前推一天。
    '''
    

    # v0.0 - First Version
    # v0.1
    # - 拿掉5MA之後的精準度有提升
    # v0.2
    # - Add buy_signal
    
    # v0.05
    # - Change local variable as host    
    
    
    # Backtest也可以用parameter做A/B    
    
    # data_period用2年的精準度會比3年高
    
    # Bug
    # print('backtest_predict - 這裡有bug，應該用global calendar')
    # add hold columns
    # 1.Excel中Last Priced地方不應該一直copy最後一筆資料


    # Worklist
    # 0. Call notification function in backtest manager, and add stop loss 
    #    in excel
    # 2. Price change of OHLC
    # 1. Add date index to indentify feature of time series
    #    > Add this to sam v7 and v8
    # 2. Add DIFF_MAPE_MEDIAN
    # 3. 除權息會拉抬N天的服價，把N用weight的方式考慮進去
    # 4. data_process中的lag，應該是要針對vars處理，還是針對y處理？
    # 5. 美股指數
    # 7. Add machine learning error function
    # 8. 之前的code是不是沒有把股本正確normalize / Add EPS
    # 9. Add DIFF_MAPE_MIN
    # 12. Set the result of long-term MA as a table
    # 13. Update, optimize capitial_level with kmeans
    # 15. excel中沒有 富鼎(8261)
    # 17. 長期的forecast by week
    # 18. When backtest, check if the symbol reached the target price in the 
    #     next N weeks.
    # 19. Analysis, 近60日投報率，用產業或主題作分群 / 財報
    # 20. Add Sell Signal
    # 21. 產業上中下游關係，SNA
    # 22. Update, load last version of model
    # 23. 在Excel中排除交易量低的
    # 24. 把股性分群
    # 25. Do actions by change
    # 26. Add Auto-Competing Model    

    
    # Parameters
    _bt_last_begin = 20211013
    predict_period = 5
    _interval = 4
    _bt_times = 2
    data_period = int(365 * 3.5)
    # data_period = int(365 * 0.86) # Shareholding    
    # data_period = 365 * 2
    # data_period = 365 * 5
    # data_period = 365 * 7
    _stock_symbol = [2520, 2605, 6116, 6191, 3481, 2409, 2603]
    _stock_symbol = []
    _stock_type = 'tw'
    # _ma_values = [5,10,20]
    # _ma_values = [5,10,20,40]
    _ma_values = [5,10,20,60]
    _volume_thld = 500


    global interval, bt_times, volume_thld
    interval = _interval
    bt_times = _bt_times
    volume_thld = _volume_thld
    ma_values = _ma_values
        

    # Rename predict period as forecast preiod
    # ......    
    global stock_symbol, stock_type, ma_values
    global bt_last_begin, bt_last_end
    
    stock_type = _stock_type    
    stock_symbol = _stock_symbol
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, to_type='str')


    # Set Date ......
    global calendar, bt_last_begin, bt_last_end
    set_calendar(_bt_last_begin, predict_period)

    
    # Predict ------
    global bt_results, precision, features
    backtest_predict(bt_last_begin=bt_last_begin, 
                     predict_period=predict_period, 
                     interval=interval,
                     bt_times=bt_times,
                     data_period=data_period,
                     load_model=True, cv=2, fast=False)

    
    # Profit ------    
    # Update, 需把Y為close和Y為price_change的情況分開處理
    # Update, stock_info每日比對檔案自動匯
    
    # y_thld=0.05
    # time_thld=predict_period
    # prec_thld=0.03
    # export_file=True
    # load_file=True
    # path=path_temp
    # file_name=None        
    

    global bt_main, actions
    
    # Optimize, 這裡的precision_thld實際上是mape, 改成precision
    # 算回測precision的時候，可以低估，但不可以高估
    global mape, mape_group, mape_extreme
    global stock_metrics_raw, stock_metrics    
    
    global hold
    hold = [8105, 2610, 3051, 4934]
    
    
    print('Bug - get_forecast_records中的Action Score根本沒用到，但可以用signal替代')
    print('評估是否可以把buy_signal移到gen_predict_action')
    cal_profit(y_thld=0.05, time_thld=predict_period, prec_thld=0.05,
               execute_begin=2108110000, 
               export_file=True, load_file=True, path=path_temp,
               file_name=None, upload_metrics=True)
    
    
    # Export ......
    time_serial = cbyz.get_time_serial(with_time=True)
    excel_name = path_export + '/actions_' + time_serial + '.xlsx'
    writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')


    workbook = writer.book
    workbook.add_worksheet('stock') 
    sht = workbook.get_worksheet_by_name("stock")


    cbyz.excel_add_df(actions, sht, 
                      startrow=0, startcol=0, header=True)

    # Add Format
    digi_format = workbook.add_format({'num_format':'0.0'})
    percent_format = workbook.add_format({'num_format':'0.0%'})

    cbyz.excel_add_format(sht=sht, cell_format=digi_format, 
                          startrow=1, endrow=9999,
                          startcol=8, endcol=8)    
    
    cbyz.excel_add_format(sht=sht, cell_format=percent_format, 
                          startrow=1, endrow=9999,
                          startcol=9, endcol=13)
    
    cbyz.excel_add_format(sht=sht, cell_format=digi_format, 
                          startrow=1, endrow=9999,
                          startcol=14, endcol=17)  
    
    cbyz.excel_add_format(sht=sht, cell_format=percent_format, 
                          startrow=1, endrow=9999,
                          startcol=18, endcol=21)
    
    writer.save()

    


# %% View ------
def view_today_limit_up():
    
    data = stk.get_data(data_begin=20210706, data_end=20210707, price_change=True)
    
    data = data[data['PRICE_CHANGE_RATIO']>0.08]
    # chk = data[data['VOLUME']>1000000]
    # 
    
    
    stock_info = stk.tw_get_stock_info(path=path_temp)
    stock_main = data.merge(stock_info, how='left', on='STOCK_SYMBOL')
    stock_main
    
    



# %% Check ------

def check():
    '''
    資料驗證
    '''    
    
    # Err01
    chk = main_data[main_data['HIGH_HIST'].isna()]   
    chk
    
    return ''


def check_price():
    
    
    chk = bt_results[bt_results['STOCK_SYMBOL']=='5521']
    chk

    chk = bt_results[bt_results['STOCK_SYMBOL']=='2702']
    chk


if __name__ == '__main__':
    results = master(begin_date=20180401)



def delete_records():
    
    # 很怪，只能刪除少量的幾筆資料
    serial = 2107082010

    sql = (" delete from forecast_records "
           " where execute_serial < " + str(serial))
    
    ar.db_execute(sql, fetch=False)



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
