#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

History

20210703 - Replaced MA with WMA in SAM

"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import random

host = 0
market = 'tw'

# Path .....
if host == 0:
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/3_Backtest'
    path_sam = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis'

elif host == 2:
    path = '/home/jupyter/3_Backtest'
    path_sam = '/home/jupyter/2_Stock_Analysis'    

# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20211229',
                 r'/home/jupyter/Arsenal/20211229',
                 path + '/Function',
                 path_sam]



for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import codebase_ml as cbml
import arsenal as ar
import arsenal_stock as stk
# import stock_analysis_manager_v2_01 as sam
# import stock_analysis_manager_v2_02 as sam
# import stock_analysis_manager_v2_03 as sam
import stock_analysis_manager_v2_04_dev as sam



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     



# %% Inner Function ------


def set_calendar():
    
    # global symbols
    global calendar, calendar_lite
    global _bt_last_begin, _bt_last_end, _predict_period
    global predict_date
    # global actions_main
    

    # Set Calendar    
    calendar_end = cbyz.date_cal(_bt_last_begin, _predict_period + 20, 'd')
    calendar = stk.get_market_calendar(begin_date=20140101, 
                                       end_date=calendar_end,
                                       market=market)
    
    calendar = calendar[calendar['TRADE_DATE']>0]
    
    
    # df_add_shift will cause NA, it's essential to drop to convert to int
    calendar, _ = \
        cbyz.df_add_shift(df=calendar, cols='WORK_DATE',
                          shift=_predict_period - 1, 
                          group_by=['TRADE_DATE'], 
                          suffix='_LAST', remove_na=True
                          )

    calendar = calendar.rename(columns={'WORK_DATE_LAST':'LAST_DATE'})
    calendar = cbyz.df_conv_col_type(df=calendar, cols='LAST_DATE', to='int')
    calendar_lite = calendar[['WORK_DATE', 'LAST_DATE']]

    
    # Get the last date
    _bt_last_end = calendar[calendar['LAST_DATE']==_bt_last_begin] \
                    .reset_index(drop=True)
    _bt_last_end = int(_bt_last_end.loc[0, 'WORK_DATE'])  


    # Get predict date
    predict_date = calendar[(calendar['WORK_DATE']>=_bt_last_begin) \
                        & (calendar['WORK_DATE']<=_bt_last_end)]

    predict_date = predict_date['WORK_DATE'].tolist()    




def set_frame():

    global symbols
    global calendar, calendar_lite, frame
    global _bt_last_begin, _bt_last_end, _predict_period
    global predict_date
    global actions_main
    # global ohlc, ohlc_ratio, ohlc_last
    
    global actions_main, hist_main
    
    
    global ohlc, ohlc_ratio, ohlc_last
    global var_y, var_y_last, var_y_hist
    
    # LAST用來參考，HIST用來算Precision
    # - 因為y可能是price，也可能是ratio，所以在filter dataframe時，盡量用ohlc相關的
    #   變數，不要用var_y
    ohlc = stk.get_ohlc(orig=True, ratio=False)
    ohlc_ratio = stk.get_ohlc(orig=False, ratio=True)
    ohlc_last = [i + '_LAST' for i in ohlc]
    # ohlc_hist = [i + '_HIST' for i in ohlc]

    var_y_last = [i + '_LAST' for i in var_y]
    var_y_hist = [i + '_HIST' for i in var_y]


    # Hist Data ......
    # - predict_period * 2 to ensure to get the complete data
    # - No matter your y is the price or the change ratio, it is essential to
    #   keep both information in the actions_main
    
      
    bt_first_begin = \
        cbyz.date_cal(_bt_last_begin, 
                      -_interval * _bt_times - _predict_period * 2, 
                      'd')    
    
    
    hist_data_raw = stk.get_data(data_begin=bt_first_begin, 
                                 data_end=_bt_last_end, 
                                 market=market, 
                                 symbol=symbols, 
                                 price_change=True,
                                 restore=False)
    
    if 'CLOSE' in var_y:
        hist_data_raw = hist_data_raw[['WORK_DATE', 'SYMBOL'] + ohlc]
    else:
        rename_dict = cbyz.li_to_dict(ohlc, ohlc_last)
        
        hist_data_raw = hist_data_raw[['WORK_DATE', 'SYMBOL'] \
                                      + ohlc + ohlc_ratio] \
                        .rename(columns=rename_dict)
            
    
    
    # Check Symbols
    if len(symbols) > 0:
        symbol_df = pd.DataFrame({'SYMBOL':symbols})
    else:
        temp_symbol = hist_data_raw['SYMBOL'].unique().tolist()
        symbol_df = pd.DataFrame({'SYMBOL':temp_symbol})        
    
    
    # Set Frame
    frame = cbyz.df_cross_join(symbol_df, calendar_lite)
    frame = frame \
        .sort_values(by=['SYMBOL', 'WORK_DATE']) \
        .reset_index(drop=True)


    # 
    rename_dict = cbyz.li_to_dict(var_y, var_y_last)        
    
    actions_main = hist_data_raw \
            .rename(columns=rename_dict) \
            .rename(columns={'WORK_DATE':'LAST_DATE'})
            
    actions_main = frame \
        .merge(actions_main, how='left', on=['LAST_DATE', 'SYMBOL']) \
        .merge(bt_results, how='left', on=['WORK_DATE', 'SYMBOL'])
    
    actions_main = actions_main[(actions_main['WORK_DATE']>=_bt_last_begin) \
                                & (actions_main['WORK_DATE']<=_bt_last_end)]
        
        
    # Hist Main    
    rename_dict = cbyz.li_to_dict(var_y, var_y_hist)        
    hist_main = hist_data_raw \
        .rename(columns=rename_dict) 

    hist_main = frame \
        .merge(hist_main, how='left', on=['WORK_DATE', 'SYMBOL']) \
        .drop('LAST_DATE', axis=1)

    hist_main = hist_main[(hist_main['WORK_DATE']>=_bt_last_begin) \
                                & (hist_main['WORK_DATE']<=_bt_last_end)]
        

# ..........
    

def backtest_predict(bt_last_begin, predict_period, interval, 
                     data_period, dev=False):
    
    global calendar, calendar_lite
    global symbols, _market, bt_info, _bt_times, _ma_values
    
    
    # Prepare For Backtest Records ......
    print('backtest_predict - 這裡有bug，應該用global calendar')
    bt_info_raw = cbyz.date_get_seq(begin_date=bt_last_begin,
                                    seq_length=_bt_times,
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
    global bt_results_raw, bt_results
    global precision, features, var_y, _volume_thld
    global pred_scores, pred_features
    
    bt_results_raw = pd.DataFrame()
    precision = pd.DataFrame()
    
    # Predict ......
    for i in range(0, len(bt_seq)):
        
        sam_result, sam_scores, \
            sam_params, sam_features = \
                sam.master(param_holder=param_holder,
                           predict_begin=bt_seq[i],
                           bt_index=i,
                           threshold=30000)

        sam_result['BACKTEST_ID'] = i
        sam_scores['BACKTEST_ID'] = i
        
        if i == 0:
            pred_scores = sam_scores.copy()

            # Optimize, sort in SAM or in UT            
            pred_features = sam_features.copy()
            pred_features = pred_features \
                .sort_values(by=['IMPORTANCE'], ascending=False) \
                .reset_index(drop=True)
            
        bt_results_raw = bt_results_raw.append(sam_result)
        precision = precision.append(sam_scores)


    # Organize ......
    global var_y
    
    # ValueError: You are trying to merge on object and int64 columns. 
    # If you wish to proceed you should use pd.concat
    bt_results = cbyz.df_conv_col_type(df=bt_results_raw, 
                                       cols='WORK_DATE', 
                                       to='int')    
    
    bt_results = bt_results.reset_index(drop=True)
    
    var_y = cbyz.df_get_cols_except(
        df=bt_results, 
        except_cols=['SYMBOL', 'WORK_DATE', 'BACKTEST_ID']
        )


# ............



def cal_profit(y_thld=2, time_thld=10, prec_thld=0.15, execute_begin=None,
               export_file=True, load_file=False, path=None, file_name=None):
    '''
    應用場景有可能全部作回測，而不做任何預測，因為data_end直接設為bt_last_begin
    '''
    
    
    global ohlc, ohlc_last
    global _predict_period
    global _interval, _bt_times 
    global bt_results, rmse, bt_main, actions, var_y
    global symbols, _market
    global _bt_last_begin, _bt_last_end    
    global calendar


    global ohlc, ohlc_ratio, ohlc_last
    global var_y, var_y_last, var_y_hist



        
    # Merge hist data ......
    global frame, actions_main
    
    if 'CLOSE_CHANGE_RATIO' in var_y:
        main_data = actions_main[['BACKTEST_ID', 'SYMBOL', 
                                  'WORK_DATE', 'LAST_DATE'] \
                                 + var_y + var_y_last + ohlc_last]
    elif 'CLOSE' in var_y:
        main_data = actions_main[['BACKTEST_ID', 'SYMBOL', 
                                  'WORK_DATE', 'LAST_DATE'] \
                                 + var_y + var_y_last]
        

    # Check NA ......
    # 1. 這裡有na是合理的，因為hist可能都是na
    # 2. 在SAM中被排除的Symbol可能會出現在這，導致每一欄的NA_COUNT不一樣
    cbyz.df_chk_col_na(df=main_data)

    
    # if len(chk) > len(var_y):
    #     print('Err01. cal_profit - main_data has na in columns.')
        
        
    if len(main_data) == 0:
        print(('Error 1. main_data is empty. Check the market data has been'
               ' updated or not, it may be the reason cause last price na.'))
        

    # Generate Actions ......
    global precision
    
    bt_main, actions = \
        stk.gen_predict_action(df=main_data,
                               precision=precision,
                               date='WORK_DATE', 
                               last_date='LAST_DATE', 
                               y=var_y, 
                               y_last=var_y_last,
                               y_thld=y_thld, 
                               time_thld=time_thld,
                               prec_thld=prec_thld)
        
    msg = 'The max value of BACKTEST_ID should be 0.'
    assert actions['BACKTEST_ID'].max() == 0, msg
        

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
            
    # Add Low Volume Symbols    
    if len(sam.low_volume_symbols) > 0:
        low_volume_df = pd.DataFrame({'SYMBOL':sam.low_volume_symbols})
        actions = actions.merge(low_volume_df, how='outer', on='SYMBOL')
        
        
    # Add name ......
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    stock_info = stock_info[['SYMBOL', 'STOCK_NAME', 'INDUSTRY']]
    actions = actions.merge(stock_info, how='left', on='SYMBOL')
    
    
    # Hold Symbols
    global _hold
    actions['HOLD'] = np.where(actions['SYMBOL'].isin(_hold), 1, 0)


    actions['BUY_SIGNAL'] = np.nan 
    
    
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


    # Day Trading Signal ......
    actions['DAY_TRADING_SIGNAL'] = actions['HIGH'] - actions['LOW']
        
    
    
    if 'CLOSE_CHANGE' not in action_cols:
        actions['CLOSE_CHANGE'] = \
            actions['CLOSE'] - actions['CLOSE_LAST']
    
    
    # Rearrange Columns ......       
    profit_cols = ['CLOSE_CHANGE', 'CLOSE_CHANGE_RATIO', 
           'RECORD_PRECISION_MEDIAN', 'RECORD_PRECISION_STD', 
           'DIFF_MEDIAN', 'DIFF_STD']
        
    
    cols_1 = ['SYMBOL', 'STOCK_NAME', 'INDUSTRY',
              'BUY_SIGNAL', 'DAY_TRADING_SIGNAL', 'HOLD', 
              'WORK_DATE', 'LAST_DATE']

    cols_2 = ['PRECISION_'+ s for s in var_y]    
    
    new_cols = cols_1 + profit_cols + ohlc + ohlc_last \
                + cols_2


    # Merge Data ......
    if len(records) > 0:
        actions = actions.merge(records, how='left', on=['SYMBOL'])
    
        actions.loc[:, 'DIFF_MEDIAN'] = \
            actions['CLOSE_CHANGE_RATIO'] - actions['RECORD_PRECISION_MEDIAN']
    
        actions.loc[:, 'DIFF_STD'] = \
            actions['CLOSE_CHANGE_RATIO'] - actions['RECORD_PRECISION_STD']
    
    else:
        actions['RECORD_PRECISION_STD'] = np.nan
        actions['RECORD_PRECISION_MEDIAN'] = np.nan
        actions['DIFF_MEDIAN'] = np.nan
        actions['DIFF_STD'] = np.nan
        
    
    actions = actions[new_cols]

        
    # Buy Signal ......
    # 是不是可以移到gen_predict_action
    
    # Decrease On First Day ...
    cond1 = actions[(actions['WORK_DATE']==_bt_last_begin) \
                   & (actions['CLOSE_CHANGE_RATIO']<0)]
    cond1 = cond1['SYMBOL'].unique().tolist()
    
    # Estimated Profit ...
    cond2 = actions[(actions['WORK_DATE']>_bt_last_begin) \
                   & (actions['CLOSE_CHANGE_RATIO']>=y_thld)]
    cond2 = cond2['SYMBOL'].unique().tolist()    
    
    # Max Error ...
    cond3 = actions[actions['DIFF_MEDIAN']<prec_thld]
    cond3 = cond3['SYMBOL'].unique().tolist()       
    
    buy_signal_symbols = cbyz.li_intersect(cond1, cond2, cond3)
    
    
    print('Close裡面有NA，可能是已經下檔的Symbol？')
    # Cannot convert non-finite values (NA or inf) to integer
    
    global cal_profit_debug
    cal_profit_debug = actions[actions['CLOSE'].isna()]
    
    actions = cbyz.df_conv_na(df=actions,
                              cols='CLOSE' ,
                              value=-1000)    
    
    # Add Level
    actions['PERCENTAGE'] = actions['CLOSE_CHANGE_RATIO'] * 100

    # There may be some inf values
    actions = cbyz.df_conv_na(df=actions, cols='PERCENTAGE', value=-1000)
    actions = cbyz.df_handle_inf(df=actions, cols='PERCENTAGE', value=-1000, 
                                 drop=False, axis=0)
    
    actions['PERCENTAGE'] = actions['PERCENTAGE'].astype('int')
      
    actions['BUY_SIGNAL'] = \
        np.where(actions['SYMBOL'].isin(buy_signal_symbols), 
                 99, actions['PERCENTAGE'])



    
# .................



def eval_metrics(export_file=False, threshold=800):

    
    # MAPE ......
    global bt_main, bt_info, rmse
    global mape, mape_group, mape_extreme
    global stock_metrics_raw, stock_metrics
    global var_y, var_y_last, var_y_hist
    
    
    
    loc_main = bt_results.merge(hist_main, on=['SYMBOL', 'WORK_DATE'])

    # 不做回測時，mape_main的length一定會等於0
    if len(loc_main) == 0:
        return
    
    # ......
    loc_precision = pd.DataFrame()
    mape_group = pd.DataFrame()
    mape_extreme = pd.DataFrame()
    stock_metrics_raw = pd.DataFrame()
    
    for i in range(len(var_y)):
        
        y = var_y[i]
        y_hist = var_y_hist[i]
        
        
        if 'CHANGE_RATIO' in y:
            loc_main['MAPE'] = loc_main[y] - loc_main[y_hist]
        else:
            loc_main['MAPE'] = \
                (loc_main[y] - loc_main[y_hist]) / loc_main[y_hist]            
                            
                            
        loc_main['OVERESTIMATE'] = \
            np.where(loc_main[y] > loc_main[y_hist], 1, 0)

        # Absolute ......
        mape_main_abs = mape_main.copy()
        mape_main_abs.loc[mape_main_abs.index, 'MAPE'] = abs(mape_main_abs['MAPE'])
                 
        
        # MAPE Overview
        new_mape, _ = cbyz.df_summary(df=mape_main_abs, group_by=[], cols=['MAPE'])
        new_mape.loc[:, 'Y'] = y
        mape = mape.append(new_mape)
        
        
        # Group MAPE ......
        new_mape, _ = cbyz.df_summary(df=mape_main_abs, group_by=['OVERESTIMATE'], 
                                 cols=['MAPE'])
        new_mape.loc[:, 'Y'] = y
        mape_group = mape_group.append(new_mape)
        
        
        # Extreme MAPE ......
        new_mape = mape_main_abs[mape_main_abs['MAPE'] > 0.1]
        new_mape, _ = cbyz.df_summary(df=new_mape, 
                                group_by=['BACKTEST_ID', 'OVERESTIMATE'], 
                                cols=['MAPE'])
        
        if len(new_mape) > 0:
            new_mape.loc[:, 'Y'] = y
            mape_extreme = mape_extreme.append(new_mape)


        # Stock MAPE ......
        new_metrics = mape_main_abs[['BACKTEST_ID', 'SYMBOL', 
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


    print('Update - 還沒把STOCK_TYPE改成Market')
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
                                         'TEST_PRECISION':'MODEL_PRECISION',
                                         'MAPE':'FORECAST_PRECISION'}) \
                        .round({'MODEL_PRECISION':3,
                                'FORECAST_PRECISION':3})

                
    stock_metrics = stock_metrics_raw[['VERSION', 'STOCK_TYPE', 'SYMBOL', 
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
            
    if len(bt_main) > threshold:
        ar.db_upload(data=stock_metrics, table_name='forecast_records')


# %% View And Log ------


def view_yesterday():

    # LAST_DATE might be float
    global actions
    last_date = actions['LAST_DATE'].min()
    last_date = int(last_date)
        

    # Stock Info
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    stock_info = stock_info[['SYMBOL', 'INDUSTRY']]
    
    
    # Market Data
    loc_data = stk.get_data(data_begin=last_date, 
                            data_end=last_date, 
                            market='tw', 
                            symbol=[], 
                            price_change=True,
                            restore=False) 
    
    loc_data = loc_data[['SYMBOL', 'WORK_DATE', 'CLOSE_CHANGE_RATIO']]
    # loc_data = loc_data[abs(loc_data['CLOSE_CHANGE_RATIO'])]
    
    
    # Combine
    loc_main = loc_data.merge(stock_info, how='inner', on='SYMBOL')
    summary, _ = cbyz.df_summary(df=loc_main, 
                                 group_by=['INDUSTRY'], 
                                 cols=['CLOSE_CHANGE_RATIO'])

    summary = summary \
            .sort_values(by='CLOSE_CHANGE_RATIO_MEAN', ascending=False) \
            .reset_index(drop=True)
            
    # Write
    stk.write_sheet(data=summary, sheet='YD_Industry')            
        


# %% Master ------


def master(bt_last_begin, predict_period=14, long=False, interval=360, 
           bt_times=2, data_period=5, ma_values=[5,10,20,60], volume_thld=400, 
           cv=2, compete_mode=1, market='tw', hold=[], dev=False):
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
    # v0.06
    # - Update for cbyz and cbml
    # - Include low volume symbols in the excel
    # v0.062
    # - Add Close Level To BUY_SIGNAL column  
    # v0.064
    # Export to specific sheet
    # 這個版本有Bug，不改了，直接跳v0_07    
    # v0.07
    # - Update for ultra_tuner
    # - Add write_sheet
    # - 列出前一天大漲的產業
    # - 增加一個欄位，標示第二天為負，為了和DAY_TRADING配合快速篩選
    # v0.071
    # - hold symbols沒有順利標示 - Done
    # - cal_profit中的last不要用還原股價，不然看的時候會很麻煩 - Done
    # - View yesterday，還沒處理日期的問題 - Done
    # v0.072
    # - Remove fast and load_model parameter - Done
    # - 把high和short的model分開，避免覆寫模型 - Done
    # - 當load_model為False，且bt_times > 1時，只有第一次會retrain，第二次會load - Done
    # v0.073
    # - The features sheet of the Actions will be overwrited by both 
    #    long and short. - Done


    # v0.074
    # - Fix last_price issues - Done, wait for checking 
    # - Optimize cal_profit - Done
    

    # Bug
    # 5. print predict date in sam to fix missing date issues    
    # 6. 如果local沒有forecast_records時，cal_profit中的get_forecast_records會出錯：
    #    AttributeError: 'list' object has no attribute 'rename'
    # 7. precision列出來的，好像都不是score最佳的log
    
    # Optimization
    # 4. Calculate IRR, remove outliers
    # 5. Google Sheet Add Manual Tick
    # 6. Think how to optimize stop loss
    # 7. Backtest也可以用parameter做A/B        



    # Worklist
    # 0. Remove Open
    # 0. Call notification function in backtest manager, and add stop loss 
    #    in excel
    # 1. Dev為True時就不匯出模型，避免檔案亂掉
    # 1. Add date index to indentify feature of time series
    #    > Add this to sam v7 and v8
    # 2. Add DIFF_MAPE_MEDIAN
    # 3. 除權息會拉抬N天的服價，把N用weight的方式考慮進去
    # 4. data_process中的lag，應該是要針對vars處理，還是針對y處理？
    # 5. 美股指數 https://iexcloud.io/pricing/#price-table-section
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
    # 24. 把股性分群
    # 25. Do actions by change
    # 26. Add Auto-Competing Model    
    # 27. Signal A and B，A是反彈的，B是low和close差距N%
    # 交易資料先在dcm中算好ma
    # global parama 應該是在一開始就訂好，而不是最後才收集，參考rtml
    # 把market改成market
    # 28. 當bt_times為2，且load_file=false時，重新訓練一次模型就好


    
    # Worklist
    # 1.Add price increse but model didn't catch
    global _interval, _bt_times, _volume_thld, _ma_values, _hold
    global symbols, _market
    global _bt_last_begin, _bt_last_end    

    _hold = [str(i) for i in hold]


    # Parameters
    
    # # Not Collected Parameters ......
    # bt_times = 1
    # interval = 4
    # market = 'tw'
    # dev = True    
    
    # # Collected Parameters ......
    # bt_last_begin = 20210913
    # predict_period = 6
    # data_period = int(365 * 3.5)
    # ma_values = [6,10,20,60]
    # volume_thld = 500
    # hold = [8105, 2610, 3051, 1904, 2611]
    # long = False
    # compete_mode = 0

    # Wait for update
    # date_manager = cbyz.Date_Manager(predict_begin=predict_begin, 
    #                                  predict_period=_predict_period,
    #                                  data_period=_data_period,
    #                                  data_period_unit='d',
    #                                  data_period_agg_unit='d',
    #                                  predict_period_unit='d',
    #                                  predict_by_time=True,
    #                                  merge_period=[], shift=data_shift, 
    #                                  week_begin=0)
    
    # date_df = date_manager.table
    # shift_begin = date_df.loc[0, 'SHIFT_BEGIN']
    # shift_end = date_df.loc[0, 'SHIFT_END']
    # data_begin = date_df.loc[0, 'DATA_BEGIN']
    # data_end = date_df.loc[0, 'DATA_END']
    # predict_begin = date_df.loc[0, 'PREDICT_BEGIN']
    # predict_end = date_df.loc[0, 'PREDICT_END']  
    # calendar = date_manager.calendar_lite    


    if dev:
        symbols = [2520, 2605, 6116, 6191, 3481, 
                   2409, 2520, 2603,
                   2409, 2603, 2611, 3051, 3562]
    else:
        symbols = []    

    # Arguments
    # 1. 目前的parameter都只有一個，但如果有多個parameter需要做A/B test時，應該要在
    #    在btm中extract，再把single parameter的Param_Holder傳入sam，因為
    #    Param_Holder的參數會影響到model_data，沒辦法同一份model_data持續使用，因此，
    #    把完整的Param_Holder傳入sam再extract沒有任何效益
    args = {'bt_last_begin':[bt_last_begin],
            'predict_period': [predict_period], 
            'data_period':[data_period],
            'ma_values':[ma_values],
            'volume_thld':[volume_thld],
            'industry':[True],
            'trade_value':[True],
            'market':['tw'],
            'compete_mode':[compete_mode],
            'train_mode':[2],            
            'cv':[cv],
            'kbest':['all'],
            'dev':[dev],
            'symbols':[symbols],
            'long':[long],
            'debug':[False]
            }
    
    global param_holder
    param_holder = ar.Param_Holder(**args)
    
    
    # Select Parameters
    params = param_holder.params
    keys = list(params.keys())
    values = list(params.values())

    param_df = pd.DataFrame()
    for i in range(0, len(params)):
        
        values_li = values[i]
        values_li = cbyz.conv_to_list(values_li)
        new_df = pd.DataFrame({keys[i]:values_li})
        
        if i == 0:
            param_df = param_df.append(new_df)
        else:
            param_df = cbyz.df_cross_join(param_df, new_df)


    _interval = interval
    _bt_times = bt_times
    _volume_thld = volume_thld
    _ma_values = ma_values
        

    # Rename predict period as forecast preiod
    # ......    
    
    _market = market    
    symbols = symbols
    symbols = cbyz.li_conv_ele_type(symbols, to_type='str')

    # Set Date ......
    global _bt_last_begin, _predict_period
    global calendar, _bt_last_end
    _predict_period = predict_period
    _bt_last_begin = bt_last_begin
    
    set_calendar()


    # Predict ------
    global bt_results, precision, features
    backtest_predict(bt_last_begin=bt_last_begin, 
                     predict_period=_predict_period, 
                     interval=interval,
                     data_period=data_period,
                     dev=dev)

    # Set Date ......
    set_frame()
    
    
    # Debug for prices columns issues
    stk.write_sheet(data=bt_results, sheet='Debug')
    
    
    # Profit ------    
    # y_thld=0.02
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
    
    
    execute_begin = cbyz.date_get_today()
    execute_begin = cbyz.date_cal(execute_begin, -14, 'd')
    # execute_begin = cbyz.date_cal(execute_begin, -40, 'd')
    execute_begin = int(str(execute_begin)[2:] + '0000')
    
    
    # Evaluate Precision ......
    
    # eval_metricsv裡面的bug還沒修好
    # eval_metrics(export_file=False) 
    
    print('Bug - get_forecast_records中的Action Score根本沒用到，但可以用signal替代')
    cal_profit(y_thld=0.02, time_thld=_predict_period, prec_thld=0.05,
               execute_begin=execute_begin, 
               export_file=True, load_file=True, path=path_temp,
               file_name=None)
    
    
    # Export ......
    time_serial = cbyz.get_time_serial(with_time=True)
    excel_name = path_export + '/actions_' + time_serial + '.xlsx'
    writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')


    workbook = writer.book
    workbook.add_worksheet('stock') 
    sht = workbook.get_worksheet_by_name("stock")
    cbyz.excel_add_df(actions, sht, startrow=0, startcol=0, header=True)


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


    
    if len(actions) > 800:
        # Write Google Sheets
        stk.write_sheet(data=actions, sheet='TW', long=long,
                        predict_begin=_bt_last_begin)
    
        # View And Log .....
        view_yesterday()

        global pred_features
        stk.write_sheet(data=pred_features, sheet='Features')
    
    
    gc.collect()



# %% Check ------


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
                            market='tw', 
                            symbol=[], 
                            price_change=True,
                            adj=True, 
                            price_limit=False, 
                            trade_value=False)

    data = data_raw[(data_raw['WORK_DATE']==20210625) \
                & (data_raw['STOCK_SYMBOL'].isin(symbols))] \
            .sort_values(by='PRICE_CHANGE_RATIO', ascending=False) \
            .reset_index(drop=True)

    
    main_data = data.merge(file, how='left', on='STOCK_SYMBOL')
    

    

# %% Dev -----


def dev():

    ledger = stk.get_ledger()



# %% Execute ------
if __name__ == '__main__':
    
    
    hold = [1909, 2009, 2485, 2605, 3041, 2633]
    
    master(bt_last_begin=20211201, predict_period=5, 
           long=False, interval=4, bt_times=1, 
           data_period=int(365 * 1), 
           ma_values=[5,10,20], volume_thld=400,
           compete_mode=2, cv=list(range(2, 7)),
           market='tw', hold=hold,
           dev=True)
    
    
    # master(bt_last_begin=20211230, predict_period=4, 
    #        long=False, interval=7, bt_times=1, 
    #        data_period=int(365 * 5), 
    #        ma_values=[10,20,60], volume_thld=300,
    #        compete_mode=1, cv=list(range(3, 8)),
    #        market='tw', hold=hold,
    #        dev=False)


    # master(bt_last_begin=20211228, predict_period=10, 
    #        long=True, interval=7, bt_times=1, 
    #        data_period=int(365 * 5), 
    #        ma_values=[10,20,60], volume_thld=300,
    #        compete_mode=1, cv=list(range(3, 7)),
    #        market='tw', hold=hold,
    #        dev=False)



