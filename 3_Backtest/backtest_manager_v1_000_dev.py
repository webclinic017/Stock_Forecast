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

host = 3
host = 2
host = 4
# host = 0
market = 'tw'


# Path .....
if host == 0:
    # Home
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/3_Backtest'
    path_sam = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis'

elif host == 2:
    # PythonAnyWhere
    path = '/home/jupyter/Production/3_Backtest'
    path_sam = '/home/jupyter/Production/2_Stock_Analysis'    
    
elif host == 3:
    # GCP
    path = '/home/jupyter/Develop/3_Backtest'
    path_sam = '/home/jupyter/Develop/2_Stock_Analysis'      

elif host == 4:    
    # RT
    path = r'D:\Data_Mining\GitHub共用\Stock_Forecast\3_Backtest'
    path_sam = r'D:\Data_Mining\GitHub共用\Stock_Forecast\2_Stock_Analysis'    


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'D:\Data_Mining\GitHub共用\Arsenal',
                 r'D:\Data_Mining\Projects\Codebase_YZ',
                 r'/Users/aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20220213',
                 r'/home/jupyter/Arsenal/20220213',
                 path + '/Function',
                 path_sam]


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk
import codebase_ml as cbml
# import stock_analysis_manager_v2_10_dev as sam
# import stock_analysis_manager_v2_11_dev as sam
# import stock_analysis_manager_v2_112_dev as sam
import stock_analysis_manager_v2_400_dev as sam



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     



# %% Inner Function ------


def set_calendar_old():
    
    # global symbol
    global calendar, calendar_lite
    global _bt_last_begin, _bt_last_end, _predict_period
    global predict_date
    # global actions_main
    

    # Set Calendar    
    calendar_end = cbyz.date_cal(_bt_last_begin, _predict_period + 20, 'd')
    calendar_raw = stk.get_market_calendar(end_date=calendar_end,
                                           market=market)
    
    calendar = calendar_raw[calendar_raw['TRADE_DATE']>0]
    calendar = calendar[['WORK_DATE', 'YEAR', 'WEEK_NUM']]
    
    if time_unit == 'w':
        calendar = calendar \
                    .drop('WORK_DATE', axis=1) \
                    .drop_duplicates()
    
    calendar = calendar.reset_index(drop=True)
    
    # df_add_shift will cause NA, it's essential to drop to convert to int
    calendar, _ = \
        cbyz.df_add_shift(df=calendar, 
                          cols=time_key,
                          shift=1, 
                          group_by=[],
                          sort_by=time_key,
                          suffix='_LAST', remove_na=True
                          )

    calendar = calendar.rename(columns={'WORK_DATE_LAST':'LAST_DATE'})
    calendar = cbyz.df_conv_col_type(df=calendar, cols='LAST_DATE', to='int')
    calendar_lite = calendar[['WORK_DATE', 'LAST_DATE']]

    
    # Get the last date
    # - Error may be raised here if database not updated
    index = calendar[calendar['WORK_DATE']==_bt_last_begin]
    assert len(index) > 0, 'Check database updated or not'
    
    index = index.index[0]
    index = index + _predict_period - 1
    _bt_last_end = calendar.loc[index, 'WORK_DATE']
    

    # Get predict date
    predict_date = calendar[(calendar['WORK_DATE']>=_bt_last_begin) \
                        & (calendar['WORK_DATE']<=_bt_last_end)]

    predict_date = predict_date['WORK_DATE'].tolist()    



def set_calendar():
    
    global sam_calendar, sam_predict_date, sam_predict_week
    
    # New Vars
    global calendar, calendar_lite, _bt_last_end
    
    
    # Set Calendar Last ......
    calendar = sam_calendar[sam_calendar['TRADE_DATE']>0]
    
    if _time_unit == 'd':
        calendar = calendar[time_key]
        sam_predict_date = sam_predict_date[time_key]
        
    elif _time_unit == 'w':
        calendar = calendar[['WORK_DATE'] + time_key]
        
        
    calendar = calendar[time_key] \
                    .drop_duplicates() \
                    .reset_index(drop=True)

    # df_add_shift will cause NA, it's essential to drop to convert to int
    calendar, _ = \
        cbyz.df_add_shift(df=calendar, cols=time_key, shift=1, 
                          group_by=[], sort_by=time_key, suffix='_LAST', 
                          remove_na=True)

    calendar = cbyz.df_conv_col_type(df=calendar, 
                                     cols=calendar.columns,
                                     to='int')
    
    if _time_unit == 'd':
        calendar = calendar.merge(sam_predict_date, on=time_key)
        calendar_lite = calendar[['WORK_DATE', 'WORK_DATE_LAST']]
        
    elif _time_unit == 'w':
        calendar = calendar.merge(sam_predict_week, on=time_key)
        calendar_lite = calendar[['YEAR', 'WEEK_NUM', 
                                  'YEAR_LAST', 'WEEK_NUM_LAST']]

    _bt_last_end = sam_calendar.loc[len(sam_calendar) - 1, 'WORK_DATE']


# .........


def set_frame():

    global id_keys, time_key, time_key_last
    global symbol
    global calendar, calendar_lite, frame
    global _bt_last_begin, _bt_last_end, _predict_period, _time_unit
    global predict_date
    global actions_main, bt_result
    
    global actions_main, hist_main
    global ohlc, var_y
    
    # New Vars
    global ohlc_ratio, ohlc_last
    global var_y_last, var_y_hist
    
    
    # LAST用來參考，HIST用來算Precision
    # - 因為y可能是price，也可能是ratio，所以在filter dataframe時，盡量用ohlc相關的
    #   變數，不要用var_y
    ohlc = stk.get_ohlc(orig=True, ratio=False)
    ohlc_ratio = stk.get_ohlc(orig=False, ratio=True)
    ohlc_last = [i + '_LAST' for i in ohlc]

    var_y_last = [i + '_LAST' for i in var_y]
    var_y_hist = [i + '_HIST' for i in var_y]


    # Hist Data ......
    # - predict_period * 2 to ensure to get the complete data
    # - No matter your y is the price or the change ratio, it is essential to
    #   keep both information in the actions_main
    bt_first_begin = \
        cbyz.date_cal(_bt_last_begin, 
                      -_interval * _bt_times - _predict_period * 2, 
                      _time_unit)    
    
    hist_data_raw = stk.get_data(data_begin=bt_first_begin, 
                                 data_end=_bt_last_end, 
                                 market=market, 
                                 symbol=symbol, 
                                 price_change=True,
                                 restore=False)
    
    if _time_unit == 'w':
        # The aggregate method should be the same with SAM
        hist_data_raw = hist_data_raw \
            .merge(sam_calendar, how='left', on='WORK_DATE') \
            .drop('WORK_DATE', axis=1)
            
        hist_data_raw = hist_data_raw \
                        .groupby(id_keys) \
                        .mean() \
                        .reset_index()
    
    if 'CLOSE' in var_y:
        hist_data_raw = hist_data_raw[['WORK_DATE', 'SYMBOL'] + ohlc]
    else:
        rename_dict = cbyz.li_to_dict(ohlc, ohlc_last)
        hist_data_raw = hist_data_raw[id_keys + ohlc + ohlc_ratio] \
                        .rename(columns=rename_dict)
            
    
    # Unique Symbols
    # - 應該用bt_result，不能用hist_data_raw，否則會出現低交易量，而被SAM排除
    #   的個股
    if len(symbol) > 0:
        symbol_df = pd.DataFrame({'SYMBOL':symbol})
    else:
        temp_symbol = bt_result['SYMBOL'].unique().tolist()
        symbol_df = pd.DataFrame({'SYMBOL':temp_symbol})        
    
    
    # Set Frame ......
    frame = cbyz.df_cross_join(symbol_df, calendar_lite)
    frame = frame \
            .sort_values(by=id_keys) \
            .reset_index(drop=True)

    # 
    rename_dict = cbyz.li_to_dict(var_y + time_key, 
                                  var_y_last + time_key_last) 
    actions_main = hist_data_raw.rename(columns=rename_dict)
            
    actions_main = frame \
        .merge(actions_main, how='left', on=['SYMBOL'] + time_key_last) \
        .merge(bt_result, how='left', on=id_keys)
    
        
    # Hist Main
    # - Real market data, used to inspect when backtesting
    rename_dict = cbyz.li_to_dict(time_key + var_y,
                                  time_key_last + var_y_hist)        
    hist_main = hist_data_raw.rename(columns=rename_dict)

    hist_main = frame \
        .merge(hist_main, how='left', on=['SYMBOL'] + time_key_last) \
        .drop(time_key_last, axis=1)
        

# ..........
    

def backtest_predict(bt_last_begin, predict_period, interval, 
                     data_period):
    
    global symbol, _market, bt_info, _bt_times, _ma_values
    global _time_unit
    global dev, test

    # New Global Vars
    global bt_results_raw, bt_result
    global precision, features, var_y, _volume_thld
    global sam_calendar, sam_predict_date, sam_predict_week
    global pred_scores, pred_features, pred_params
    
    
    if load_result:
        bt_result_file = path_temp + '/bt_result_' + _time_unit + '.csv'
        precision_file = path_temp + '/precision_' + _time_unit + '.csv'
        sam_calendar_file = path_temp + '/sam_calendar_' + _time_unit + '.csv'        
        sam_predict_date_file = path_temp + '/sam_predict_date_' \
                                + _time_unit + '.csv'

        sam_predict_week_file = path_temp + '/sam_predict_week_' \
                                + _time_unit + '.csv'

        today = cbyz.date_get_today()
        
        # 為了避免跨日的問題，多計算一天
        bt_result_mdate = cbyz.os_get_file_modify_date(bt_result_file)
        bt_result_mdate = cbyz.date_cal(bt_result_mdate, 1, 'd')
        bt_result_date_diff = cbyz.date_diff(today, bt_result_mdate, 
                                             absolute=True)        

        precision_mdate = cbyz.os_get_file_modify_date(precision_file)
        precision_mdate = cbyz.date_cal(precision_mdate, 1, 'd')
        prec_date_diff = cbyz.date_diff(precision_mdate, bt_result_mdate, 
                                        absolute=True)
        

        if os.path.exists(bt_result_file) and os.path.exists(precision_file) \
            and bt_result_date_diff <= 2 and  prec_date_diff <= 2:
            
            bt_result = pd.read_csv(bt_result_file)
            bt_result['SYMBOL'] = bt_result['SYMBOL'].astype('str')
            
            sam_calendar = pd.read_csv(sam_calendar_file)
            
            
            sam_predict_date = pd.read_csv(sam_predict_date_file)
            
            # if _time_unit == 'w':
            #     sam_predict_week = pd.read_csv(sam_predict_week_file)
            #     precision = pd.read_csv(precision_file)
                
            sam_predict_week = pd.read_csv(sam_predict_week_file)
            precision = pd.read_csv(precision_file)                
                
            
            var_y = cbyz.df_get_cols_except(
                df=bt_result,
                except_cols=['SYMBOL', 'WORK_DATE', 'BACKTEST_ID']
                )
            return
    
    
    # Prepare For Backtest Records ......
    bt_info_raw = cbyz.date_get_seq(begin_date=bt_last_begin,
                                    seq_length=_bt_times,
                                    unit='d', interval=-interval,
                                    simplify_date=True)
    
    bt_info = bt_info_raw[['WORK_DATE']] \
            .reset_index() \
            .rename(columns={'index':'BACKTEST_ID'})
    
    bt_info.loc[:, 'DATA_PERIOD'] = data_period
    bt_info.loc[:, 'PREDICT_PERIOD'] = predict_period
    bt_info = bt_info.drop('WORK_DATE', axis=1)
    
    bt_seq = bt_info_raw['WORK_DATE'].tolist()
    
    
    # Work area ----------
    bt_results_raw = pd.DataFrame()
    precision = pd.DataFrame()
    
    sam_calendar = None
    sam_predict_date = None
    sam_predict_week = None
    
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
                
            pred_params = sam_params.copy()                
            
            # Get Calendar
            sam_calendar = sam.calendar.copy()
            sam_predict_date = sam.predict_date.copy()
            sam_predict_week = sam.predict_week.copy()
            
        bt_results_raw = bt_results_raw.append(sam_result)
        precision = precision.append(sam_scores)


    # Organize ......
    bt_result = ar.df_simplify_dtypes(df=bt_results_raw)
    bt_result = bt_result.reset_index(drop=True)
    
    var_y = cbyz.df_get_cols_except(
        df=bt_result, 
        except_cols=id_keys + ['BACKTEST_ID']
        )

    if len(bt_result) > 800 or dev:
        bt_result.to_csv(path_temp + '/bt_result_' + _time_unit + '.csv',
                         index=False)
        
        sam_calendar.to_csv(path_temp + '/sam_calendar_' + _time_unit + '.csv',
                            index=False)
        
        sam_predict_date.to_csv(path_temp + '/sam_predict_date_' \
                                + _time_unit + '.csv',
                                index=False)        
        
        sam_predict_week.to_csv(path_temp + '/sam_predict_week_' \
                                + _time_unit + '.csv',
                                index=False)        

        precision.to_csv(path_temp + '/precision_' + _time_unit + '.csv',
                         index=False)


# ............


def cal_profit(y_thld=2, time_thld=10, prec_thld=0.15, execute_begin=None,
               export_file=True, load_file=False, path=None, file_name=None):
    '''
    應用場景有可能全部作回測，而不做任何預測，因為data_end直接設為bt_last_begin
    '''
    
    
    global ohlc, ohlc_last
    global _predict_period
    global _interval, _bt_times 
    global bt_result, rmse, bt_main, actions, var_y
    global symbol, _market
    global _bt_last_begin, _bt_last_end    
    global calendar
    global ohlc, ohlc_ratio, ohlc_last
    global var_y, var_y_last, var_y_hist
    global time_key, time_key_last
    global _hold

        
    # Merge hist data ......
    global frame, actions_main
    global precision
    
    
    # .......
    if 'CLOSE_CHANGE_RATIO' in var_y:
        main_data = actions_main[['BACKTEST_ID', 'SYMBOL'] \
                                 + time_key + time_key_last \
                                 + var_y + var_y_last + ohlc_last]
    elif 'CLOSE' in var_y:
        main_data = actions_main[['BACKTEST_ID', 'SYMBOL'] \
                                 + time_key + time_key_last \
                                 + var_y + var_y_last]

            
    main_data = cbyz.df_fillna(df=main_data, 
                               cols=ohlc_last, 
                               sort_keys=['SYMBOL', 'WORK_DATE'],
                               method='ffill') 
            

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
    bt_main, actions = \
        stk.gen_predict_action(df=main_data,
                               precision=precision,
                               date=time_key, 
                               last_date=time_key_last, 
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
    try:
        sam.low_volume_symbols
    except:
        pass
    else:
        if len(sam.low_volume_symbols) > 0:
            low_volume_df = pd.DataFrame({'SYMBOL':sam.low_volume_symbols})
            actions = actions.merge(low_volume_df, how='outer', on='SYMBOL')
        
        
    # Add name ......
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    stock_info = stock_info[['SYMBOL', 'STOCK_NAME', 'INDUSTRY']]
    actions = actions.merge(stock_info, how='left', on='SYMBOL')
    
    
    # Hold Symbols
    actions['HOLD'] = np.where(actions['SYMBOL'].isin(_hold), 1, 0)

    # Initialize Signal
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
        actions.loc[:, 'CLOSE_CHANGE'] = \
            actions['CLOSE'] - actions['CLOSE_LAST']
    
        actions.loc[:, 'CLOSE_CHANGE_RATIO'] = \
            actions['CLOSE_CHANGE'] / actions['CLOSE_LAST']
    
    # Rearrange Columns ......       
    profit_cols = ['CLOSE_CHANGE', 'CLOSE_CHANGE_RATIO', 
           'RECORD_PRECISION_MEDIAN', 'RECORD_PRECISION_STD', 
           'DIFF_MEDIAN', 'DIFF_STD']
        
    
    cols_1 = ['SYMBOL', 'STOCK_NAME', 'INDUSTRY',
              'BUY_SIGNAL', 'DAY_TRADING_SIGNAL', 'HOLD']

    cols_2 = ['PRECISION_'+ s for s in var_y]    
    
    new_cols = cols_1 + time_key + time_key_last + profit_cols \
                + ohlc + ohlc_last + cols_2


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
    
    # # Decrease On First Day ...
    # cond1 = actions[(actions['WORK_DATE']==_bt_last_begin) \
    #                & (actions['CLOSE_CHANGE_RATIO']<0)]
    # cond1 = cond1['SYMBOL'].unique().tolist()
    
    # # Estimated Profit ...
    # cond2 = actions[(actions['WORK_DATE']>_bt_last_begin) \
    #                & (actions['CLOSE_CHANGE_RATIO']>=y_thld)]
    # cond2 = cond2['SYMBOL'].unique().tolist()    
    
    # # Max Error ...
    # cond3 = actions[actions['DIFF_MEDIAN']<prec_thld]
    # cond3 = cond3['SYMBOL'].unique().tolist()       
    
    # buy_signal_symbols = cbyz.li_intersect(cond1, cond2, cond3)
    
    
    print('Close裡面有NA，可能是已經下檔的Symbol？')
    # Cannot convert non-finite values (NA or inf) to integer
    
    global cal_profit_debug
    cal_profit_debug = actions[actions['CLOSE'].isna()]
    
    actions = cbyz.df_conv_na(df=actions, cols='CLOSE', value=-1000)    
    
    # Add Level
    actions['PERCENTAGE'] = actions['CLOSE_CHANGE_RATIO'] * 100

    # There may be some inf values
    actions = cbyz.df_conv_na(df=actions, cols='PERCENTAGE', value=-1000)
    actions = cbyz.df_handle_inf(df=actions, cols='PERCENTAGE', value=-1000, 
                                 drop=False, axis=0)
    
    actions['PERCENTAGE'] = actions['PERCENTAGE'].astype('int')
      
    # actions['BUY_SIGNAL'] = \
    #     np.where(actions['SYMBOL'].isin(buy_signal_symbols), 
    #              99, actions['PERCENTAGE'])

    # 20220209 - 暫時移除BUY_SIGNAL for weekly forecast，先直接採用PERCENTAGE
    actions['BUY_SIGNAL'] = actions['PERCENTAGE']



# .................


def eval_metrics(export_file=False, threshold=800):
    '''
    

    Parameters
    ----------
    export_file : TYPE, optional
        DESCRIPTION. The default is False.
    threshold : TYPE, optional
        DESCRIPTION. The default is 800.

    Returns
    -------
    None.

    '''

    
    global _market
    global bt_main, bt_info, rmse
    global var_y, var_y_last, var_y_hist
    global serial, compete_mode
    global id_keys
    
    
    loc_main = bt_result.merge(hist_main, on=id_keys)
    loc_main = loc_main.dropna(axis=0)

    # All hist columns will be NA if not a backtest
    if len(loc_main) == 0:
        return
    
    # ......
    result = pd.DataFrame()
    
    for i in range(len(var_y)):
        y = var_y[i]
        y_hist = var_y_hist[i]
        
        
        if 'CHANGE_RATIO' in y:
            loc_main['PRECISION'] = loc_main[y] - loc_main[y_hist]
        else:
            loc_main['PRECISION'] = \
                (loc_main[y] - loc_main[y_hist]) / loc_main[y_hist]            
                            
        loc_main['PRECISION'] = loc_main['PRECISION'].abs()     


        new_result = loc_main[id_keys + ['PRECISION']]
        new_result = new_result.assign(Y=y)
        result = result.append(new_result)
                            
    
    result = result.dropna(axis=0)
    result.loc[:, 'MARKET'] = _market
    result.loc[:, 'PRECISION_METRIC'] = 'MSE'
    result.loc[:, 'SERIAL'] = serial
    
    # 如果讀取暫存檔的話，會抓不到version
    try:
        result.loc[:, 'VERSION'] = sam.version
    except:
        result.loc[:, 'VERSION'] = 0
    
    
    # Not Finished Yet
    result['SCORE_METRIC'] = 'R2'
    result['MODEL_SCORE'] = 0
    result = result.assign(PARAMS='')
    
    
    # WORK_DATE means predict date.
    result = result[['MARKET', 'VERSION', 'SYMBOL', 'SERIAL', 'Y'] \
                    + time_key \
                    + ['SCORE_METRIC', 'MODEL_SCORE', 
                      'PRECISION_METRIC', 'PRECISION', 'PARAMS']]
    

    # Save model only when retrain model
    if len(bt_main) > threshold and _compete_mode == 2:
        try:
            ar.db_upload(data=result, table_name='backtest_records')
        except Exception as e:
            print(e)
            result.to_csv(path_temp + '/backtest_records_' + serial + '.csv',
                          index=False)
    


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
            
    summary.loc[:, 'LAST_DATE'] = last_date
            
    # Write
    stk.write_sheet(data=summary, sheet='YD_Industry')            
        


# %% Master ------


def master(bt_last_begin, predict_period=14, time_unit='d', long=False, 
           interval=360, bt_times=2, data_period=5, ma_values=[5,10,20,60], 
           volume_thld=400, cv=2, compete_mode=1, market='tw', hold=[]):
    '''
    主工作區
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
    # - Include low volume symbol in the excel
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
    # - hold symbol沒有順利標示 - Done
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
    # - Update eval_metrics and rebuild forecast_records
    # v0.075
    # - Fix bug
    # v0.076
    # - Add capability to read saved bt_result.csv > Done
    # - Add date to result of view_yesterday > Done
    # v0.077 - 20220118
    # - Rename symbols as symbol
    # - 是不是還是必須把bt mape和score合併
    # - 權殖股，依照產業列出rank
    # - view_yesterday and view_industry is the same
    # v0.078 - 20220119
    # - Add form and ml_df_to_time_series in sam
    # v0.0781 - 20220124
    # - Add Time Unit
    # - MVP version for SAM weekly prediction, but the calendar in BTM broke
    # - Before modifing for BTM calendar
    # v0.079
    # - Remove set_calendar, then get calendar from SAM whose BACKTEST_ID is 0
    # v0.08
    # - Add suffix of time unit for bt_result saved file
    # - Combine result of daily prediction and weekly prediction
    # - Rename dev mode and test mode
    
    # v1.000 - 20220214
    # - Temp
    

    # Bug
    # 1. Fix UT, execution correlation before split data, but not drop columns
    #    inmmediately
    # 5. print predict date in sam to fix missing date issues    
    # 6. 如果local沒有forecast_records時，cal_profit中的get_forecast_records會出錯：
    #    AttributeError: 'list' object has no attribute 'rename'

    
    # Optimization
    # 4. Calculate IRR, remove outliers
    # 5. Google Sheet Add Manual Tick
    # 6. Think how to optimize stop loss
    # 7. Backtest也可以用parameter做A/B        
    # Add signal type: 接下來全部漲，或是先跌再漲



    # Worklist
    # 1. Remove Open
    # 2. 測試如果week_begin不是星期一時，是否會出錯
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
    # 18. When backtest, check if the symbol reached the target price in the 
    #     next N weeks.
    # 19. Analysis, 近60日投報率，用產業或主題作分群 / 財報
    # 20. Add Sell Signal
    # 21. 產業上中下游關係，SNA
    # 22. Update, load last version of model
    # 24. 把股性分群；隱含波動率
    # 25. Do actions by change
    # 26. Add Auto-Competing Model    
    # 27. Signal A and B，A是反彈的，B是low和close差距N%
    # 交易資料先在dcm中算好ma
    # global parama 應該是在一開始就訂好，而不是最後才收集，參考rtml
    # 把market改成market
    # 28. 當bt_times為2，且load_file=false時，重新訓練一次模型就好


    # # Not Collected Parameters ......
    # global dev, test
    # dev = True
    # test = True
    # bt_times = 1
    # bt_index = 0
    # interval = 4
    # market = 'tw'
    # time_unit = 'd'
    # time_unit = 'w'
    # hold = []
    
    
    # # Collected Parameters ......
    # bt_last_begin = 20220207
    # predict_period = 2
    # data_period = int(365 * 3.5)
    # ma_values = [6,10,20,60]
    # volume_thld = 500
    # hold = [8105, 2610, 3051, 1904, 2611]
    # long = False
    # compete_mode = 0
    # cv = list(range(2, 3))
    # load_result = False
    # dev = True
    # predict_begin = bt_last_begin
    
    
    # Keys ------
    global id_keys, time_key, time_key_last
    
    if time_unit == 'w':
        id_keys = ['SYMBOL', 'YEAR', 'WEEK_NUM']
        time_key = ['YEAR', 'WEEK_NUM']
        
    elif time_unit == 'd':
        id_keys = ['SYMBOL', 'WORK_DATE']    
        time_key = ['WORK_DATE']    

    time_key_last = [y + '_LAST' for y in time_key]

    
    # Worklist
    # 1.Add price increse but model didn't catch
    global _interval, _bt_times, _volume_thld, _ma_values, _hold
    global symbol, _market
    global _bt_last_begin, _bt_last_end
    global serial

    _hold = [str(i) for i in hold]
    serial = cbyz.get_time_serial(with_time=True)


    # Parameters ......
    
    # 0 for original, 1 for MA, 2 for shifted time series
    data_form = 1

    # Data Forme Issues
    # - 沒辦法像machinlearningmastery的範例，把全部的資料當成
    #   time series處理，因為我用了很多的資料集，像是三大法人持股成本和COVID-19，代表這些
    #   欄位全用都需要用time series的方式處理，才有辦法預測第2-N天    
    #   https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
    # - 可以把predict_period全部設為1，但調整time_unit
    if data_form == 2:
        predict_period = 1


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


    if dev or test:
        symbol = [2520, 2605, 6116, 6191, 3481, 
                   2409, 2520, 2603, 2409, 2603, 
                   2611, 3051, 3562, 2301, 
                   '2211', '3138', '3530']
    else:
        symbol = []
        
        
    global _compete_mode, _time_unit
    _market = market    
    _compete_mode = compete_mode
    _time_unit = time_unit


    # Arguments
    # 1. 目前的parameter都只有一個，但如果有多個parameter需要做A/B test時，應該要在
    #    在btm中extract，再把single parameter的Param_Holder傳入sam，因為
    #    Param_Holder的參數會影響到model_data，沒辦法同一份model_data持續使用，因此，
    #    把完整的Param_Holder傳入sam再extract沒有任何效益
    
    args = {'bt_last_begin':[bt_last_begin],
            'time_unit':[time_unit],
            'predict_period': [predict_period], 
            'data_period':[data_period],
            'data_form':[data_form],
            'ma_values':[ma_values],
            'volume_thld':[volume_thld],
            'industry':[True],
            'trade_value':[True],
            'market':['tw'],
            'compete_mode':[_compete_mode],
            'train_mode':[2],            
            'cv':[cv],
            'kbest':['all'],
            'dev':[dev],
            'test':[test],
            'symbol':[symbol],
            'long':[long]
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
    symbol = cbyz.li_conv_ele_type(symbol, to_type='str')


    # Set Date ......
    global _bt_last_begin, _predict_period
    global calendar, _bt_last_end
    _predict_period = predict_period
    _bt_last_begin = bt_last_begin
    

    # Predict ------
    global bt_result, precision, features
    backtest_predict(bt_last_begin=bt_last_begin, 
                     predict_period=_predict_period, 
                     interval=interval,
                     data_period=data_period)
    
    
    # Set Calendar ------
    set_calendar()


    # Set Date ......
    set_frame()
    
    
    # Debug for prices columns issues
    if len(bt_result) > 800:
        stk.write_sheet(data=bt_result, sheet='Debug')
    
    
    # Profit ------    
    # y_thld=0.02
    # time_thld=predict_period
    # prec_thld=0.05
    # export_file=True
    # load_file=True
    # path=path_temp
    # file_name=None        
    

    global bt_main, actions
    
    # Optimize, 這裡的precision_thld實際上是mape, 改成precision
    global mape, mape_group, mape_extreme
    global stock_metrics_raw, stock_metrics    
    
    today = cbyz.date_get_today()
    execute_begin = cbyz.date_cal(today, -14, 'd')
    execute_begin = int(str(execute_begin) + '0000')
    
    
    # Evaluate Precision ......
    print('Bug - get_forecast_records中的Action Score根本沒用到，但可以用signal替代')
    cal_profit(y_thld=0.02, time_thld=_predict_period, prec_thld=0.05,
               execute_begin=execute_begin, 
               export_file=True, load_file=True, path=path_temp,
               file_name=None)
    
    eval_metrics()
    
    
    # Write Google Sheets ...... 
    if len(actions) > 800:
        
        # # Action Workbook
        # stk.write_sheet(data=actions, sheet='TW', long=long,
        #                 predict_begin=_bt_last_begin)
    
        # # View And Log .....
        # view_yesterday()

        
        # # Error when load_result = True
        # global pred_features
        # try:
        #     stk.write_sheet(data=pred_features, sheet='Features')
        # except Exception as e:
        #     print(e)
            
        pass
    
    gc.collect()

    return actions



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
    
    
    symbol = file_raw['STOCK_SYMBOL'].unique().tolist()
    
    
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
                & (data_raw['STOCK_SYMBOL'].isin(symbol))] \
            .sort_values(by='PRICE_CHANGE_RATIO', ascending=False) \
            .reset_index(drop=True)

    
    main_data = data.merge(file, how='left', on='STOCK_SYMBOL')
    

    

# %% Dev -----

def cbmbine_action():
    
    global action_weekly, action_daily    
    action_key = ['SYMBOL', 'STOCK_NAME', 'INDUSTRY']
    
    
    cols = cbyz.df_get_cols_except(df=action_daily, except_cols=action_key)
    new_cols = ['DAY_' + c for c in cols]
    rename_dict = cbyz.li_to_dict(cols, new_cols)
    action_daily = action_daily.rename(columns=rename_dict)
    
    action_final = action_weekly \
        .merge(action_daily, how='outer', on=action_key)
    
    stk.write_sheet(data=action_final, sheet='Combine')



# %% Execute ------

if __name__ == '__main__':
    

    # SAM Note ------
    # 1. 20220107 v2.06 - 原本在normalize的時候，會group by symbol，讓每一檔都和自己
    #   比較，否則高價股和低價股比感覺很虧。這個版本試著把sam_load_data中的group by
    #   改成[]。經測試過後，R2差不多，所以保留新的版本，應該可以提高計算速度。
    # 2. y為price時會一直overfitting

    # BTM Note
    # 1. 如果用change_ratio當成Y的話，對模型來說，最安全的選項是不是設為0？
    # 2. 當price為y時，industry的importance超高，可能是造成overfitting的主因
    # 3. 以industry作為target_type時，一定要用change_ratio當Y
    # 4. data_form為1，y為change_ratio時
    # - MA為[5,10,20,60,120]，open和low較佳；MA為[5,10,20,60]，high和close較佳，
    #   但兩者MSE差異只在0.01等級
    # 5. time_unit為week，且data_form為1時，兩者都會讓曲線變得更加平滑，因此可能
    #    低估振幅，但強制縮放成-0.1 - 0.1也不妥，如果整體盤勢向下，這個方法會
    #    高估走勢

    
    # Change Ratio - XGB Params ------
    # eta: 0.3 / 0.01, 0.03, 0.08, 0.1, 0.2
    # min_child_weight: 0.8 / 1
    # max_depth: 10 / 8, 12
    # subsample: 1 / 0.8    
    

    # Price - XGB Params ------
    # eta: 0.2 /
    # min_child_weight: 0.8 / 
    # max_depth: 10 / 8, 12
    # subsample: 1 / 0.8
    
    
    global weekly_actions, daily_actions
    hold = [3596, 6698]
    
    global dev, test, load_result
    dev = True
    test = True
    load_result = True    
    load_result = False
    
    global action_weekly, action_daily
    
    # test mode take few data to run, and dev mode will decrease the threshold
    # to export temp file
    
    
    # Dev - Week
    action_weekly = \
        master(bt_last_begin=20220214, predict_period=1, 
            time_unit='w',long=False, interval=4, bt_times=1, 
            data_period=int(365 * 1), 
            ma_values=[5,10,20], volume_thld=400,
            compete_mode=0, cv=list(range(3, 4)),
            market='tw', hold=hold)
        
    # stk.write_sheet(data=actions, sheet='Week')
    
    
    # Dev - Day
    action_daily = \
        master(bt_last_begin=20220214, predict_period=1, 
               time_unit='d', long=False, interval=4, bt_times=1, 
               data_period=int(365 * 1), 
               ma_values=[5,10,20], volume_thld=400,
               compete_mode=0, cv=list(range(3, 4)),
               market='tw', hold=hold)
        
    
    # # Production - Week
    # weekly_actions = \
    #     master(bt_last_begin=20220207, predict_period=1, 
    #            time_unit='w',long=False, interval=4, bt_times=1, 
    #            data_period=int(365 * 5), 
    #            ma_values=[5,10,20,60], volume_thld=400,
    #            compete_mode=0, cv=list(range(3, 4)),
    #            market='tw', hold=hold)


    # # Production - Day
    # weekly_actions = \
    #     master(bt_last_begin=20220207, predict_period=1, 
    #            time_unit='d',long=False, interval=4, bt_times=1, 
    #            data_period=int(365 * 5), 
    #            ma_values=[5,10,20,60], volume_thld=400,
    #            compete_mode=0, cv=list(range(3, 4)),
    #            market='tw', hold=hold)
        
        
        
        