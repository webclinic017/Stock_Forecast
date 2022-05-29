#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import random

host = 3
# host = 2
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
    path = r'D:\GitHub\Stock_Forecast\3_Backtest'
    path_sam = r'D:\GitHub\Stock_Forecast\2_Stock_Analysis'    


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'D:\GitHub\Arsenal',
                 r'D:\Data_Mining\Projects\Codebase_YZ',
                 r'/home/jupyter/Arsenal/20220522',
                 path + '/Function',
                 path_sam]


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
# import arsenal as ar
# import arsenal_stock as stk
import arsenal_v0200_dev as ar
import arsenal_stock_v0200_dev as stk
import stock_analysis_manager_v3_0806_dev as sam



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     



# %% inner function ------


def set_calendar():
    
    global sam_calendar, sam_predict_date, sam_predict_week
    
    # new vars
    global calendar, calendar_lite, _bt_last_end
    
    
    # set calendar last ......
    calendar = sam_calendar[sam_calendar['trade_date']>0]
    
    if _time_unit == 'd':
        calendar = calendar[time_key]
        sam_predict_date = sam_predict_date[time_key]
        
    elif _time_unit == 'w':
        calendar = calendar[['work_date'] + time_key]
        
        
    calendar = calendar[time_key] \
                    .drop_duplicates() \
                    .reset_index(drop=True)

    # df_add_shift will cause na, it's essential to drop to convert to int
    calendar, _ = \
        cbyz.df_add_shift(df=calendar, cols=time_key, shift=1, 
                          group_by=[], sort_by=time_key, suffix='_last', 
                          remove_na=True)

    calendar = cbyz.df_conv_col_type(df=calendar, 
                                     cols=calendar.columns,
                                     to='int')
    
    if _time_unit == 'd':
        calendar = calendar.merge(sam_predict_date, on=time_key)
        calendar_lite = calendar[['work_date', 'work_date_last']]
        
    elif _time_unit == 'w':
        calendar = calendar.merge(sam_predict_week, on=time_key)
        calendar_lite = calendar[['year_week_iso', 'year_week_iso_last']]

    _bt_last_end = sam_calendar.loc[len(sam_calendar) - 1, 'work_date']


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
    
    # new vars
    global ohlc_ratio, ohlc_last
    global var_y_last, var_y_hist
    
    
    # last用來參考，hist用來算precision
    # - 因為y可能是price，也可能是ratio，所以在filter dataframe時，盡量用ohlc相關的
    #   變數，不要用var_y
    ohlc = stk.get_ohlc(orig=True, ratio=false)
    ohlc_ratio = stk.get_ohlc(orig=false, ratio=True)
    ohlc_last = [i + '_last' for i in ohlc]

    var_y_last = [i + '_last' for i in var_y]
    var_y_hist = [i + '_hist' for i in var_y]


    # hist data ......
    # - predict_period * 2 to ensure to get the complete data
    # - no matter your y is the price or the change ratio, it is essential to
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
                                 adj=false)
    
    if _time_unit == 'w':
        # the aggregate method should be the same with sam
        hist_data_raw = hist_data_raw \
            .merge(sam_calendar, how='left', on='work_date') \
            .drop('work_date', axis=1)
            
        hist_data_raw = hist_data_raw \
                        .groupby(id_keys) \
                        .mean() \
                        .reset_index()
    
    if 'close' in var_y:
        hist_data_raw = hist_data_raw[['work_date', 'symbol'] + ohlc]
    else:
        rename_dict = cbyz.li_to_dict(ohlc, ohlc_last)
        hist_data_raw = hist_data_raw[id_keys + ohlc + ohlc_ratio] \
                        .rename(columns=rename_dict)
            
    
    # unique symbols
    # - 應該用bt_result，不能用hist_data_raw，否則會出現低交易量，而被sam排除
    #   的個股
    if len(symbol) > 0:
        symbol_df = pd.dataframe({'symbol':symbol})
    else:
        temp_symbol = bt_result['symbol'].unique().tolist()
        symbol_df = pd.dataframe({'symbol':temp_symbol})        
        
    
    # set frame ......
    frame = cbyz.df_cross_join(symbol_df, calendar_lite)
    frame = frame \
            .sort_values(by=id_keys) \
            .reset_index(drop=True)

    # 
    rename_dict = cbyz.li_to_dict(var_y + time_key, 
                                  var_y_last + time_key_last) 
    actions_main = hist_data_raw.rename(columns=rename_dict)


    # symbole will be integers when load_data = True ......
    print('optimize - it should not be here')
    bt_result = cbyz.df_conv_col_type(df=bt_result, cols='symbol',
                                       to='str')

    frame = cbyz.df_conv_col_type(df=frame, cols='symbol',
                                       to='str')
    
    actions_main = cbyz.df_conv_col_type(df=actions_main, cols='symbol',
                                         to='str')

    # ......
    actions_main = frame \
        .merge(actions_main, how='left', on=['symbol'] + time_key_last) \
        .merge(bt_result, how='left', on=id_keys)
    
        
    # hist main
    # - real market data, used to inspect when backtesting
    rename_dict = cbyz.li_to_dict(var_y, var_y_hist)        
    hist_main = hist_data_raw.rename(columns=rename_dict)

    hist_main = frame \
        .merge(hist_main, how='left', on=id_keys) \
        .drop(time_key_last, axis=1)
        

# ..........
    

def backtest_predict(bt_last_begin, predict_period, interval, 
                     data_period):
    
    global symbol, _market, bt_info, _bt_times, _ma_values
    global _time_unit, _bt_last_begin
    global dev, test, serial

    # new global vars
    global bt_results_raw, bt_result
    global precision, features, var_y, _volume_thld
    global sam_calendar, sam_predict_date, sam_predict_week
    global pred_scores, pred_features, pred_params
    
    
    suffix = _time_unit + '_' + str(_bt_last_begin)
    
    if load_result:
        
        # search for suffix and the serial of last executed
        files = cbyz.os_get_dir_list(path=path_temp, level=-1, 
                                     extensions='csv', remove_temp=True)
        
        files = files['files']
        files = files[files['file_name'].str.contains('bt_result_' + suffix)] \
                .sort_values(by='file_name', ascending=false) \
                .reset_index(drop=True)
        
        # file serial
        file_serial = files.loc[0, 'file_name']
        file_serial = file_serial[-19:- 4]
        
        
        # ......
        bt_result_file = path_temp + '/bt_result_' \
                        + suffix + '_' + file_serial + '.csv'
                            
        precision_file = path_temp + '/precision_' \
                        + suffix + '_' + file_serial + '.csv'
            
        sam_calendar_file = path_temp + '/sam_calendar_' \
                        + suffix + '_' + file_serial + '.csv'
                        
        sam_predict_date_file = path_temp + '/sam_predict_date_' \
                                + suffix + '_' + file_serial + '.csv'

        sam_predict_week_file = path_temp + '/sam_predict_week_' \
                                + suffix + '_' + file_serial + '.csv'

        today = cbyz.date_get_today()
        
        # 為了避免跨日的問題，多計算一天 ......
        print('這段真的需要嗎')
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
            bt_result['symbol'] = bt_result['symbol'].astype('str')
            
            sam_calendar = pd.read_csv(sam_calendar_file)
            
            
            sam_predict_date = pd.read_csv(sam_predict_date_file)
            
            # if _time_unit == 'w':
            #     sam_predict_week = pd.read_csv(sam_predict_week_file)
            #     precision = pd.read_csv(precision_file)
                
            sam_predict_week = pd.read_csv(sam_predict_week_file)
            precision = pd.read_csv(precision_file)                
                
            
            var_y = cbyz.df_get_cols_except(
                df=bt_result,
                except_cols=id_keys + ['backtest_id']
                )
            return
    
    
    # prepare for backtest records ......
    bt_info_raw = cbyz.date_get_seq(begin_date=bt_last_begin,
                                    seq_length=_bt_times,
                                    unit='d', interval=-interval,
                                    simplify_date=True)
    
    bt_info = bt_info_raw[['work_date']] \
            .reset_index() \
            .rename(columns={'index':'backtest_id'})
    
    bt_info.loc[:, 'data_period'] = data_period
    bt_info.loc[:, 'predict_period'] = predict_period
    bt_info = bt_info.drop('work_date', axis=1)
    
    bt_seq = bt_info_raw['work_date'].tolist()
    
    
    # work area ----------
    bt_results_raw = pd.dataframe()
    precision = pd.dataframe()
    features = pd.dataframe()
    
    sam_calendar = none
    sam_predict_date = none
    sam_predict_week = none
    
    # predict ......
    for i in range(0, len(bt_seq)):
        
        sam_result, sam_scores, \
            sam_params, sam_features = \
                sam.master(param_holder=param_holder,
                           predict_begin=bt_seq[i],
                           bt_index=i,
                           threshold=30000)

        sam_result['backtest_id'] = i
        sam_scores['backtest_id'] = i
        
        if i == 0:
            pred_scores = sam_scores.copy()

            # optimize, sort in sam or in ut            
            pred_features = sam_features.copy()
            pred_features = pred_features \
                .sort_values(by=['importance'], ascending=false) \
                .reset_index(drop=True)
                
            pred_params = sam_params.copy()                
            
            # get calendar
            sam_calendar = sam.calendar.copy()
            sam_predict_date = sam.predict_date.copy()
            sam_predict_week = sam.predict_week.copy()
            
        bt_results_raw = bt_results_raw.append(sam_result)
        precision = precision.append(sam_scores)
        features = features.append(sam_features)


    # organize ......
    bt_result = ar.df_simplify_dtypes(df=bt_results_raw)
    bt_result = bt_result.reset_index(drop=True)
    bt_result['sam_version'] = sam.version
    
    var_y = cbyz.df_get_cols_except(
        df=bt_result, 
        except_cols=id_keys + ['backtest_id', 'sam_version']
        )
    
    
    if _time_unit == 'd':
        global precision_day
        precision_day = precision.copy()
        
    elif _time_unit == 'w':
        global precision_week
        precision_week = precision.copy()
    

    if len(bt_result) > 300  or dev:
        bt_result.to_csv(path_temp + '/bt_result_' \
                         + suffix + '_' + serial + '.csv',
                         index=false)
        
        sam_calendar.to_csv(path_temp + '/sam_calendar_' \
                            + suffix + '_' + serial + '.csv',
                            index=false)
        
        sam_predict_date.to_csv(path_temp + '/sam_predict_date_' \
                                + suffix + '_' + serial + '.csv',
                                index=false)        
        
        sam_predict_week.to_csv(path_temp + '/sam_predict_week_' \
                                + suffix + '_' + serial + '.csv',
                                index=false)        

        precision.to_csv(path_temp + '/precision_' \
                         + suffix + '_' + serial + '.csv',
                         index=false)


# ............


def cal_profit(y_thld=2, time_thld=10, prec_thld=0.15, execute_begin=none,
               export_file=True, load_file=false, path=none, file_name=none):
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

        
    # merge hist data ......
    global frame, actions_main
    global precision
    
    
    # .......
    if 'close_change_ratio' in var_y:
        main_data = actions_main[['backtest_id', 'symbol'] \
                                 + time_key + time_key_last \
                                 + var_y + var_y_last + ohlc_last]
    elif 'close' in var_y:
        main_data = actions_main[['backtest_id', 'symbol'] \
                                 + time_key + time_key_last \
                                 + var_y + var_y_last]

            
    main_data = cbyz.df_fillna(df=main_data, 
                               cols=ohlc_last, 
                               sort_keys=['symbol', 'work_date'],
                               method='ffill') 
            

    # check na ......
    # 1. 這裡有na是合理的，因為hist可能都是na
    # 2. 在sam中被排除的symbol可能會出現在這，導致每一欄的na_count不一樣
    cbyz.df_chk_col_na(df=main_data)

    
    # if len(chk) > len(var_y):
    #     print('err01. cal_profit - main_data has na in columns.')
        
        
    if len(main_data) == 0:
        print(('error 1. main_data is empty. check the market data has been'
               ' updated or not, it may be the reason cause last price na.'))
        

    # generate actions ......
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
        
    msg = 'the max value of backtest_id should be 0.'
    assert actions['backtest_id'].max() == 0, msg
        

    # forecast records ......
    print('bug - get_forecast_records中的action score根本沒用到')
    records = stk.get_forecast_records(forecast_begin=none, 
                                       forecast_end=none,
                                       execute_begin=execute_begin,
                                       execute_end=none,
                                       y=['close_change_ratio'],
                                       summary=True)
    
    if len(records)  > 0:
        records = records \
            .rename(columns={'forecast_precision_median':'record_precision_median',
                             'forecast_precision_std':'record_precision_std'})
            
    # add low volume symbols    
    try:
        sam.low_volume_symbols
    except:
        pass
    else:
        if len(sam.low_volume_symbols) > 0:
            low_volume_df = pd.dataframe({'symbol':sam.low_volume_symbols})
            actions = actions.merge(low_volume_df, how='outer', on='symbol')
        
        
    # add name ......
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    stock_info = stock_info[['symbol', 'stock_name', 'industry']]
    actions = actions.merge(stock_info, how='left', on='symbol')
    
    
    # hold symbols
    actions['hold'] = np.where(actions['symbol'].isin(_hold), 1, 0)

    # initialize signal
    actions['buy_signal'] = np.nan 
    
    
    # add ohlc ......
    action_cols = actions.columns
    if 'open_change_ratio' in action_cols and 'open' not in action_cols:
        actions['open'] = actions['open_last'] \
                            * (1 + actions['open_change_ratio'])
    
    if 'high_change_ratio' in action_cols and 'high' not in action_cols:
        actions['high'] = actions['high_last'] \
                            * (1 + actions['high_change_ratio'])
        
    if 'low_change_ratio' in action_cols and 'low' not in action_cols:
        actions['low'] = actions['low_last'] \
                            * (1 + actions['low_change_ratio'])

    if 'close_change_ratio' in action_cols and 'close' not in action_cols:
        actions['close'] = actions['close_last'] \
                            * (1 + actions['close_change_ratio'])


    # day trading signal ......
    actions['day_trading_signal'] = actions['high'] - actions['low']
        
    
    if 'close_change' not in action_cols:
        actions.loc[:, 'close_change'] = \
            actions['close'] - actions['close_last']
    
        actions.loc[:, 'close_change_ratio'] = \
            actions['close_change'] / actions['close_last']
    
    # rearrange columns ......       
    profit_cols = ['close_change', 'close_change_ratio', 
           'record_precision_median', 'record_precision_std', 
           'diff_median', 'diff_std']
        
    
    cols_1 = ['symbol', 'stock_name', 'industry',
              'buy_signal', 'day_trading_signal', 'hold']

    cols_2 = ['precision_'+ s for s in var_y]    
    
    # before remove open from var_y
    # new_cols = cols_1 + time_key + time_key_last + profit_cols \
    #             + ohlc + ohlc_last + cols_2
                
    new_cols = cols_1 + time_key + time_key_last + profit_cols \
                + ['high', 'low', 'close', 
                   'high_last', 'low_last', 'close_last'] \
                + ohlc_last + cols_2


    # merge data ......
    if len(records) > 0:
        actions = actions.merge(records, how='left', on=['symbol'])
    
        actions.loc[:, 'diff_median'] = \
            actions['close_change_ratio'] - actions['record_precision_median']
    
        actions.loc[:, 'diff_std'] = \
            actions['close_change_ratio'] - actions['record_precision_std']
    
    else:
        actions['record_precision_std'] = np.nan
        actions['record_precision_median'] = np.nan
        actions['diff_median'] = np.nan
        actions['diff_std'] = np.nan
    
    actions = actions[new_cols]

        
    # buy signal ......
    # 是不是可以移到gen_predict_action
    
    # # decrease on first day ...
    # cond1 = actions[(actions['work_date']==_bt_last_begin) \
    #                & (actions['close_change_ratio']<0)]
    # cond1 = cond1['symbol'].unique().tolist()
    
    # # estimated profit ...
    # cond2 = actions[(actions['work_date']>_bt_last_begin) \
    #                & (actions['close_change_ratio']>=y_thld)]
    # cond2 = cond2['symbol'].unique().tolist()    
    
    # # max error ...
    # cond3 = actions[actions['diff_median']<prec_thld]
    # cond3 = cond3['symbol'].unique().tolist()       
    
    # buy_signal_symbols = cbyz.li_intersect(cond1, cond2, cond3)
    
    
    print('close裡面有na，可能是已經下檔的symbol？')
    # cannot convert non-finite values (na or inf) to integer
    
    actions = cbyz.df_conv_na(df=actions, cols='close', value=-1000)    
    
    # add level
    actions['percentage'] = actions['close_change_ratio'] * 100

    # there may be some inf values
    actions = cbyz.df_conv_na(df=actions, cols='percentage', value=-1000)
    actions = cbyz.df_handle_inf(df=actions, cols='percentage', value=-1000, 
                                 drop=false, axis=0)
    
    actions['percentage'] = actions['percentage'].astype('int')
      
    # actions['buy_signal'] = \
    #     np.where(actions['symbol'].isin(buy_signal_symbols), 
    #              99, actions['percentage'])

    # 20220209 - 暫時移除buy_signal for weekly forecast，先直接採用percentage
    actions['buy_signal'] = actions['percentage']



# .................


def eval_metrics(export_file=false, threshold=800):
    '''
    

    parameters
    ----------
    export_file : type, optional
        description. the default is false.
    threshold : type, optional
        description. the default is 800.

    returns
    -------
    none.

    '''

    
    global _market
    global bt_main, bt_info, rmse
    global var_y, var_y_last, var_y_hist
    global serial, compete_mode
    global id_keys
    
    
    loc_main = bt_result.merge(hist_main, on=id_keys)
    loc_main = loc_main.dropna(axis=0)

    # all hist columns will be na if not a backtest
    if len(loc_main) == 0:
        return
    
    # ......
    result = pd.dataframe()
    
    for i in range(len(var_y)):
        y = var_y[i]
        y_hist = var_y_hist[i]
        
        
        if 'change_ratio' in y:
            loc_main['precision'] = loc_main[y] - loc_main[y_hist]
        else:
            loc_main['precision'] = \
                (loc_main[y] - loc_main[y_hist]) / loc_main[y_hist]            
                            
        loc_main['precision'] = loc_main['precision'].abs()     


        new_result = loc_main[id_keys + ['precision']]
        new_result = new_result.assign(y=y)
        result = result.append(new_result)
                            
    
    result = result.dropna(axis=0)
    result.loc[:, 'market'] = _market
    result.loc[:, 'precision_metric'] = 'mse'
    result.loc[:, 'serial'] = serial
    
    # 如果讀取暫存檔的話，會抓不到version
    try:
        result.loc[:, 'version'] = sam.version
    except:
        result.loc[:, 'version'] = 0
    
    
    # not finished yet
    result['score_metric'] = 'r2'
    result['model_score'] = 0
    result = result.assign(params='')
    
    
    # work_date means predict date.
    result = result[['market', 'version', 'symbol', 'serial', 'y'] \
                    + time_key \
                    + ['score_metric', 'model_score', 
                      'precision_metric', 'precision', 'params']]
    

    # save model only when retrain model
    if len(bt_main) > threshold and _compete_mode == 2:
        try:
            ar.db_upload(data=result, table_name='backtest_records')
        except exception as e:
            print(e)
            result.to_csv(path_temp + '/backtest_records_' + serial + '.csv',
                          index=false)
    


# %% view and log ------


def view_yesterday():

    # last_date might be float
    global actions
    last_date = actions['last_date'].min()
    last_date = int(last_date)
        

    # stock info
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    stock_info = stock_info[['symbol', 'industry']]
    
    
    # market data
    loc_data = stk.get_data(data_begin=last_date, 
                            data_end=last_date, 
                            market='tw', 
                            symbol=[], 
                            price_change=True,
                            adj=false) 
    
    loc_data = loc_data[['symbol', 'work_date', 'close_change_ratio']]
    # loc_data = loc_data[abs(loc_data['close_change_ratio'])]
    
    
    # combine
    loc_main = loc_data.merge(stock_info, how='inner', on='symbol')
    summary, _ = cbyz.df_summary(df=loc_main, 
                                 group_by=['industry'], 
                                 cols=['close_change_ratio'])

    summary = summary \
            .sort_values(by='close_change_ratio_mean', ascending=false) \
            .reset_index(drop=True)
            
    summary.loc[:, 'last_date'] = last_date
            
    # write
    stk.write_sheet(data=summary, sheet='yd_industry')            
        


# %% master ------


def master(bt_last_begin, predict_period=14, time_unit='d', long=false, 
           interval=360, bt_times=2, data_period=5, ma_values=[5,10,20,60], 
           volume_thld=400, cv=2, compete_mode=1, market='tw', hold=[]):
    '''
    主工作區
    '''

    # v0.0781 - 20220124
    # - add time unit
    # - mvp version for sam weekly prediction, but the calendar in btm broke
    # - before modifing for btm calendar
    # v0.079
    # - remove set_calendar, then get calendar from sam whose backtest_id is 0
    # v0.08
    # - add suffix of time unit for bt_result saved file
    # - combine result of daily prediction and weekly prediction
    # - rename dev mode and test mode
    # v1.000 - 20220214
    # - remove open from var_y
    # - replace year with year_iso, and week with week_iso
    # v1.0100 - 20220216
    # - 當time_unit為w時，讓predict_begin可以不是星期一
    # - rename backtest_manager_v1_001_dev as backtest_manager_v1_0100_dev
    # - add date as suffix of bt_result
    # v1.0200 - 20220305
    # - add time serial of execution as suffix of bt_result to prevent 
    #   duplicated and overwrite
    # - add sam version in the bt_result
    # v1.0300 - 20220305
    # - calculate data_period from 20170101
    
    # v1.0400 - 20220305
    # - update for sam 3.07
    # - add wma to params
    
    # v1.0500
    # - replace time_key from ['year_iso', 'week_num_iso'] to ['year_week_iso'],
    #   then can be apply df_add_ma
    
    # v1.0501
    # - remvoe local cbyz
    
    # v1.0502 - 20220526
    # convert columns to lowercase    


    msg = ('test為True的時候，gdp_mean_ma_36的unique value太少，會導致df_scaler_v2'
           '出錯，最後維持原始數值')
    print(msg)


    # bug
    # - the best buying price is prediction - rmse
    # - actions會有大量重複，總筆數11465筆，但直接drop_duplicates()，沒設
    # 任何subset後，就剩961筆
    # - bug, test=True時，所有industry_high_ma_1_mean的數值都一樣
    # - 當load_data為True時，會出現錯誤，應該是讀取csv時，symbol被當成int
    # /tmp/ipykernel_5939/1527144950.py in set_frame()
    #     224 
    #     225     actions_main = frame \
    # --> 226         .merge(actions_main, how='left', on=['symbol'] + time_key_last) \
    #     227         .merge(bt_result, how='left', on=id_keys)
        
    # valueerror: you are trying to merge on int64 and object columns. if you wish to proceed you should use pd.concat
    

    # trading bot
    # 是不是能在btm中測停損不停利策略
    # 寫一個stk_simulate()，目的是讓回測的數值可以算得出損益
    # stop_loss strategy該怎麼寫成function
    # - 停損策略應該看獲利百分比，還是看support and resistance
    
    
    # optimization ......
    # - weekly forecast的y如果是price，就應該設為high of high, low of low
    # 4. calculate irr, remove outliers
    # 5. google sheet add manual tick
    # 6. think how to optimize stop loss
    # 7. backtest也可以用parameter做a/b        
    # add signal type: 接下來全部漲，或是先跌再漲
    # 1. 當time_unit為w時，目前的日期一定要用星期一，否則會出錯
    # - 加入三大法人買賣明細後，df_summary需要加上sum，代表總賣張數，所以不一定完全
    #   套用global summary params
    # - 大事紀
    #   烏克蘭戰爭，20220224傳出48小時內開戰
    # - 在weekly forecast action中增加日期，不然只用week_num有點難判斷    


    # worklist ......
    # 0. call notification function in backtest manager, and add stop loss 
    #    in excel
    # 1. dev為True時就不匯出模型，避免檔案亂掉
    # 1. add date index to indentify feature of time series
    #    > add this to sam v7 and v8
    # 2. add diff_mape_median
    # 3. 除權息會拉抬n天的服價，把n用weight的方式考慮進去
    # 7. add machine learning error function
    # 8. 之前的code是不是沒有把股本正確normalize / add eps
    # 12. set the result of long-term ma as a table
    # 13. update, optimize capitial_level with kmeans
    # 18. when backtest, check if the symbol reached the target price in the 
    #     next n weeks.
    # 19. analysis, 近60日投報率，用產業或主題作分群 / 財報
    # 20. add sell signal
    # 21. 產業上中下游關係，sna
    # 24. 把股性分群；隱含波動率
    # 27. signal a and b，a是反彈的，b是low和close差距n%
    # global parama 應該是在一開始就訂好，而不是最後才收集，參考rtml
    # - drl資產配置
    #   http://nccur.lib.nccu.edu.tw/bitstream/140.119/137167/1/100701.pdf
    # - 賽局理論
    #   https://luckylong.pixnet.net/blog/post/61539988
    #   證券市場主力與散戶投資人跟從行為之研究 - ─一個簡單的賽局模型
    #   http://www.bm.nsysu.edu.tw/tutorial/vwliu/publish/journal/herd%20behavior.pdf

    # - entity resolution
    #   不同的資料顆粒度導致same value issues 不確定有沒有誤解名詞的意思
    #   an introduction to entity resolution — needs and challenges
    #   https://towardsdatascience.com/an-introduction-to-entity-resolution-needs-and-challenges-97fba052dde5
    # 
    #   erblox: combining matching dependencies with machine learning for entity resolution
    #   https://www.sciencedirect.com/science/article/pii/s0888613x17300439
    # 
    #   how to perform data cleaning for machine learning with python
    #   https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/


    # 費城半導體指數	
    # - 外資券商不等於外資 https://www.thenewslens.com/article/139790



    # # not collected parameters ......
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
    # load_model_data = false
    
    # # collected parameters ......
    # bt_last_begin = 20220215
    # predict_period = 2
    # data_period = int(365 * 3.5)
    # ma_values = [6,10,20]
    # volume_thld = 500
    # hold = [8105, 2610, 3051, 1904, 2611]
    # long = false
    # compete_mode = 0
    # cv = list(range(2, 3))
    # load_result = false
    # dev = True
    # predict_begin = bt_last_begin
    
    
    # keys ------
    global id_keys, time_key, time_key_last
    
    if time_unit == 'w':
        id_keys = ['symbol', 'year_week_iso']
        time_key = ['year_week_iso']
        
        # id_keys = ['symbol', 'year_iso', 'week_num_iso']
        # time_key = ['year_iso', 'week_num_iso']
        
    elif time_unit == 'd':
        id_keys = ['symbol', 'work_date']    
        time_key = ['work_date']    

    time_key_last = [y + '_last' for y in time_key]

    
    # worklist
    # 1.add price increse but model didn't catch
    global _interval, _bt_times, _volume_thld, _ma_values, _hold
    global symbol, _market
    global _bt_last_begin, _bt_last_end
    global serial
    global load_model_data

    _hold = [str(i) for i in hold]
    serial = cbyz.get_time_serial(with_time=True)


    # parameters ......
    
    # 0 for original, 1 for ma, 2 for shifted time series
    data_form = 1

    # data forme issues
    # - 沒辦法像machinlearningmastery的範例，把全部的資料當成
    #   time series處理，因為我用了很多的資料集，像是三大法人持股成本和covid-19，代表這些
    #   欄位全用都需要用time series的方式處理，才有辦法預測第2-n天    
    #   https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
    # - 可以把predict_period全部設為1，但調整time_unit
    if data_form == 2:
        predict_period = 1


    # wait for update
    # date_manager = cbyz.date_manager(predict_begin=predict_begin, 
    #                                  predict_period=_predict_period,
    #                                  data_period=_data_period,
    #                                  data_period_unit='d',
    #                                  data_period_agg_unit='d',
    #                                  predict_period_unit='d',
    #                                  predict_by_time=True,
    #                                  merge_period=[], shift=data_shift, 
    #                                  week_begin=0)
    
    # date_df = date_manager.table
    # shift_begin = date_df.loc[0, 'shift_begin']
    # shift_end = date_df.loc[0, 'shift_end']
    # data_begin = date_df.loc[0, 'data_begin']
    # data_end = date_df.loc[0, 'data_end']
    # predict_begin = date_df.loc[0, 'predict_begin']
    # predict_end = date_df.loc[0, 'predict_end']  
    # calendar = date_manager.calendar_lite    


    if dev or test:
        
        # stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)        
        
        #
        # 2417, 3092, 2014, 3189, 1609, 2884
        symbol = [2520, 2605, 6116, 6191, 3481, 
                   2409, 2520, 2603, 2409, 2603, 
                   2611, 3051, 3562, 2301, 
                   '2211', '3138', '3530', 3041, 3596]
        
        symbol = [str(s) for s in symbol]
        symbol = cbyz.li_drop_duplicates(symbol)
        
    
    else:
        symbol = []
        
        
    global _compete_mode, _time_unit
    _market = market    
    _compete_mode = compete_mode
    _time_unit = time_unit


    # arguments
    # 1. 目前的parameter都只有一個，但如果有多個parameter需要做a/b test時，應該要在
    #    在btm中extract，再把single parameter的param_holder傳入sam，因為
    #    param_holder的參數會影響到model_data，沒辦法同一份model_data持續使用，因此，
    #    把完整的param_holder傳入sam再extract沒有任何效益
    
    args = {'bt_last_begin':[bt_last_begin],
            'time_unit':[time_unit],
            'predict_period': [predict_period], 
            'data_period':[data_period],
            'data_form':[data_form],
            'load_model_data':[load_model_data],
            'ma_values':[ma_values],
            'volume_thld':[volume_thld],
            'industry':[True],
            'trade_value':[True],
            'wma':[True],
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
    param_holder = ar.param_holder(**args)
    
    
    # select parameters
    params = param_holder.params
    keys = list(params.keys())
    values = list(params.values())

    param_df = pd.dataframe()
    for i in range(0, len(params)):
        
        values_li = values[i]
        values_li = cbyz.conv_to_list(values_li)
        new_df = pd.dataframe({keys[i]:values_li})
        
        if i == 0:
            param_df = param_df.append(new_df)
        else:
            param_df = cbyz.df_cross_join(param_df, new_df)


    _interval = interval
    _bt_times = bt_times
    _volume_thld = volume_thld
    _ma_values = ma_values
        

    # rename predict period as forecast preiod
    symbol = cbyz.li_conv_ele_type(symbol, to_type='str')


    # set date ......
    global _bt_last_begin, _predict_period
    global calendar, _bt_last_end
    _predict_period = predict_period
    _bt_last_begin = bt_last_begin
    

    # predict ------
    global bt_result, precision, features
    backtest_predict(bt_last_begin=bt_last_begin, 
                     predict_period=_predict_period, 
                     interval=interval,
                     data_period=data_period)
    
    
    # set calendar ------
    set_calendar()


    # set date ......
    set_frame()
    
    # debug for prices columns issues
    # if len(bt_result) > 300:
    #     stk.write_sheet(data=bt_result, sheet='debug')
    
    
    # profit ------    
    # y_thld=0.02
    # time_thld=predict_period
    # prec_thld=0.05
    # export_file=True
    # load_file=True
    # path=path_temp
    # file_name=none        
    

    global bt_main, actions
    
    # optimize, 這裡的precision_thld實際上是mape, 改成precision
    global mape, mape_group, mape_extreme
    global stock_metrics_raw, stock_metrics    
    
    today = cbyz.date_get_today()
    execute_begin = cbyz.date_cal(today, -14, 'd')
    execute_begin = int(str(execute_begin) + '0000')
    
    
    # evaluate precision ......
    print('bug - get_forecast_records中的action score根本沒用到，但可以用signal替代')
    cal_profit(y_thld=0.02, time_thld=_predict_period, prec_thld=0.05,
               execute_begin=execute_begin, 
               export_file=True, load_file=True, path=path_temp,
               file_name=none)
    
    eval_metrics()
    
    
    # write google sheets ...... 
    if len(actions) > 300:
        
        print('bug - actions has duplicated rows when load_data = True,'
              ' or it may be caused by stk.write_sheet')
        
        actions = actions.drop_duplicates().reset_index(drop=True)
        
        # action workbook
        if time_unit == 'd':
            stk.write_sheet(data=actions, sheet='day', long=long,
                            predict_begin=_bt_last_begin)
        elif time_unit == 'w':
            stk.write_sheet(data=actions, sheet='week', long=long,
                            predict_begin=_bt_last_begin)            
    
        # view and log .....
        print(('view_yesterday暫時移除，time_unit為w時，actions中沒有last_date，'
               '所以會出錯'))
        # view_yesterday()

        
        # error when load_result = True
        global pred_features
        try:
            stk.write_sheet(data=pred_features, sheet='features')
        except exception as e:
            print(e)
            
    
    gc.collect()

    return actions


def master_history():

    # v0.0 - first version
    # v0.1
    # - 拿掉5ma之後的精準度有提升
    # v0.2
    # - add buy_signal
    # v0.05
    # - change local variable as host
    # v0.06
    # - update for cbyz and cbml
    # - include low volume symbol in the excel
    # v0.062
    # - add close level to buy_signal column  
    # v0.064
    # export to specific sheet
    # 這個版本有bug，不改了，直接跳v0_07    
    # v0.07
    # - update for ultra_tuner
    # - add write_sheet
    # - 列出前一天大漲的產業
    # - 增加一個欄位，標示第二天為負，為了和day_trading配合快速篩選
    # v0.071
    # - hold symbol沒有順利標示 - done
    # - cal_profit中的last不要用還原股價，不然看的時候會很麻煩 - done
    # - view yesterday，還沒處理日期的問題 - done
    # v0.072
    # - remove fast and load_model parameter - done
    # - 把high和short的model分開，避免覆寫模型 - done
    # - 當load_model為false，且bt_times > 1時，只有第一次會retrain，第二次會load - done
    # v0.073
    # - the features sheet of the actions will be overwrited by both 
    #    long and short. - done
    # v0.074
    # - fix last_price issues - done, wait for checking 
    # - optimize cal_profit - done
    # - update eval_metrics and rebuild forecast_records
    # v0.075
    # - fix bug
    # v0.076
    # - add capability to read saved bt_result.csv > done
    # - add date to result of view_yesterday > done
    # v0.077 - 20220118
    # - rename symbols as symbol
    # - 是不是還是必須把bt mape和score合併
    # - 權殖股，依照產業列出rank
    # - view_yesterday and view_industry is the same
    # v0.078 - 20220119
    # - add form and ml_df_to_time_series in sam    
    pass


# %% verify ......
def verify_prediction_results():

    
    # diary
    # 0627 - 預測的標的中，實際漲跌有正有負，但負的最多掉2%。

    
    end = 20210625
    begin = cbyz.date_cal(end, -1, 'd')
    
    
    # files ......
    file_raw = pd.read_excel(path_export + '/actions_20210624_233111.xlsx')
    
    file = cbyz.df_conv_col_type(df=file_raw, cols='stock_symbol', to='str')
    file = file_raw[file_raw['work_date']==end]
    file = file[['stock_symbol', 'close', 'high', 'low']]
    file.columns = ['stock_symbol', 'close_predict', 
                    'high_predict', 'low_predict']
    
    
    symbol = file_raw['stock_symbol'].unique().tolist()
    
    
    # market data ......
    data_raw = stk.get_data(data_begin=begin, 
                            data_end=end, 
                            market='tw', 
                            symbol=[], 
                            price_change=True,
                            ratio_limit=True, 
                            price_limit=false, 
                            trade_value=false)

    data = data_raw[(data_raw['work_date']==20210625) \
                & (data_raw['stock_symbol'].isin(symbol))] \
            .sort_values(by='price_change_ratio', ascending=false) \
            .reset_index(drop=True)

    
    main_data = data.merge(file, how='left', on='stock_symbol')
    

    
# %% gcp ------


def export_gcp_data():
    
    # too few rows: 179. minimum number is: 1000
    
    model_data = sam.model_data.copy()
    
    model_data = model_data.dropna(subset=['close_change_ratio'], axis=0)
    cbyz.df_chk_col_na(df=model_data)
    
    model_data = model_data.drop(['high_change_ratio', 'low_change_ratio'],
                                 axis=1)
    
    model_data = model_data.drop(id_keys, axis=0)
    
    
    model_data.to_csv(path_export + '/model_data_20220511.csv',
                      index=false)    



# %% dev -----

def cbmbine_action():
    
    global action_weekly, action_daily    
    action_key = ['symbol', 'stock_name', 'industry']
    
    
    cols = cbyz.df_get_cols_except(df=action_daily, except_cols=action_key)
    new_cols = ['day_' + c for c in cols]
    rename_dict = cbyz.li_to_dict(cols, new_cols)
    action_daily = action_daily.rename(columns=rename_dict)
    
    action_final = action_weekly \
        .merge(action_daily, how='outer', on=action_key)
    
    stk.write_sheet(data=action_final, sheet='combine')



def simulate(df):
    '''
    - 如果time_unit為w，且predict_begin不是星期一的話，week_num會shift，這會導致
      simulate的時候不準確；但通常只有回測的時候需要simulate

    parameters
    ----------
    df : bt_result generated by btm
        description.

    returns
    -------
    none.

    '''
    
    print('把market設為stk的global var')
    
    ohlc = stk.get_ohlc()
    ohlc.remove('open')
    ohlc_prev = [c + '_prev' for c in ohlc]
    ohlc_predict = [c + '_predict' for c in ohlc]
    

    # dev ......
    # symbol = [2520, 2605, 6116, 6191, 3481, 
    #           2409, 2520, 2603, 2409, 2603, 
    #           2611, 3051, 3562, 2301, 
    #           '2211', '3138', '3530']
    
    # symbol = [str(i) for i in symbol]

    # df = pd.read_csv(path_temp + '/bt_result_simulate_2.csv')
    # df = df.drop_duplicates().reset_index(drop=True)
    # df = cbyz.df_conv_col_type(df=df, cols='symbol', to='str')
    # df['buy_signal'] = 1
    # df['year_iso'] = 2021


    # setting ......
    loc_df = df.copy()
    symbol = loc_df[['symbol']].drop_duplicates()
    symbol_li = symbol.tolist()

    expand_calendar_raw = cbyz.date_get_prev_calendar(df=loc_df, amount=3)
    frame = cbyz.df_cross_join(symbol, expand_calendar_raw)
    

        
    if 'work_date' in loc_df.columns:
        time_unit = 'd'
        time_key = ['work_date']
        # calendar_key = ['work_date']
    
    elif 'week_num_iso' in loc_df.columns:
        time_unit = 'w'
        time_key = ['year_iso', 'week_num_iso']
        # calendar_key = ['work_date', 'year_iso', 'week_num_iso']
    
    id_key = ['symbol'] + time_key
    df_time_begin = loc_df[time_key]
    df_time_begin = df_time_begin[df_time_begin.index==0]
    df_time_begin['begin'] = 1
    
    # get hist data
    hist_begin = expand_calendar_raw['work_date'].min()
    hist_begin = cbyz.date_cal(hist_begin, -20, 'd')
    end_date = cbyz.date_get_today()
  
    hist_data = stk.get_data(data_begin=hist_begin, 
                             data_end=end_date, 
                             market=market, 
                             symbol=symbol_li, 
                             price_change=false,
                             adj=false)
    
    hist_data = hist_data[['symbol', 'work_date'] + ohlc]
    
    last_price, _ = \
        cbyz.df_add_shift(df=hist_data, cols=ohlc, shift=-1, 
                          sort_by=['symbol', 'work_date'], 
                          group_by=[], suffix='_prev', 
                          remove_na=false)
    
    # calculate with high of high, and low of low
    if time_unit == 'd':
        print('time_unit = d還沒寫')
        
        
    elif time_unit == 'w':
        
        future_price = expand_calendar_raw \
            .merge(hist_data, how='left', on='work_date')
            
        future_price = future_price.dropna(axis=0)
        
        print('simulate by week的cloes是用mean agg')
        future_price = future_price \
                    .groupby(id_key) \
                    .agg({'high':'max',
                          'low':'min',
                          'close':'mean'}) \
                    .reset_index()
                    
    
    # frame
    loc_main = frame \
            .merge(last_price, how='left', on=['symbol', 'work_date'])
            
    loc_main = loc_main \
                .dropna(axis=0) \
                .sort_values(by=['symbol', 'work_date'], ascending=false) \
                .drop_duplicates(subset=id_key)
                    
    loc_main = loc_main[id_key + ohlc_prev]
    loc_main = loc_df.merge(loc_main, how='left', on=id_key)

    
    # calculate ohlc price by change ratio
    for i in range(len(ohlc)):
        col = ohlc[i]
        
        if col + '_change_ratio' in loc_main.columns:
            loc_main[col + '_predict'] = loc_main[col + '_prev'] \
                                        * (1 + loc_main[col + '_change_ratio'])
    
    loc_main = loc_main[id_key + ohlc_predict]
    
    
    # simulate calendar
    simulate_calendar = expand_calendar_raw[time_key] \
                        .drop_duplicates() \
                        .reset_index(drop=True)
    
    simulate_calendar = simulate_calendar \
        .merge(df_time_begin, how='left', on=time_key)
    
    mark_index = simulate_calendar.dropna(subset=['begin'], axis=0)
    mark_index = mark_index.index.min()
    
    simulate_calendar = \
            simulate_calendar[simulate_calendar.index >= mark_index] \
            .reset_index(drop=True) \
            .drop('begin', axis=1)
    
    # dev
    # loc_main = loc_main.dropna(axis=0)
    
    # simulate ......
    highest = pd.dataframe()
    trans_raw = pd.dataframe()
    cost = pd.dataframe()
    sell_symbol = pd.dataframe()
    
    stop_loss_pos = 0.8
    stop_loss_neg = 0.95
    
    for i in range(len(simulate_calendar)):
        
        new_trans = pd.dataframe()
        new_sell_symbol = pd.dataframe()

        cur_calendar = simulate_calendar[simulate_calendar.index==i]
        cur_data = cur_calendar.merge(future_price, how='left', on=time_key)
        
        # buy
        if i  == 0:
            cur_data = cur_data.merge(loc_main, on=id_key)
            cur_data = cur_data[cur_data['low_predict']>=cur_data['low']]
            cur_data['buy'] = 1

            # highest
            # 用這個算法的話，沒辦法判斷是先漲後跌，還是先跌後漲，所以可能會有誤差
            highest = cur_data[['symbol', 'high']] \
                        .rename(columns={'high':'highest'})      
                    
            # transaction
            new_trans = cur_data[id_key + ['low_predict']] \
                    .rename(columns={'low_predict':'price'})    
            new_trans['type'] = 'buy'
            
            # cost
            # - 把這個df獨立出來，這樣後面在計算stop_loss的時候比較方便merge
            print('add tax and trans fee')
            cost = new_trans[['symbol', 'price']] \
                .rename(columns={'price':'cost'})
            
        else:
            stop_loss = cost.merge(highest, how='left', on='symbol')
            stop_loss['stop_loss_pos'] = stop_loss['cost'] \
                + 0.8 * (stop_loss['highest'] - stop_loss['cost'])
                
            stop_loss['stop_loss_neg'] = stop_loss['cost'] * stop_loss_neg
                
            
            # transaction ......
            new_trans = cur_data.merge(stop_loss, how='left', on='symbol') 
            new_trans = cbyz.df_anti_merge(new_trans, sell_symbol,
                                           on='symbol')
            
            # 判斷是獲利了解，還是認賠殺出
            new_trans = \
                new_trans[(new_trans['low']<=new_trans['stop_loss_pos']) \
                          | (new_trans['low']<=new_trans['stop_loss_neg'])]
                    
            new_trans['price'] = \
                np.where(new_trans['low']<=new_trans['stop_loss_neg'],
                         new_trans['stop_loss_neg'], 
                         new_trans['stop_loss_pos'])
            
            new_trans['type'] = 'sell'
            
                
            # sell symbol ......
            if len(new_trans) > 0:
                new_sell_symbol = new_trans[['symbol']]
                sell_symbol = sell_symbol.append(new_sell_symbol)


            # update highest ......
            new_highest = cur_data[['symbol', 'high']] \
                        .rename(columns={'high':'highest'})
                    
            highest = highest.append(new_highest)              
            highest = highest \
                .sort_values(by=['symbol', 'highest'], ascending=false) \
                .reset_index(drop=True)
                    
        if len(new_trans) > 0:
            new_trans = new_trans[id_key + ['price', 'type']]
            trans_raw = trans_raw.append(new_trans)

    
    trans_raw = trans_raw.reset_index(drop=True)
        
   
    # test
    chk_sell = sell_symbol[['symbol']].drop_duplicates()
    assert len(sell_symbol) == len(chk_sell), \
        'the length of sell symbol should be the same'



# %% debug ------
        
def debug():

    sam.predict_df
    sam.main_data    
    
    pass


def note():

    # sam note ------
    # 1. 20220107 v2.06 - 原本在normalize的時候，會group by symbol，讓每一檔都
    #    和自己比較，否則高價股和低價股比感覺很虧。這個版本試著把sam_load_data中的
    #    group by改成[]。經測試過後，r2差不多，所以保留新的版本，應該可以提高計算
    #    速度。
    # 2. y為price時會一直overfitting
    # 3. 當test為True，time_unit為w時，train_precision約0.0004，test_precision
    #    為0.043743，但正式執行時，test_precision也大約是0.04-0.05，是不是代表
    #    model一直在overfitting
    # 4. 20220422 chat with grace。當市場恐慌時，像中華電信這種平常不太會動的股票就
    #    有機會可以破新高，因為有些大量的錢沒地方流，像是基金，所以就會轉往看起來相對安全
    #    的標的。要如何在model中處理類似的情況？    
    # 5. 修改df_scaler的時候，發現確實需要scaled by symbol，否則全部資料全部丟進去	
    #    scaled，normaltest的p valueh前是0
    # 6. scaling process: summary > ma > scaler > merge frame > fillna

    

    # btm note
    # 1. 如果用change_ratio當成y的話，對模型來說，最安全的選項是不是設為0？
    # 2. 當price為y時，industry的importance超高，可能是造成overfitting的主因
    # 3. 以industry作為target_type時，一定要用change_ratio當y
    # 4. data_form為1，y為change_ratio時
    # - ma為[5,10,20,60,120]，open和low較佳；ma為[5,10,20,60]，high和close較佳，
    #   但兩者mse差異只在0.01等級
    # 5. time_unit為week，且data_form為1時，兩者都會讓曲線變得更加平滑，因此可能
    #    低估振幅，但強制縮放成-0.1 - 0.1也不妥，如果整體盤勢向下，這個方法會
    #    高估走勢    
    pass    



# %% execute ------

if __name__ == '__main__':
    

    # change ratio by week

    # randomforest ......
    # n_estimatorsint
    # max_depth 6 / 4, 8, 10
    
    # xgboost ......
    # n_estimators: 200 / 100, 300
    # eta: 0.2 / 0.1, 0.3
    # max_depth: 8 / 4, 6, 10
    # min_child_weight: 1
    # subsample: 1
    # cv: 2 / 3, 4
    
    # bug, 只有high的param log有寫入，low和close都沒有

        
    global weekly_actions, daily_actions
    hold = [3596, 6698]
    
    global dev, test, load_result, load_model_data
    dev = True
    dev = false
    test = True
    # test = false
    load_result = True    
    load_result = false
    
    load_model_data = True
    load_model_data = false
    
    global action_weekly, action_daily
    
    # test mode take few data to run, and dev mode will decrease the threshold
    # to export temp file
    
    if not dev and not test:
        # - tej的資料從2017年開始，但用dev的測試結果，即使data_period從20150110
        #   開始算也不會出錯
        # - tej報價
        #   集保股權分散的資料，個人使用9000/年，歷史資料5400/年
        today = cbyz.date_get_today()
        data_period = cbyz.date_diff(today, 20170101, absolute=True)
    else:
        data_period = 365 * 5
        # data_period = 365 * 1 # test shareholding_spread
    
    
    # week
    # - ma 48會超級久，連dev mode都很久
    # - ma max 為24時，drop corr後的欄位數量為530
    action_weekly = \
        master(bt_last_begin=20220523, predict_period=1, 
            time_unit='w', long=false, interval=4, bt_times=1, 
            data_period=data_period,
            ma_values=[1,4,12,24,36], volume_thld=400,
            compete_mode=0, cv=list(range(3, 4)),
            market='tw', hold=hold)
        
    # stk.write_sheet(data=actions, sheet='week')
    
    
    # day
    # action_daily = \
    #     master(bt_last_begin=20220216, predict_period=1, 
    #             time_unit='d', long=false, interval=4, bt_times=1, 
    #             data_period=data_period, 
    #             ma_values=[5,10,20,60], volume_thld=400,
    #             compete_mode=0, cv=list(range(3, 4)),
    #             market='tw', hold=hold)
        
        
        
