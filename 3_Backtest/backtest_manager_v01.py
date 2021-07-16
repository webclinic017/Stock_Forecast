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


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/3_Backtest'
    path_sam = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/2_Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/3_Backtest'
    path_sam = '/home/aronhack/stock_forecast/2_Stock_Analysis'    

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
# import stock_analysis_manager_v05 as sam
# import stock_analysis_manager_v06 as sam
import stock_analysis_manager_v07 as sam



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
                                       stock_type=stock_type, 
                                       local=local)
    
    calendar['TRADE'] = np.where(calendar['TRADE_DATE']==0, 0, 1)
    
    
    calendar, _ = cbyz.df_add_shift(df=calendar, cols='WORK_DATE',
                                    shift=predict_period - 1, 
                                    group_by=['TRADE'], 
                                    suffix='_LAST', remove_na=False)
    
    bt_last_end = calendar[calendar['WORK_DATE_LAST']==bt_last_begin]
    bt_last_end = int(bt_last_end['WORK_DATE'])    



# ..........
    

def backtest_predict(bt_last_begin, predict_period, interval, 
                     bt_times, data_period):
    
    
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
    global bt_results, rmse, features, model_y, volume_thld
    
    bt_results_raw = pd.DataFrame()
    rmse = pd.DataFrame()
    features = pd.DataFrame()
    
    # Predict ......
    for i in range(0, len(bt_seq)):
        
        begin = bt_seq[i]

        results_raw = sam.master(_predict_begin=begin,
                                 _predict_end=None, 
                                 _predict_period=predict_period,
                                 _data_period=data_period, 
                                 _stock_symbol=stock_symbol,
                                 ma_values=ma_values,
                                 _volume_thld=volume_thld)


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



def cal_profit(y_thld=2, time_thld=10, rmse_thld=0.15, execute_begin=None,
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
    hist_cols = [i + '_HIST' for i in model_y]
    hist_cols_dict = cbyz.li_to_dict(model_y, hist_cols)
    
    last_cols = [i + '_LAST' for i in model_y]


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
    # loc_begin = 20180101
    hist_data_raw = stk.get_data(data_begin=bt_first_begin, 
                                 data_end=bt_last_end, 
                                 stock_type=stock_type, 
                                 stock_symbol=stock_symbol, 
                                 price_change=True,
                                 local=local)
    
    hist_data_raw = hist_data_raw[['WORK_DATE', 'STOCK_SYMBOL'] + model_y]
    
 
    
    # hist_data = hist_data_raw.rename(columns={'WORK_DATE':'LAST_DATE'})
    # hist_data_raw = hist_data_raw.rename(columns={'WORK_DATE':'LAST_DATE'})


    # calendar = hist_data_raw[['WORK_DATE']] \
    #             .append(forecast_calendar) \
    #             .drop_duplicates()
    
    # Get Last Date ....        
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
    main_data_pre, _ = cbyz.df_add_shift(df=main_data_pre, cols=model_y, 
                                          shift=1, group_by=['STOCK_SYMBOL'], 
                                          suffix='_LAST', remove_na=False)

    main_data_pre = main_data_pre.rename(columns=hist_cols_dict)


    # Organize ......
    main_data = bt_results.merge(main_data_pre, how='left', 
                                 on=['WORK_DATE', 'STOCK_SYMBOL'])


    main_data = main_data[['BACKTEST_ID', 'STOCK_SYMBOL', 'MODEL', 
                           'WORK_DATE', 'LAST_DATE'] \
                          + model_y + hist_cols + last_cols]


    # 把LAST全部補上最後一個交易日的資料
    # 因為迴測的時間有可能是假日，所以這裡的LAST可能會有NA
    main_data = cbyz.df_shift_fill_na(df=main_data, 
                                      loop_times=predict_period+1, 
                                      cols=last_cols, 
                                      group_by=['STOCK_SYMBOL', 'BACKTEST_ID'])

    # Fill na in the forecast period.
    # for i in hist_cols:
    #     main_data[i] = np.where(main_data['WORK_DATE'].isin(forecast_calendar),
    #                             np.nan, main_data[i])

    # Check na ......
    # 這裡有na是合理的，因為hist可能都是na
    chk = cbyz.df_chk_col_na(df=main_data, positive_only=True)
    main_data = main_data.dropna(subset=last_cols)
    
    if len(chk) > len(model_y):
        print('Err01. cal_profit - main_data has na in columns.')
        

    # Generate Actions ......
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
        
    # Evaluate Precision ......
    eval_metrics(export_file=False, upload=upload_metrics)            
    
    
    # Forecast Records ......
    # records_begin = cbyz.date_cal(bt_last_begin, -2, 'm')
    print('暫時移除records_begin')
    records = stk.get_forecast_records(forecast_begin=None, 
                                        forecast_end=None, 
                                        execute_begin=execute_begin, 
                                        execute_end=None, 
                                        y=['CLOSE'], summary=True,
                                        local=local)
    
    if len(records)  > 0:
        records = records \
            .rename(columns={'FORECAST_PRECISION_MEAN':'RECORD_PRECISION_MEAN',
                             'FORECAST_PRECISION_MAX':'RECORD_PRECISION_MAX'})
        
    # Tick
    actions.loc[actions.index, 'TICK'] = np.nan
    
    # Hold 
    global hold
    hold = [str(i) for i in hold]
    actions['HOLD'] = np.where(actions['STOCK_SYMBOL'].isin(hold), 1, 0)

    
    # Rearrange Columns ......            
    if 'CLOSE' in model_y:
        profit_cols = ['CLOSE_PROFIT_PREDICT', 
                       'CLOSE_PROFIT_RATIO_PREDICT']
        
    profit_cols = profit_cols \
        + ['RECORD_PRECISION_MEAN', 'RECORD_PRECISION_MAX', 
           'DIFF_MEAN', 'DIFF_MAX']
        
    
    cols_1 = ['BACKTEST_ID', 'STOCK_SYMBOL', 'STOCK_NAME', 
              'TICK', 'HOLD', 'WORK_DATE', 'LAST_DATE']

    model_y_last = [s + '_LAST' for s in model_y]
    model_y_hist = [s + '_HIST' for s in model_y]    
    cols_2 = ['RMSE_MEAN']
    
    new_cols = cols_1 + profit_cols + model_y + model_y_last \
                + model_y_hist + cols_2


    # Merge Data ......
    if len(records) > 0:
        actions = actions.merge(records, how='left', on=['STOCK_SYMBOL'])
    
        actions.loc[:, 'DIFF_MEAN'] = actions['CLOSE_PROFIT_RATIO_PREDICT'] \
            - actions['RECORD_PRECISION_MEAN']
    
        actions.loc[:, 'DIFF_MAX'] = actions['CLOSE_PROFIT_RATIO_PREDICT'] \
            - actions['RECORD_PRECISION_MAX']
    
        actions = actions[new_cols]
    


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
    
    
    for y in model_y:
        
        mape_main.loc[:, 'MAPE'] = (mape_main[y] \
                                - mape_main[y + '_HIST']) \
                            / mape_main[y + '_HIST']
                            
        mape_main.loc[:, 'OVERESTIMATE'] = \
            np.where(mape_main[y] > mape_main[y + '_HIST'], 1, 0)

        # Absolute ......
        mape_main_abs = mape_main.copy()
        mape_main_abs.loc[mape_main_abs.index, 'MAPE'] = abs(mape_main_abs['MAPE'])
                 
        
        # MAPE Overview
        new_mape = cbyz.summary(df=mape_main_abs, group_by=[], cols='MAPE')
        new_mape.loc[:, 'Y'] = y
        mape = mape.append(new_mape)
        
        
        # Group MAPE ......
        new_mape = cbyz.summary(df=mape_main_abs, group_by='OVERESTIMATE', 
                                 cols='MAPE')
        new_mape.loc[:, 'Y'] = y
        mape_group = mape_group.append(new_mape)
        
        
        # Extreme MAPE ......
        new_mape = mape_main_abs[mape_main_abs['MAPE'] > 0.1]
        new_mape = cbyz.summary(df=new_mape, 
                                group_by=['BACKTEST_ID', 'OVERESTIMATE'], 
                                cols='MAPE')
        
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
                        .merge(rmse, how='left', on=['Y', 'BACKTEST_ID'])

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
                                         'RMSE':'MODEL_PRECISION',
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
        ar.db_upload(data=stock_metrics, table_name='forecast_records', 
                     local=local)


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
    
    # Backtest也可以用parameter做A/B    
    # Bug, symbol var不需要算ma
    
    
    # Excel add tick column
    
    # Bug
    # print('backtest_predict - 這裡有bug，應該用global calendar')
    # add hold columns
    # excel format
    
    
    # Bug
    # 1.Excel中Last Priced地方不應該一直copy最後一筆資料

    
    
    # Parameters
    # 把要預測的時間放在第三天
    _bt_last_begin = 20210715
    # _bt_last_begin = 20210707
    predict_period = 5
    # interval = random.randrange(90, 180)
    _interval = 4
    _bt_times = 3
    data_period = int(365 * 1.7)
    # data_period = int(365 * 0.86) # Shareholding    
    # data_period = 365 * 2
    # data_period = 365 * 5
    # data_period = 365 * 7
    _stock_symbol = [2520, 2605, 6116, 6191, 3481, 2409, 2603]
    _stock_symbol = []
    _stock_type = 'tw'
    # _ma_values = [5,10,20]
    _ma_values = [5,10,20,40]
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
    global bt_results, rmse, features
    backtest_predict(bt_last_begin=bt_last_begin, 
                     predict_period=predict_period, 
                     interval=interval,
                     bt_times=bt_times,
                     data_period=data_period)

    
    # Profit ------    
    # Update, 需把Y為close和Y為price_change的情況分開處理
    # Update, stock_info每日比對檔案自動匯
    
    # y_thld=-100
    # time_thld=predict_period
    # rmse_thld=5
    # export_file=True
    # load_file=True
    # path=None
    # file_name=None        
    

    global bt_main, actions
    
    # 算回測precision的時候，可以低估，但不可以高估
    global mape, mape_group, mape_extreme
    global stock_metrics_raw, stock_metrics    
    
    global hold
    hold = [1474, 1718, 2002, 2504, 3576, 5521, 8105, 1809]    
    
    
    cal_profit(y_thld=-100, time_thld=predict_period, rmse_thld=5,
               execute_begin=2107120000,
               export_file=True, load_file=True, path=path_temp,
               file_name=None, upload_metrics=False) 
    
    
    # actions = actions[actions['MODEL']=='model_6']
    # actions = cbyz.df_add_size(df=actions, group_by='STOCK_SYMBOL',
    #                            col_name='ROWS')


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

    cbyz.excel_add_format(sht=sht, cell_format=percent_format, 
                          startrow=1, endrow=9999,
                          startcol=8, endcol=12)

    cbyz.excel_add_format(sht=sht, cell_format=digi_format, 
                          startrow=1, endrow=9999,
                          startcol=7, endcol=7)    
    
    cbyz.excel_add_format(sht=sht, cell_format=digi_format, 
                          startrow=1, endrow=9999,
                          startcol=13, endcol=16)    
    writer.save()


    
    # Full Data
    # full_data = sam.sam_load_data(data_begin=None, data_end=None, 
    #                               stock_type=stock_type, period=None, 
    #                               stock_symbol=stock_symbol, 
    #                               lite=False, full=True)


    # full_data = full_data['FULL_DATA']    

    
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
    
    ar.db_execute(sql, local=local, fetch=False)



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
