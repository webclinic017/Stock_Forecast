#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:08 2020

@author: Aron
"""

# Worklist
# 1. Some models only work for some stock symbol.
# > Clustering
# 2. Add crypto


# Optimization
# 1. 如何在回測的時候，不要分多次讀取歷史資料，而可以一次讀取完？



# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV    
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
import xgboost as xgb



host = 0


# Path .....
if host == 0:
    path = '/Users/Aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis'
elif host == 2:
    path = '/home/jupyter//2_Stock_Analysis'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20211212',
                 r'/home/jupyter/Arsenal/20211212',
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import codebase_ml as cbml
import arsenal as ar
import arsenal_stock as stk
import ultra_tuner_v0_16_dev as ut

ar.host = host



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])        


# %% Inner Function ------


def get_market_data_raw(industry=True, trade_value=True, support_resist=False):
    
    
    global symbols, market
    global market_data_raw
    global predict_period
    global stock_info_raw
    global log
    

    stock_info_raw = \
        stk.tw_get_stock_info(daily_backup=True, 
                              path=path_temp)
        
    stock_info_raw = \
        stock_info_raw[['SYMBOL', 'CAPITAL', 
                        'CAPITAL_LEVEL', 'ESTABLISH_DAYS', 
                        'LISTING_DAYS', 'INDUSTRY_ONE_HOT']]    


    # Market Data ...
    # Shift one day forward to get complete PRICE_CHANGE_RATIO
    loc_begin = cbyz.date_cal(shift_begin, -1, 'd')    
    
    
    if len(symbols) == 0:
        market_data_raw = \
            stk.get_data(
                data_begin=loc_begin, 
                data_end=data_end, 
                market=market, 
                symbol=[],
                adj=True,
                price_change=True, 
                price_limit=True, 
                trade_value=trade_value
                )
    else:
        market_data_raw = \
            stk.get_data(
                data_begin=loc_begin,
                data_end=data_end, 
                market=market, 
                symbol=symbols,
                adj=True,
                price_change=True, 
                price_limit=True,
                trade_value=trade_value
                )

    # Check        
    global ohlc
    for c in ohlc:
        col = c + '_CHANGE_RATIO'
        min_value = market_data_raw[col].min()
        max_value = market_data_raw[col].max()
        
        # 這裡可能會出現0.7，不確定是不是bug
        # >> 應該是除權息的問題
        msg_min = col + ' min_value is ' + str(min_value)
        msg_max = col + ' max_value is ' + str(max_value)
        
        # assert min_value >= -0.25, msg_min
        # assert max_value <= 0.25, msg_max
        if  min_value >= -0.25:
            print(msg_min)
            
        if  max_value <= 0.25:
            print(msg_max)
        

    # Exclude Low Volume Symbols ......
    market_data_raw = select_symbols()
    

    # Add K line ......
    market_data_raw = market_data_raw \
                    .sort_values(by=['SYMBOL', 'WORK_DATE']) \
                    .reset_index(drop=True)
            
    market_data_raw = stk.add_k_line(market_data_raw)
    market_data_raw = \
        cbml.df_get_dummies(
            df=market_data_raw, 
            cols=['K_LINE_COLOR', 'K_LINE_TYPE']
        )
    
    # Add Support Resistance ......
    
    if support_resist:
        # Check，確認寫法是否正確
        # print('add_support_resistance - days == True時有bug，全部數值一樣，導致
        # 沒辦法標準化？')
        global data_period
        market_data_raw, _ = \
            stk.add_support_resistance(df=market_data_raw, cols='CLOSE',
                                       rank_thld=int(data_period * 2 / 360),
                                       prominence=4, days=False)

    # Predict Symbols ......
    # 1. Prevent some symbols excluded by select_symbols(), but still
    #    exists.
    global symbol_df    
    all_symbols = market_data_raw['SYMBOL'].unique().tolist()
    symbol_df = pd.DataFrame({'SYMBOL':all_symbols})
    
    
    # Calendar
    global calendar
    calendar_proc = calendar[calendar['TRADE_DATE']>0] \
                    .reset_index(drop=True) \
                    .reset_index() \
                    .rename(columns={'index':'DATE_INDEX'})
    
    calendar_proc, _, _ = \
        cbml.ml_data_process(
            df=calendar_proc, 
            ma=False, normalize=True, lag=False, 
            group_by=[],
            cols=[], 
            except_cols=['WORK_DATE', 'TRADE_DATE'],
            cols_mode='equal',
            drop_except=[],
            ma_values=ma_values, 
            lag_period=predict_period
            )
        
    
    # Merge As Main Data
    main_data_frame = cbyz.df_cross_join(symbol_df, calendar_proc)
    main_data_frame = \
        main_data_frame[
            (main_data_frame['TRADE_DATE']>=1) \
            & (main_data_frame['WORK_DATE']<=predict_date[-1])] \
        .drop('TRADE_DATE', axis=1)

    market_data_raw = main_data_frame \
        .merge(market_data_raw, how='left', on=['SYMBOL', 'WORK_DATE'])
    

    # Check ......
    
    # 執行到這裡，因為加入了預測區間，所以會有NA，但所有NA的數量應該要一樣多
    na_cols = cbyz.df_chk_col_na(df=market_data_raw)
    na_min = na_cols['NA_COUNT'].min()
    na_max = na_cols['NA_COUNT'].max()
    
    msg = 'Number of NA in each column should be the same.'
    assert na_min == na_max, msg
    
    
    # Check Predict Period
    chk = market_data_raw[market_data_raw['WORK_DATE'].isin(predict_date)]
    chk = chk[['WORK_DATE']].drop_duplicates()
    assert len(chk) == len(predict_date), 'predict_date error'
    

# ...........


def sam_load_data(industry=True, trade_value=True):
    '''
    讀取資料及重新整理
    '''
    
    global symbols
    global market_data_raw
    global predict_period
    global symbol_df
    global stock_info_raw
    global debug
    
        
    # Process Market Data ......
    loc_main = market_data_raw.drop('TOTAL_TRADE_VALUE', axis=1)
    
    
    # Normalize Ratio Columns Globally ......
    # 1. 如果y是OHLC Change Ratio的話，by WORK_DATE或SYMBOL的意義不大，反而讓運算
    #    速度變慢，唯一有影響的是一些從來沒有漲跌或跌停的Symbol
    # 2. 如果y是OHLC的價格的話，這一段的normalize邏輯有點怪
    msg = '如果y是OHLC的價格的話，這一段的normalize邏輯有點怪'
    assert 'CLOSE' not in var_y, msg


    # 應要先獨立把y的欄位標準化，否則ml_data_process中處理的會是ma後的欄位
    loc_main, norm_orig = \
        cbml.df_normalize(
            df=loc_main,
            cols=var_y,
            groupby=[],
            show_progress=False
            )  
                
    
    ratio_cols = []
    if 'CLOSE_CHANGE_RATIO' in loc_main.columns:
        
        # cols = []
        # for i in range(len(var_y)):
        #     col = var_y[i]
        #     cols.append(col)
            # loc_main[col + '_GLOB_NORM'] = loc_main[col]
    
        cols = list(loc_main.columns)
        cols = [c for c in cols if 'RATIO' in c]
        # ['OPEN_CHANGE_RATIO',
        #  'OPEN_CHANGE_ABS_RATIO',
        #  'HIGH_CHANGE_RATIO',
        #  'HIGH_CHANGE_ABS_RATIO',
        #  'LOW_CHANGE_RATIO',
        #  'LOW_CHANGE_ABS_RATIO',
        #  'CLOSE_CHANGE_RATIO',
        #  'CLOSE_CHANGE_ABS_RATIO',
        #  'VOLUME_CHANGE_RATIO',
        #  'VOLUME_CHANGE_ABS_RATIO',
        #  'SYMBOL_TRADE_VALUE_RATIO']       
    
        loc_main, ratio_cols, _ = \
            cbml.ml_data_process(df=loc_main,
                                 ma=True, normalize=True, lag=True, 
                                 group_by=[],
                                 cols=cols,
                                 except_cols=[],
                                 drop_except=var_y,
                                 cols_mode='equal',
                                 date_col='WORK_DATE',
                                 ma_values=ma_values, 
                                 lag_period=predict_period
                                 )


    # Check again after deleting 
    # na_cols = cbyz.df_chk_col_na(df=loc_main)
    # max_value = na_cols['NA_COUNT'].max()
    # min_value = na_cols['NA_COUNT'].min()
    # assert max_value == min_value, 'max and min should be the same'
    
    
    # Normalize By Stock ......
    except_cols = ['WORK_DATE', 'YEAR', 'MONTH', 'WEEKDAY', 'WEEK_NUM'] \
                    + ratio_cols
    
    loc_main, _, _ = \
        cbml.ml_data_process(df=loc_main, 
                             ma=True, normalize=True, lag=True,
                             date_col='WORK_DATE',
                             group_by=['SYMBOL'],   
                             cols=[], 
                             except_cols=except_cols,
                             drop_except=var_y,
                             cols_mode='equal',
                             ma_values=ma_values, 
                             lag_period=predict_period
                             )

    # assert 2 < 1, '這裡的na數量不一致'    
    # na_cols = cbyz.df_chk_col_na(df=loc_main)
    
        
    # Drop Except會導致CLOSE_LAG, HIGH_LAG沒被排除
    if 'CLOSE' in var_y:
        global ohlc
        ohlc_str = '|'.join(ohlc)
        drop_cols = cbyz.df_chk_col_na(df=loc_main, positive_only=True)
        
        drop_cols = drop_cols[(~drop_cols['COLUMN'].isin(var_y)) \
                              & (~drop_cols['COLUMN'].str.contains('MA')) \
                              & (drop_cols['COLUMN'].str.contains(ohlc_str))]
        
        drop_cols = drop_cols['COLUMN'].tolist()
        loc_main = loc_main.drop(drop_cols, axis=1)
    
        
    # Total Market Trade
    if trade_value:
        total_trade = market_data_raw[['WORK_DATE', 'TOTAL_TRADE_VALUE']] \
                    .drop_duplicates(subset=['WORK_DATE'])
        
        total_trade, _, _ = \
            cbml.ml_data_process(df=total_trade, 
                                 ma=True, normalize=True, lag=True, 
                                 date_col='WORK_DATE',
                                 cols_mode='equal',
                                 group_by=[],
                                 cols=['TOTAL_TRADE_VALUE'], 
                                 except_cols=[],
                                 drop_except=['WORK_DATE'],
                                 ma_values=ma_values, 
                                 lag_period=predict_period
                                 )
        
        loc_main = loc_main.merge(total_trade, how='left', on=['WORK_DATE'])  


    # Stock Info ...
    # ['SYMBOL', 'CAPITAL', 'CAPITAL_LEVEL', 
    # 'ESTABLISH_DAYS', 'LISTING_DAYS']
    stock_info = stock_info_raw.drop(['INDUSTRY_ONE_HOT'], axis=1)
    
    stock_info, _, _ = \
        cbml.ml_data_process(df=stock_info, 
                             ma=False, normalize=True, lag=False,
                             date_col='WORK_DATE',
                             cols_mode='equal',
                             group_by=[],
                             cols=[],
                             except_cols=['SYMBOL'],
                             drop_except=[],
                             ma_values=ma_values, 
                             lag_period=predict_period
                             )
    
    loc_main = loc_main.merge(stock_info, how='left', on=['SYMBOL'])      


    # Merge Other Data ......        
    if industry:        
        stock_industry = stock_info_raw[['SYMBOL', 'INDUSTRY_ONE_HOT']]
        
        stock_info_dummy = \
            cbml.df_get_dummies(df=stock_industry, 
                                cols='INDUSTRY_ONE_HOT'
                                )
        
        # Industry Data ...
        print('sam_load_data - 當有新股上市時，產業資料的比例會出現大幅變化，' \
              + '評估如何處理')
        
        if trade_value:
            industry_data = \
                market_data_raw[['SYMBOL', 'WORK_DATE', 'VOLUME', 
                                 'OPEN', 'OPEN_CHANGE', 
                                 'HIGH', 'HIGH_CHANGE',
                                 'LOW', 'LOW_CHANGE', 
                                 'CLOSE', 'CLOSE_CHANGE', 
                                 'SYMBOL_TRADE_VALUE', 
                                 'TOTAL_TRADE_VALUE']]
        else:
            industry_data = \
                market_data_raw[['SYMBOL', 'WORK_DATE', 'VOLUME', 
                                 'OPEN', 'OPEN_CHANGE', 
                                 'HIGH', 'HIGH_CHANGE',
                                 'LOW', 'LOW_CHANGE', 
                                 'CLOSE', 'CLOSE_CHANGE']]

        # Merge        
        industry_data = industry_data.merge(stock_industry, on='SYMBOL')
        
        
        if trade_value:
            
            industry_data['TRADE_VALUE'] = \
                industry_data \
                .groupby(['WORK_DATE', 'INDUSTRY_ONE_HOT'])['SYMBOL_TRADE_VALUE'] \
                .transform('sum')
    
            industry_data['TRADE_VALUE_RATIO'] = \
                industry_data['TRADE_VALUE'] / industry_data['TOTAL_TRADE_VALUE']            
            
            industry_data = industry_data \
                            .groupby(['WORK_DATE', 'INDUSTRY_ONE_HOT']) \
                            .agg({'OPEN':'sum', 
                                  'HIGH':'sum', 
                                  'LOW':'sum', 
                                  'CLOSE':'sum', 
                                  'VOLUME':'sum',
                                  'OPEN_CHANGE':'sum',
                                  'HIGH_CHANGE':'sum',
                                  'LOW_CHANGE':'sum',
                                  'CLOSE_CHANGE':'sum',
                                  'TRADE_VALUE':'sum',
                                  'TRADE_VALUE_RATIO':'sum'}) \
                            .reset_index()        
        else:
            industry_data = industry_data \
                            .groupby(['WORK_DATE', 'INDUSTRY_ONE_HOT']) \
                            .agg({'OPEN':'sum', 
                                  'HIGH':'sum', 
                                  'LOW':'sum', 
                                  'CLOSE':'sum', 
                                  'VOLUME':'sum',
                                  'OPEN_CHANGE':'sum',
                                  'HIGH_CHANGE':'sum',
                                  'LOW_CHANGE':'sum',
                                  'CLOSE_CHANGE':'sum'}) \
                            .reset_index()
        
        # Rename ...
        cols = cbyz.df_get_cols_except(
            df=industry_data,
            except_cols=['WORK_DATE', 'INDUSTRY_ONE_HOT']
            )
        
        new_cols = ['INDUSTRY_' + c for c in cols]                  
        rename_dict = cbyz.li_to_dict(cols, new_cols)
        industry_data = industry_data.rename(columns=rename_dict)
                       
        
        industry_data, _, _ = \
             cbml.ml_data_process(df=industry_data, 
                                  ma=True, normalize=True, lag=True,
                                  group_by=['INDUSTRY_ONE_HOT'],
                                  cols=[], 
                                  except_cols=['WORK_DATE'],
                                  drop_except=[],
                                  date_col='WORK_DATE',
                                  cols_mode='equal',
                                  ma_values=ma_values, 
                                  lag_period=predict_period)
        
        # Merge ...
        loc_main = loc_main \
            .merge(stock_info_dummy, how='left', on='SYMBOL') \
            .merge(stock_industry, how='left', on='SYMBOL') \
            .merge(industry_data, how='left', on=['WORK_DATE', 'INDUSTRY_ONE_HOT']) \
            .drop('INDUSTRY_ONE_HOT', axis=1)
        
        



    # Check NA ......
    # 有些新股因為上市時間較晚，在MA_LAG中會有較多的NA，所以只處理MA的欄位
    na_cols = cbyz.df_chk_col_na(df=loc_main)
    na_cols = na_cols[~na_cols['COLUMN'].isin(var_y)]
    na_cols = na_cols['COLUMN'].tolist()
    
    loc_main = loc_main.dropna(subset=na_cols, axis=0)

        
    # Check for min max
    chk_min_max = cbyz.df_chk_col_min_max(df=loc_main)
    chk_min_max = chk_min_max[chk_min_max['COLUMN']!='WORK_DATE']
    chk_min_max = chk_min_max[(chk_min_max['MIN_VALUE']<0) \
                              | (chk_min_max['MAX_VALUE']>1)]
        
    assert len(chk_min_max) == 0, 'chk_min_max failed'
    
        
    return loc_main, norm_orig


# ...........


def get_sale_mon_data():
    
    '''
    除權息時間
    Optimize
    '''
    
    file_raw = pd.DataFrame()
    
    print('bug - 目前只有到2018')
    years = list(range(2018, 2022))

    for y in years:
        new_file = pd.read_excel(path_resource + '/sale_mon/SaleMonDetail_' \
                                 + str(y) + '.xlsx')
        file_raw = file_raw.append(new_file)
        
        
    new_cols = range(len(file_raw.columns))
    new_cols = ['SALE_MON_' + str(c) for c in new_cols]

    file_raw.columns = new_cols

    file_raw = file_raw[['SALE_MON_1', 'SALE_MON_4', 'SALE_MON_5', 'SALE_MON_6']]
    file_raw.columns = ['SYMBOL', 'WORK_DATE', 
                        'EX_DIVIDENDS_PRICE', 'EX_DIVIDENDS_DONE']
    file_raw = file_raw.dropna()
    
    
    file1 = file_raw[['SYMBOL', 'WORK_DATE', 'EX_DIVIDENDS_PRICE']]
    file1['WORK_DATE'] = '20' + file1['WORK_DATE']
    file1['WORK_DATE'] = file1['WORK_DATE'].str.replace("'", "")
    file1['WORK_DATE'] = file1['WORK_DATE'].str.replace("/", "")
    file1 = cbyz.df_conv_col_type(df=file1, cols='WORK_DATE', to='int')
    file1 = cbyz.df_conv_col_type(df=file1, cols='EX_DIVIDENDS_PRICE',
                                  to='float')    

    file1['SALE_MON_DATE'] = 1
    file1 = cbyz.df_conv_na(df=file1, 
                            cols=['EX_DIVIDENDS_PRICE', 'SALE_MON_DATE'])

    # 填息
    file2 = file_raw[['SYMBOL', 'EX_DIVIDENDS_DONE']]
    file2.columns = ['SYMBOL', 'WORK_DATE']
    file2['WORK_DATE'] = '20' + file2['WORK_DATE']
    file2['WORK_DATE'] = file2['WORK_DATE'].str.replace("'", "")
    file2['WORK_DATE'] = file2['WORK_DATE'].str.replace("/", "")
    file2 = cbyz.df_conv_col_type(df=file2, cols='WORK_DATE', to='int')
    file2['EX_DIVIDENDS_DONE'] = 1
    
    file2 = cbyz.df_conv_na(df=file2, cols=['EX_DIVIDENDS_DONE'])
    
    return file1, file2


# .............


def select_symbols():

    '''
    Version Note
    
    1. Exclude small capital

    '''    

    global market_data_raw
    global stock_info_raw


    # Exclude ETF ......
    all_symbols = stock_info_raw[['SYMBOL']]
    df = all_symbols.merge(market_data_raw, on=['SYMBOL']) 


    # Exclude low volume in the past 7 days
    global volume_thld
    global data_end
    loc_begin = cbyz.date_cal(data_end, -7, 'd')
    
    low_volume = df[(df['WORK_DATE']>=loc_begin) & (df['WORK_DATE']<=data_end)]
    low_volume = low_volume \
                .groupby(['SYMBOL']) \
                .agg({'VOLUME':'min'}) \
                .reset_index()
        
    low_volume = low_volume[low_volume['VOLUME']<=volume_thld * 1000]
    low_volume = low_volume[['SYMBOL']].drop_duplicates()
    
    global low_volume_symbols
    low_volume_symbols = low_volume['SYMBOL'].tolist()
    
    # 為了避免low_volume_symbols的數量過多，導致計算效率差，因此採用df做anti_merge，
    # 而不是直接用list
    if len(low_volume_symbols) > 0:
        df = cbyz.df_anti_merge(df, low_volume, on='SYMBOL')
        
    # Add log
    log_msg = 'low_volume_symbols - ' + str(len(low_volume_symbols))
    log.append(log_msg)        
    
    return df



# %% Process ------


def get_model_data(industry=True, trade_value=True):
    
    
    global shift_begin, shift_end, data_begin, data_end, ma_values
    global predict_date, predict_period, calendar    
    global symbols
    global var_y
    global market_data_raw  
    global params, error_msg
    global id_keys
    

    # Check ......
    msg = 'get_model_data - predict_period is longer than ma values, ' \
            + 'and it will cause na.'
    assert predict_period <= min(ma_values), msg


    # Symbols ......
    symbols = cbyz.conv_to_list(symbols)
    symbols = cbyz.li_conv_ele_type(symbols, 'str')


    # Market Data ......
    # market_data_raw
    get_market_data_raw(trade_value=trade_value)
    gc.collect()
    
    
    # Load Historical Data ......
    main_data_raw, norm_orig = \
        sam_load_data(industry=industry, trade_value=trade_value) 

        
    main_data = main_data_raw.copy()
    cbyz.df_chk_col_na(df=main_data_raw)
    
    # TODC Shareholdings Spread ......
    # sharehold = stk.tdcc_get_sharehold_spread(shift_begin, end_date=None,
    #                                           local=local) 
    
    # main_data = main_data.merge(sharehold, how='left', 
    #                           on=['SYMBOL', 'WORK_DATE'])      


    # Government Invest ......
    gov_invest = stk.opd_get_gov_invest(path=path_resource)
    main_data = main_data.merge(gov_invest, how='left', on=['SYMBOL'])
    main_data = cbyz.df_conv_na(df=main_data, cols=['GOV_INVEST'])


    # 除權息資料 ......
    # Close Lag ...
    daily_close = market_data_raw[['WORK_DATE', 'SYMBOL', 'CLOSE']]
    daily_close, _ = cbyz.df_add_shift(df=daily_close, 
                                    cols='CLOSE', shift=1,
                                    group_by=['SYMBOL'],
                                    suffix='_LAG', 
                                    remove_na=False)
    daily_close = daily_close \
                .drop('CLOSE', axis=1) \
                .rename(columns={'CLOSE_LAG':'CLOSE'})
    
    # 除權息 ...
    sale_mon_data1, sale_mon_data2 = get_sale_mon_data()
    
    # Data 1 - 除權息日期及價錢 ...
    sale_mon_data1 = daily_close.merge(sale_mon_data1, how='left', 
                                       on=['WORK_DATE', 'SYMBOL'])
    
    sale_mon_data1['EX_DIVIDENDS_PRICE'] = \
        sale_mon_data1['EX_DIVIDENDS_PRICE'] / sale_mon_data1['CLOSE']    
        
    sale_mon_data1 = sale_mon_data1.drop('CLOSE', axis=1)
    sale_mon_data1 = cbyz.df_conv_na(df=sale_mon_data1, 
                                     cols=['EX_DIVIDENDS_PRICE', 
                                           'SALE_MON_DATE'])
    
    sale_mon_data1, _, _ = \
             cbml.ml_data_process(df=sale_mon_data1, 
                                  ma=False, normalize=True, lag=False, 
                                  group_by=['SYMBOL'], 
                                  cols=[],
                                  except_cols=[],
                                  drop_except=[],
                                  date_col='WORK_DATE',
                                  ma_values=ma_values, 
                                  lag_period=predict_period
                                  ) 
    
    # Data 2 - 填息 ...
    sale_mon_data2, _, _ = \
        cbml.ml_data_process(df=sale_mon_data2, 
                             ma=False, normalize=True, lag=False, 
                             group_by=['SYMBOL'],
                             cols=[], 
                             except_cols=['EX_DIVIDENDS_DONE'],
                             drop_except=[],
                             cols_mode='equal',
                             date_col='WORK_DATE',
                             ma_values=ma_values, 
                             lag_period=predict_period
                             )
        
    main_data = main_data \
        .merge(sale_mon_data1, how='left', on=['WORK_DATE', 'SYMBOL']) \
        .merge(sale_mon_data2, how='left', on=['WORK_DATE', 'SYMBOL'])
    
    # Convert NA
    temp_cols = ['EX_DIVIDENDS_PRICE', 'SALE_MON_DATE', 'EX_DIVIDENDS_DONE']    
    main_data = cbyz.df_conv_na(df=main_data, cols=temp_cols)


    
    # TEJ 三大法人持股成本 ......
    ewtinst1c_raw = stk.tej_get_ewtinst1c(begin_date=shift_begin, 
                                          end_date=None, 
                                          trade=True)
    
    ewtinst1c = ewtinst1c_raw.copy()
    
    # 獲利率HROI、Sell、Buy用globally normalize，所以要分兩段
    hroi_cols = cbyz.df_get_cols_contains(df=ewtinst1c, 
                                         string=['_HROI', '_SELL', '_BUY'])
    
    # other_cols = cbyz.df_get_cols_except(
    #     df=ewtinst1c, 
    #     except_cols=hroi_cols + ['SYMBOL', 'WORK_DATE'])
    
    
    # Keep Needed Symbols Only
    ewtinst1c = ewtinst1c.merge(symbol_df, on=['SYMBOL'])
    
    print('ewtinst1c - 這段結束時會有NA')
    ewtinst1c, cols_1, _ = \
        cbml.ml_data_process(df=ewtinst1c, 
                             ma=True, normalize=True, lag=True, 
                             group_by=[],
                             cols=hroi_cols, 
                             except_cols=[],
                             drop_except=[],
                             cols_mode='contains',
                             date_col='WORK_DATE',
                             ma_values=ma_values, 
                             lag_period=predict_period
                             ) 

    ewtinst1c, cols_2, _ = \
        cbml.ml_data_process(df=ewtinst1c, 
                             ma=True, normalize=True, lag=True, 
                             group_by=['SYMBOL'],
                             cols=['_HAP'], 
                             except_cols=[],
                             drop_except=[],
                             cols_mode='contains',
                             date_col='WORK_DATE',
                             ma_values=ma_values, 
                             lag_period=predict_period
                             )
        
    main_data = main_data \
        .merge(ewtinst1c, how='left', on=['SYMBOL', 'WORK_DATE'])  
    
    print('Check - 全部填NA是否合理？')
    main_data = cbyz.df_conv_na(df=main_data, cols=cols_1 + cols_2)


    # 月營收資料表 ......
    # 1. 當predict_date=20211101，且為dev時, 造成每一個symbol都有na，先移除
    # 1. 主要邏輯就是顯示最新的營收資料
    # print('Update - 增加date index')
    # ewsale = stk.tej_get_ewsale(begin_date=shift_begin, end_date=None, 
    #                             SYMBOL=symbols, trade=True)
    
    # ewsale, cols, _ = \
    #     cbml.ml_data_process(df=ewsale, 
    #                          ma=False, normalize=True, lag=False, 
    #                          ma_group_by=['SYMBOL'],
    #                          norm_group_by=['SYMBOL'], 
    #                          lag_group_by=['SYMBOL'],
    #                          ma_cols_contains=[], 
    #                          ma_except_contains=[],
    #                          norm_cols_contains=['D000'], 
    #                          norm_except_contains=[],
    #                          lag_cols_contains=[], 
    #                          lag_except_contains=[], 
    #                          drop_except_contains=[],
    #                          ma_values=ma_values, 
    #                          lag_period=predict_period)    
    
    # ewsale, pre_cols = cbyz.df_add_shift(df=ewsale,
    #                                      cols=['D0001', 'D0002', 'D0003'], 
    #                                      shift=-1, group_by=['SYMBOL'],
    #                                      suffix='_PRE', remove_na=False)
    
    # # 因為main_data的YEAR和MONTH已經標準化過了，要直接採用標準化後的數值
    # # Bug，但是這裡的year沒有標準化，現在是在get_model_data最後面直接把YEAR刪除
    # year_month = main_data[['WORK_DATE', 'YEAR', 'MONTH']] \
    #             .drop_duplicates()
                
    # ewsale = ewsale.merge(year_month, how='inner', on=['WORK_DATE'])
    # ewsale = ewsale.rename(columns={'WORK_DATE':'EWSALE_WORK_DATE',
    #                                 'D0001':'D0001_CUR',
    #                                 'D0002':'D0002_CUR',
    #                                 'D0003':'D0003_CUR'})
    
    # main_data = main_data \
    #         .merge(ewsale, how='left', on=['SYMBOL', 'YEAR', 'MONTH'])  

    # cond = main_data['WORK_DATE'] >= main_data['EWSALE_WORK_DATE']
    # main_data['D0001'] = np.where(cond, 
    #                               main_data['D0001_CUR'], 
    #                               main_data['D0001_PRE'])
    
    # main_data['D0002'] = np.where(cond, 
    #                               main_data['D0002_CUR'], 
    #                               main_data['D0002_PRE'])

    # main_data['D0003'] = np.where(cond, 
    #                               main_data['D0003_CUR'], 
    #                               main_data['D0003_PRE'])
    
    # main_data = main_data.drop(['EWSALE_WORK_DATE'] + cols + pre_cols,
    #                            axis=1)
    
    
    # 指數日成本 ......
    # - Increase prediction time a lot, but not increase mape obviously.
    # ewiprcd = stk.tej_get_ewiprcd()
    # main_data = main_data.merge(ewiprcd, how='left', on=['WORK_DATE'])  


    # Pytrends Data ......
    # - Increase prediction time a lot, and made mape decrease.
    # - Pytrends已經normalize過後才pivot，但後面又normalize一次
    # pytrends, pytrends_cols = get_google_treneds(begin_date=shift_begin, 
    #                                               end_date=data_end, 
    #                                               normalize=True, 
    #                                               stock_type=stock_type, 
    #                                               local=local)
    
    # main_data = main_data.merge(pytrends, how='left', on=['WORK_DATE'])      
    
    
    # COVID-19 ......
    covid19, _ = cbyz.get_covid19_data()
    
    covid19, covid19_cols, _ = \
                cbml.ml_data_process(df=covid19, 
                                     ma=True, normalize=True, lag=True, 
                                     group_by=[],
                                     cols=[], 
                                     except_cols=[],
                                     drop_except=[],
                                     cols_mode='equal',
                                     date_col='WORK_DATE',
                                     ma_values=ma_values, 
                                     lag_period=predict_period
                                     )
        
        
    main_data = main_data.merge(covid19, how='left', on='WORK_DATE')
    main_data = cbyz.df_conv_na(df=main_data, cols=covid19_cols)


    # Variables ......
    model_x = cbyz.df_get_cols_except(df=main_data, 
                                      except_cols=var_y + id_keys)
    
    
    # Model Data ......
    main_data = main_data[main_data['WORK_DATE']>=data_begin] \
                        .reset_index(drop=True)


    # Check NA ......
    hist_df = main_data[main_data['WORK_DATE']<predict_date[0]]
    cbyz.df_chk_col_na(df=hist_df, mode='stop')
    
    # Predict有NA是正常的，但NA_COUNT必須全部都一樣
    global chk_predict_na
    predict_df = main_data[main_data['WORK_DATE']>=predict_date[0]]
    chk_predict_na = cbyz.df_chk_col_na(df=predict_df, mode='alert')
    
    
    min_value = chk_predict_na['NA_COUNT'].min()
    max_value = chk_predict_na['NA_COUNT'].max()    
    assert min_value == max_value, 'All the NA_COUNT should be the same. '
    

    # Check min max ......
    global chk_min_max
    chk_min_max = cbyz.df_chk_col_min_max(df=main_data)
    
    chk_min_max = \
        chk_min_max[(~chk_min_max['COLUMN'].isin(id_keys)) \
                    & ((chk_min_max['MIN_VALUE']<0) \
                       | (chk_min_max['MAX_VALUE']>1))]
    
    assert len(chk_min_max) == 0, 'get_model_data - normalize error'

    #                      COLUMN    MIN_VALUE     MAX_VALUE
    # 0       VOLUME_CHANGE_RATIO         -1.0           inf
    # 1   VOLUME_CHANGE_ABS_RATIO          0.0           inf
    # 2                    VOLUME          0.0  1.281795e+09
    # 3             VOLUME_CHANGE -834536728.0  1.109633e+09
    # 4         VOLUME_CHANGE_ABS          0.0  1.109633e+09
    # 5        SYMBOL_TRADE_VALUE          0.0  1.452704e+08
    # 7             SHADOW_LENGTH          0.0  1.170000e+02
    # 8                       BAR          0.0  1.070000e+02
    # 9                TOP_SHADOW         -2.0  7.300000e+01
    # 10            BOTTOM_SHADOW         -2.5  6.700000e+01

            
    export_dict = {'MODEL_DATA':main_data,
                   'MODEL_X':model_x,
                   'NORM_ORIG':norm_orig}
    
    return export_dict



# %% Master ------

def master(param_holder, predict_begin, export_model=True, load_model=False, 
           threshold=30000):
    '''
    主工作區
    '''
    
    # date_period為10年的時候會出錯
    

    # v1.0
    # - Add ml_data_process
    # v1.0.1
    # - Add Auto Model Tuning
    # v1.03
    # - Merge predict and tuning function from uber eats order forecast
    # - Add small capitall back
    # - Add TEJ market data function, multiply 1000
    # v1.04
    # - Add price change for OHLC
    # - Fix industry bug
    # v1.05
    # - Update for new df_normalize
    # - Test price_change_ratio as y
    # - Add check for min max
    # v1.06
    # - Change local variable as host
    # v1.07
    # - Update for cbyz and cbml
    # - Add low_volume_symbols
    # v1.08
    # - Add 每月投報率 data
    
    
    # v2.00
    # - Add Ultra_Tuner
    # - Add Correlation detection
    # - Fix Ex-dividend issues
    # - Rename stock_type and SYMBOL >> Done
    # - Set df_fillna with interpolate method in arsenal_stock
    # - Fix Date Issues
    # - Add correlations detection
    # - select_symbols用過去一周的總成交量檢查
        
    
    # cbyz.df_get_cols_contains(df, string=[], exclude=[])不需要用for loop，可以直接
    # 合併，並用|
    # ohlc = '|'.join(ohlc)
    # .str.contaINS(ohlc)
    
    
    # v1.10
    # - Combine cbyz >> detect_cycle.py, including support_resistance and 
    #    season_decompose
    # - Add 流通股數 from ewprcd
    
    
    # v1.11
    # - Daily backup
    # >> tw_get_stock_info_twse
    # - 合併Yahoo Finance和TEJ的market data，兩邊都有可能缺資料。現在的方法是用interpolate，
    #   但如果begin_date剛好缺值，這檔股票就會被排除
    # - 在data中多一個欄位標註新股，因為剛上市的時候通常波動較大
    
    
    # Bug
    # 1. get_sale_mon_data - bug - 目前只有到2018
    # 2. Add covid-19 daily backup
    # 4. Fix industry issues，是不是要做PCA
    # 1. Fill na with interpolate
    # 3. 在get_market_data_raw中, OPEN_CHANGE_RATIO min_value 
    #    is -0.8897097625329815，暫時移除assert
    # 5. 有些symbole會不見，像正隆
    # 2. pred price和Last price都有錯置的問題，應該是stk有bug, df_fillna的interpolate
    
    # -NA issues, replace OHLC na in market data function, and add replace 
    # na with interpolation. And why 0101 not excluded?
    # print('Bug - 執行到這裡時，main_data裡面會有NA, 主要是INDUSTRY的問題')>這是lag的問題


    # Optimization .....
    # - Add week
    # 建長期投資的基本面模型
    # 5. print('add_support_resistance - days == True時有bug，全部數值一樣，導致沒辦法標準化？')
    # 6. 加上PRICE_CHANGE_ABS / PRICE_CHANGE加上STD
    # 6. 技術分析型態學
    # 8. Update CBYZ and auto-competing model
    # 9. 上櫃的也要放
    # 10. buy_signal to int
    
    
    global bt_last_begin, data_period, predict_period
    global debug, dev
    
    global symbols, ma_values, volume_thld, market


    holder = param_holder.params
    
    bt_last_begin = holder['bt_last_begin'][0]
    industry = holder['industry'][0]
    trade_value = holder['trade_value'][0]
    data_period = holder['data_period'][0]
    market = holder['market'][0]
    volume_thld = holder['volume_thld'][0]   
    compete_mode = holder['compete_mode'][0]   
    train_mode = holder['train_mode'][0]       
    dev = holder['dev'][0]   
    symbols = holder['symbols'][0]   
    ma_values = holder['ma_values'][0]   
    data_shift = -(max(ma_values) * 2)
    
    # Modeling
    predict_period = holder['predict_period'][0]
    kbest = holder['kbest'][0]
    cv = holder['cv'][0]
    
    # Program
    debug = holder['debug'][0]
    
    
    # fast = True
    # export_model = False
    # dev = True
    # threshold = 20000
    # predict_begin=20211209
    
    
    global version, exe_serial
    version = 2.00
    exe_serial = cbyz.get_time_serial(with_time=True, remove_year_head=True)

    global log, error_msg, ohlc
    log = []
    error_msg = []    
    ohlc = stk.get_ohlc()
    
    
    global shift_begin, shift_end, data_begin, data_end
    global predict_date, calendar        
    
    shift_begin, shift_end, \
            data_begin, data_end, predict_date, calendar = \
                stk.get_period(predict_begin=predict_begin,
                               predict_period=predict_period,
                               data_period=data_period,
                               shift=data_shift)  

    # .......
    # global symbols, market
    # market = _market
    # symbols = _symbols


    # ......
    global model_data
    global model_x, var_y, id_keys
    global norm_orig
        
    
    var_y = ['OPEN_CHANGE_RATIO', 'HIGH_CHANGE_RATIO',
               'LOW_CHANGE_RATIO', 'CLOSE_CHANGE_RATIO']
    id_keys = ['SYMBOL', 'WORK_DATE']    
    
    
    # 20210707 - industry可以提高提精準，trade_value會下降
    data_raw = get_model_data(industry=industry, 
                              trade_value=trade_value)
    
    
    model_data = data_raw['MODEL_DATA']
    model_x = data_raw['MODEL_X']
    norm_orig = data_raw['NORM_ORIG']

    print('WEEK_NUM的type是OBJ，先排除')
    model_data = model_data.drop('WEEK_NUM', axis=1)

    # Training Model ......
    if len(symbols) > 0 and len(symbols) < 10:
        model_params = [{'model': LinearRegression(),
                         'params': {
                             'normalize': [True, False],
                             }
                         }]         
    else:
        model_params = [
                        {'model': LinearRegression(),
                          'params': {
                              'normalize': [True, False],
                              }
                          },
                        {'model': xgb.XGBRegressor(),
                         'params': {
                            'n_estimators': [200],
                            'gamma':[0],
                            'max_depth':[4],                
                            }
                        }
                       ] 

    # 1. 如果selectkbest的k設得太小時，importance最高的可能都是industry，導致同產業
    #    的預測值完全相同
    global pred_result, pred_scores, pred_params, pred_features
    
    
    for i in range(len(var_y)):
        
        cur_y = var_y[i]
        remove_y = [var_y[j] for j in range(len(var_y)) if j != i]
        
        tuner = ut.Ultra_Tuner(id_keys=id_keys, y=cur_y, model_type='reg',
                               compete_mode=compete_mode, 
                               train_mode=train_mode, 
                               path=path_temp)
        
        # 排除其他y，否則會出錯
        cur_model_data = model_data.drop(remove_y, axis=1)
        
        return_result, return_scores, return_params, return_features, \
                log_scores, log_params, log_features = \
                    tuner.fit(data=cur_model_data, model_params=model_params,
                              k=kbest, cv=cv, threshold=threshold, 
                              norm_orig=norm_orig,
                              export_model=True, export_log=True)
                    
        # ut的norm_orig有bug，所以先獨立出來
        # return_result = cbml.df_normalize_restore(df=return_result, 
        #                                        original=norm_orig)

        if i == 0:
            pred_result = return_result.copy()
            pred_scores = return_scores.copy()
            pred_params = return_params.copy()
            pred_features = return_features.copy()
        else:
            pred_result = pred_result.merge(return_result, how='left', on=id_keys)
            pred_scores = pred_scores.append(return_scores)
            pred_params = pred_scores.append(return_params)
            pred_features = pred_scores.append(return_features)            
        
        
    # Upload to Google Sheet
    if predict_period == bt_last_begin:
        stk.write_sheet(data=pred_features, sheet='Features')
        
    
    return pred_result, pred_scores, pred_params, pred_features




def update_history():
    
    # v0.0 - Mess version
    # v0.1 - Fix mess version
    # v0.2 - 1. Add calendar    
    # - Support Days True
    # v0.3 
    # - Fix induxtry
    # - Add daily backup for stock info
    # - Update resistance and support function in stk
    # v0.4
    # - Fix trade value
    # - Add TEJ data
    # - Precision stable version
    # v0.5
    # - Add symbol vars
    # v0.6
    # - Add TODC shareholding spread data, but temporaily commented
    # - Add TEJ 指數日行情
    # v0.7
    # - Add Google Trends
    # - Disable Pytrends and ewiprcd
    # v0.8
    # - Optimize data processing
    # v0.9
    # - Add industry in Excel
    
    pass




# %% Check ------


def check():
    
    pass

# %% Manually Analyze ------


def select_symbols_manually(data_begin, data_end):


    # Select Rules
    # 1. 先找百元以下的，才有資金可以買一整張
    # 2. 不要找疫情後才爆漲到歷史新高的

    global market
    data_end = cbyz.date_get_today()
    data_begin = cbyz.date_cal(data_end, -1, 'm')


    # Stock info
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    
    
    # Section 1. 資本額大小 ------
    
    # 1-1. 挑選中大型股 ......
    level3_symbol = stock_info[stock_info['CAPITAL_LEVEL']>=2]
    level3_symbol = level3_symbol['SYMBOL'].tolist()
    
    data_raw = stk.get_data(data_begin=data_begin, data_end=data_end, 
                            SYMBOL=level3_symbol, 
                            price_change=True,
                            shift=0, stock_type=market)
    
    data = data_raw[data_raw['SYMBOL'].isin(level3_symbol)]
    

    # 1-2. 不排除 ......
    data = stk.get_data(data_begin=data_begin, data_end=data_end, 
                            SYMBOL=[], 
                            price_change=True,
                            shift=0, stock_type=market)
    
    
    # Section 2. 依價格篩選 ......
    
    # 2-1. 不篩選 .....
    target_symbols = data[['SYMBOL']] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    
    
    # 2-2. 低價股全篩 .....
    # 目前排除80元以上
    last_date = data['WORK_DATE'].max()
    last_price = data[data['WORK_DATE']==last_date]
    last_price = last_price[last_price['CLOSE']>80]
    last_price = last_price[['SYMBOL']].drop_duplicates()
    
    
    target_symbols = cbyz.df_anti_merge(data, last_price, on='SYMBOL')
    target_symbols = target_symbols[['SYMBOL']].drop_duplicates()
    
    
    # 2-3. 3天漲超過10%  .....
    data, cols_pre = cbyz.df_add_shift(df=data, 
                                       group_by=['SYMBOL'], 
                                       cols=['CLOSE'], shift=3,
                                       remove_na=False)
    

    data['PRICE_CHANGE_RATIO'] = (data['CLOSE'] - data['CLOSE_PRE']) \
                            / data['CLOSE_PRE']
    
    
    results_raw = data[data['PRICE_CHANGE_RATIO']>=0.15]
    
    
    summary = results_raw \
                .groupby(['SYMBOL']) \
                .size() \
                .reset_index(name='COUNT')
                
    
    # Select Symboles ......
    target_symbols = results_raw.copy()
    target_symbols = cbyz.df_add_size(df=target_symbols,
                                      group_by='SYMBOL',
                                      col_name='TIMES')
        
    target_symbols = target_symbols \
                    .groupby(['SYMBOL']) \
                    .agg({'CLOSE':'mean',
                          'TIMES':'mean'}) \
                    .reset_index()
    
    target_symbols = target_symbols.merge(stock_info, how='left', 
                                          on='SYMBOL')
    
    target_symbols = target_symbols \
                        .sort_values(by=['TIMES', 'CLOSE'],
                                     ascending=[False, True]) \
                        .reset_index(drop=True)
                        
    target_symbols = target_symbols[target_symbols['CLOSE']<=100] \
                            .reset_index(drop=True)


    # Export ......
    time_serial = cbyz.get_time_serial(with_time=True)
    target_symbols.to_csv(path_export + '/target_symbols_' \
                          + time_serial + '.csv',
                          index=False, encoding='utf-8-sig')

    # target_symbols.to_excel(path_export + '/target_symbols_' \
    #                         + time_serial + '.xlsx',
    #                         index=False)

    # Plot ......       
    # plot_data = results.melt(id_keys='PROFIT')

    # cbyz.plotly(df=plot_data, x='PROFIT', y='value', groupby='variable', 
    #             title="", xaxes="", yaxes="", mode=1)

    
    return results_raw, stock_info



def check_price_limit():
    
    loc_stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    loc_stock_info = loc_stock_info[['SYMBOL', 'CAPITAL_LEVEL']]
    
    
    loc_market = stk.get_data(data_begin=20190101, 
                        data_end=20210829, 
                        stock_type='tw', SYMBOL=[], 
                        price_change=True, price_limit=True, 
                        trade_value=True)
    
    loc_main = loc_market.merge(loc_stock_info, how='left', 
                                on=['SYMBOL'])

    # Check Limit Up ......
    chk_limit = loc_main[~loc_main['CAPITAL_LEVEL'].isna()]
    chk_limit = chk_limit[chk_limit['LIMIT_UP']==1]

    chk_limit_summary = chk_limit \
            .groupby(['CAPITAL_LEVEL']) \
            .size() \
            .reset_index(name='COUNT')


    # Check Volume ......
    #    OVER_1000  COUNT
    # 0          0    115
    # 1          1    131    
    chk_volum = loc_main[loc_main['CAPITAL_LEVEL']==1]
    chk_volum = chk_volum \
                .groupby(['SYMBOL']) \
                .agg({'VOLUME':'min'}) \
                .reset_index()
                
    chk_volum['OVER_1000'] = np.where(chk_volum['VOLUME']>=1000, 1, 0)
    chk_volum_summary = chk_volum \
                        .groupby(['OVER_1000']) \
                        .size() \
                        .reset_index(name='COUNT')


# %% Suspend ------


def get_google_treneds(begin_date=None, end_date=None, 
                       normalize=True):
    
    global market
    print('get_google_treneds - 增加和get_stock_info一樣的daily backup')

    # begin_date = 20210101
    # end_date = 20210710
    
    
    # 因為Google Trends有假日的資料，所以應該要作平均
    # 全部lag 1
    # 存檔

    # 避免shift的時候造成NA
    temp_begin = cbyz.date_cal(begin_date, -20, 'd')
    temp_end = cbyz.date_cal(end_date, 20, 'd')
    
    calendar = stk.get_market_calendar(begin_date=temp_begin, 
                                       end_date=temp_end, 
                                       stock_type=market)

    # Word List ......
    file_path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/2_Stock_Analysis/Resource/google_trends_industry.xlsx'
    file = pd.read_excel(file_path, sheet_name='words')
    
    file = file[(file['STOCK_TYPE'].str.contains(market)) \
                & (~file['WORD'].isna()) \
                & (file['REMOVE'].isna())]
    
    words_df = file.copy()
    words_df = words_df[['ID', 'WORD']]
    words = words_df['WORD'].unique().tolist()
    
    
    # Pytrends ......
    trends = cbyz.pytrends_multi(begin_date=begin_date, end_date=end_date, 
                                 words=words, chunk=180, unit='d', hl='zh-TW', 
                                 geo='TW')    
    
    trends = cbyz.df_date_simplify(df=trends, cols=['DATE']) \
                .rename(columns={'DATE':'WORK_DATE', 
                                 'VARIABLE':'WORD'})
    
    # Merge Data......
    main_data = calendar[['WORK_DATE', 'TRADE_DATE']]
    main_data = cbyz.df_cross_join(main_data, words_df)
    main_data = main_data.merge(trends, how='left', on=['WORK_DATE', 'WORD'])
    
    main_data['WORD_TREND'] = 'WORD_' + main_data['ID'].astype('str')
    main_data = cbyz.df_conv_na(df=main_data, cols='VALUE')
    
    main_data = main_data \
                .sort_values(by=['WORD_TREND', 'WORK_DATE']) \
                .reset_index(drop=True)
    
    
    # Trade Date ......
    # Here are some NA values
    main_data['NEXT_TRADE_DATE'] = np.where(main_data['TRADE_DATE']==1,
                                           main_data['WORK_DATE'],
                                           np.nan)
    
    main_data = cbyz.df_shift_fillna(df=main_data, loop_times=len(calendar), 
                                     group_by='WORD_TREND',
                                     cols='NEXT_TRADE_DATE', forward=False)
    
    main_data = main_data \
                .groupby(['NEXT_TRADE_DATE', 'WORD_TREND']) \
                .agg({'VALUE':'mean'}) \
                .reset_index() \
                .sort_values(by=['WORD_TREND', 'NEXT_TRADE_DATE'])

    # Add Lag. The main idea is that the news will be reflected on the stock 
    # price of tomorrow.                
    main_data, _ = cbyz.df_add_shift(df=main_data, cols='NEXT_TRADE_DATE', 
                                    shift=1, group_by=[], suffix='_LAG', 
                                    remove_na=False)
    
    main_data = main_data \
        .drop('NEXT_TRADE_DATE', axis=1) \
        .rename(columns={'NEXT_TRADE_DATE_LAG':'WORK_DATE'}) \
        .dropna(subset=['WORK_DATE'], axis=0) \
        .reset_index(drop=True)
    
    
    cbyz.df_chk_col_na(df=main_data)
    
    
    # ......
    if normalize:
        main_data, _, _, _ = cbml.df_normalize(df=main_data, cols='VALUE')
    
    
    # Pivot
    main_data = main_data.pivot_table(index='WORK_DATE',
                                    columns='WORD_TREND', 
                                    values='VALUE') \
                .reset_index()
                
                
                
    # Calculate MA ......
    
    # Bug
  # File "/Users/Aron/Documents/GitHub/Codebase_YZ/codebase_yz.py", line 2527, in df_add_ma
    # temp_main['variable'] = temp_main['variable'].astype(int)    
    
    
    global ma_values
    cols = cbyz.df_get_cols_except(df=main_data, except_cols=['WORK_DATE'])
    main_data, ma_cols = cbyz.df_add_ma(df=main_data, cols=cols, 
                                       group_by=[], 
                                       date_col='WORK_DATE', values=ma_values,
                                       wma=False)
    
                
                
    
    # Convert NA
    cols = cbyz.df_get_cols_except(df=main_data, except_cols=['WORK_DATE'])
    main_data = cbyz.df_conv_na(df=main_data, cols=cols)
    
    
    return main_data, cols


# %% Archive ------


def split_data():
    
    global model_data, model_x, var_y, id_keys, predict_date    
    global _symbols
    
    # Model Data ......
    cur_model_data = model_data[model_data['WORK_DATE']<predict_date[0]] \
                    .reset_index(drop=True)
    
    # Predict Data ......
    cur_predict_data = model_data[model_data['WORK_DATE']>=predict_date[0]]    


    # if len(SYMBOL) == 0:
    #     cur_model_data = model_data[model_data['WORK_DATE']<predict_date[0]] \
    #                     .reset_index(drop=True)
        
    #     # Predict Data ......
    #     cur_predict_data = model_data[model_data['WORK_DATE']>=predict_date[0]]
    # else:
    #     cur_model_data = model_data[(model_data['SYMBOL']==symbol) \
    #             & (model_data['WORK_DATE']<predict_date[0])] \
    #         .reset_index(drop=True)
    
    #     # Predict Data ......
    #     cur_predict_data = model_data[
    #         (model_data['SYMBOL']==symbol) \
    #             & (model_data['WORK_DATE']>=predict_date[0])]
        

    global X_train, X_test, y_train, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite

            
    # Predict            
    X_predict = cur_predict_data[model_x + id_keys]
    # X_predict = X_predict[X_predict['SYMBOL'].isin(SYMBOL)]
 
    
    # Traning And Test
    X = cur_model_data[model_x + id_keys]
    y = cur_model_data[var_y + id_keys]      
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    
    X_train_lite = X_train.drop(id_keys, axis=1)   
    X_test_lite = X_test.drop(id_keys, axis=1)   
    X_predict_lite = X_predict.drop(id_keys, axis=1)   
    
    
    # y_train_lite = y_train[var_y[0]]
    # y_test_lite = y_test[var_y[0]]    
    y_train_lite = []
    y_test_lite = []    
    
    for i in range(len(var_y)):
        y_train_lite.append(y_train[var_y[i]])
        y_test_lite.append(y_test[var_y[i]])        
    


def get_model(y_index, cv=2, dev=False, 
              load_model=False, export_model=True, path=None):
    
    print('Update, copy predict_and_tuning from order_forecast')
    
    global shift_begin, shift_end, data_begin, data_end
    global predict_begin, predict_end    
    global model_data, predict_date, model_x, var_y, norm_orig
   
    
    import xgboost as xgb    
    global SYMBOL, model_data, predict_date

    global X_train, y_train, X_test, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite
    

    y = var_y[y_index].lower()
    model_path = path_export + '/saved_model_' + y + '.sav'    
    
    if load_model and path != None:
        try:
            loaded_model = pickle.load(open(model_path, 'rb'))    
        except Exception as e:
            print(e)
        else:
            return loaded_model

    # 參數設定 ......
    if dev:
        model_params = {
            'random_foreest': {
                'model': RandomForestRegressor(),
                'params': {
                    'max_depth': [2, 3],
                }
            }
        }            
                    
        
    else:
        model_params = {
            
            'xbgoost': {
                'model': xgb.XGBRegressor(),
                'params': {
                    # 'n_estimators': [200, 250],
                    # 'gamma':[0, 0.5],
                    # 'max_depth':[4, 6],
                    # 'objective':['reg:tweedie', 'reg:squarederror']
                    'n_estimators': [200],
                    'gamma':[0],
                    'max_depth':[4],                
                },
            },  
            # 'random_foreest': {
            #     'model': RandomForestRegressor(),
            #     'params': {
            #         'max_depth': [2, 3],
            #     }
            # }
        }            
            
    # 自動測試 ......
    scores = []
    best_score = 0
    saved_model = None
    
    for model_name, mp in model_params.items():
        
        temp_model = GridSearchCV(mp['model'], mp['params'], cv=cv, 
                                  return_train_score=False)
        
        temp_model.fit(X_train_lite, y_train_lite[y_index])  
        
        # 保留分數最高的模型
        if temp_model.best_score_ > best_score:
            best_score = temp_model.best_score_
            saved_model = temp_model
        
        scores.append({
            'model': model_name,
            'best_score': temp_model.best_score_,
            'best_params': temp_model.best_params_
        })
        
    df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    df

    # 儲存模型 ......
    if export_model and path != None:
        pickle.dump(saved_model, open(model_path, 'wb'))
        
        print('get_model - tempraily remove.')
        # best_params = saved_model.best_params_
        # best_params = cbyz.dict_to_df(best_params)
        # best_params.to_csv(path + '/best_params' + y + '.csv', index=False)

    return saved_model


# ..............


# def predict(load_model=False, cv=2, dev=False):
    
#     global shift_begin, shift_end, data_begin, data_end
#     global predict_begin, predict_end    
#     global model_data, predict_date, model_x, var_y, norm_orig
    
#     split_data()
    
   
#     # Model ......
#     results = pd.DataFrame()
#     rmse = pd.DataFrame()    
#     features = pd.DataFrame()    

    
#     for i in range(0, len(var_y)):

#         model = get_model(y_index=i, cv=cv, load_model=load_model, 
#                           export_model=True, path=path_export, dev=dev)


#         # Feature Importance ......
#         features_new = {'FEATURES':list(X_train_lite.columns),
#                         'IMPORTANCE':model.best_estimator_.feature_importances_}
        
#         features_new = pd.DataFrame(features_new)            
#         features_new['Y'] = var_y[i]
#         features = features.append(features_new)        

    
#         # RMSE ......
#         preds_test = model.predict(X_test_lite)
#         rmse_new = np.sqrt(mean_squared_error(y_test_lite[i], preds_test))
#         rmse_new = pd.DataFrame(data=[rmse_new], columns=['RMSE'])
#         rmse_new['Y'] = var_y[i]
#         rmse = rmse.append(rmse_new)

    
#         # Results ......
#         preds = model.predict(X_predict_lite)
#         results_new = X_predict[id_keys].reset_index(drop=True)
#         results_new['VALUES'] = preds
#         results_new['Y'] = var_y[i]
#         results = results.append(results_new)        
    
    
#     # Organize ......
#     rmse = rmse.reset_index(drop=True)
    
#     # 
#     features = features \
#                 .sort_values(by='IMPORTANCE', ascending=False) \
#                 .reset_index(drop=True)
                
#     features.to_csv(path_export + '/features_' + exe_serial + '.csv',
#                     index=False)           
                
    
#     # results
#     results_pivot = results \
#                     .pivot_table(index=['SYMBOL', 'WORK_DATE'],
#                                  columns='Y',
#                                  values='VALUES') \
#                     .reset_index()
    
#     results = cbyz.df_normalize_restore(df=results_pivot, 
#                                         original=norm_orig)
    
    
#     return results, rmse, features




def predict_and_tuning(cv=5, load_model=False, export_model=True, path=None, 
                       fast=False):
    '''
    1. 這個function會匯出模型、scores和features，並且會在檔名加上系統時間，因此
       不會覆蓋檔案。
   2. 讀取模型的時候，會找檔名包含saved_model，且修改日期最新的檔案。
   3. 這支function會在多個專案中重複使用，因為可能會預測多個Y，所以把var_y視為
      list處理。

    Parameters
    ----------
    cv : TYPE, optional
        DESCRIPTION. The default is 5.
    load_model : TYPE, optional
        DESCRIPTION. The default is False.
    export_model : TYPE, optional
        DESCRIPTION. The default is True.
    path : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    # 模型套件
    import pickle
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor    
    
    # 預測用的全域變數
    global model_x, var_y
    global model_data
    global norm_orig


    # .........
    global exe_serial
    
    result = pd.DataFrame()
    features = pd.DataFrame()
    precision = pd.DataFrame()      
    
    scores = pd.DataFrame()
    best_params = pd.DataFrame()
    
    
    for i in range(len(var_y)):

        cur_y = var_y[i]
        suffix = cur_y.lower() + '_' + exe_serial        
        model_name = 'model_'+ cur_y.lower()
        new_model_name = 'saved_model_' + suffix        
        
        
        # 區分訓練集和測試集 ...
        train_data = model_data \
                    .dropna(subset=model_x + [cur_y], axis=0) \
                    .reset_index(drop=True)
        
        X = train_data[model_x]
        y = train_data[cur_y]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y)
    

        # 預測資料集 ......  
        X_predict = model_data[model_data[cur_y].isna()]
        X_predict = X_predict.dropna(subset=model_x, axis=0)    
        X_predict_lite = X_predict[model_x]


        # ...
        load_fail = True
    
        if load_model and path != None:
            files = cbyz.os_get_dir_list(path, level=0, extensions='sav', 
                                         remove_temp=True)    
            files = files['FILES']
            files = files[files['FILE_NAME'].str.contains(model_name)] 
            
            # 只保留修改時間最晚的檔案
            files['MAX_MODIFIED_TIME'] = files['MODIFIED_TIME'].max()
            files = files[files['MAX_MODIFIED_TIME']==files['MODIFIED_TIME']] \
                    .drop_duplicates(subset=['MODIFIED_TIME']) \
                    .reset_index(drop=True)
            try:
                model = pickle.load(open(files.loc[0, 'PATH'], 'rb'))    
            except Exception as e:
                print(e)
            else:
                print('get_model讀取儲存的模型 - ' + model_name)
                load_fail = False

        
        # 重新訓練模型 ......
        if load_fail:
            from sklearn.model_selection import GridSearchCV    
            
            # 參數設定 ...
            if fast:
                model_params = {
                    'xgboost': {
                        'model': xgb.XGBRegressor(),
                        'params': {
                            'n_estimators': [200],
                            'gamma':[0],
                            'max_depth':[4],
                            'objective':['reg:squarederror']
                        },
                    }
                }
                
            else:
                model_params = {
                    'xgboost': {
                        'model': xgb.XGBRegressor(),
                        'params': {
                            'n_estimators': [200, 400],
                            'gamma':[0],
                            'max_depth':[4, 6],
                            'objective':['reg:squarederror']
                            # 'objective':['reg:tweedie', 'reg:squarederror']
                        },
                    }
                    # ,
                    # 'random_forest': {
                    #     'model': RandomForestRegressor(),
                    #     'params': {
                    #         'max_depth': [2, 3],
                    #     }
                    # }
                }
            
            
            # 自動測試 ...
            scores_li = []
            best_score = 0
            model = None
            
            for model_name, mp in model_params.items():
                
                # Cross Validation
                temp_model = GridSearchCV(mp['model'], mp['params'], cv=cv, 
                                          return_train_score=False)
                temp_model.fit(X_train, y_train)
                
                scores_li.append({
                    'model': model_name,
                    'best_score': temp_model.best_score_,
                    'best_params': temp_model.best_params_
                })
                
                # 保留分數最高的模型
                if temp_model.best_score_ > best_score:
                    best_score = temp_model.best_score_
                    model = temp_model
                    
            
            # Record Scores
            new_scores = pd.DataFrame(scores_li, 
                                      columns=['MODEL', 'BEST_SCORE', 
                                               'BEST_PARAMS'])
            new_scores['Y'] = cur_y
            scores = scores.append(new_scores)
            
            
            # Record Parameters
            new_best_params = pd.DataFrame.from_dict(model.best_params_, 
                                                     orient='index',
                                                     columns=['VALUE'])
        
            new_best_params = new_best_params.reset_index() \
                                .rename(columns={'index':'KEY'})            
            
            new_best_params['Y'] = cur_y
            best_params = best_params.append(new_best_params)            
            

        # 儲存模型 ...
        if export_model and path != None:
            
            # Export Model
            pickle.dump(model, open(path + '/' + new_model_name + '.sav', 'wb'))

        # 預測 ......

        # Feature Importance ......
        new_features = {'FEATURES':list(X_train.columns),
                        'IMPORTANCE':model.best_estimator_.feature_importances_}
        
        new_features = pd.DataFrame(new_features)        
        new_features['Y'] = cur_y
        features = features.append(new_features)
    
        # RMSE ......
        preds_test = model.predict(X_test)
        new_precision = np.sqrt(mean_squared_error(y_test, preds_test))
        new_precision = pd.DataFrame(data=[new_precision], 
                                     columns=['PRECISION'])

        new_precision['Y'] = cur_y        
        precision = precision.append(new_precision)
    
        # Results ......
        preds = model.predict(X_predict_lite)
        new_result = X_predict[id_keys].reset_index(drop=True)
        new_result[cur_y] = preds
        new_result = cbml.df_normalize_restore(df=new_result, 
                                               original=norm_orig)
        
        if len(result) == 0:
            result = new_result.copy()
        else:
            result = result.merge(new_result, how='left', on=id_keys)
        
    
    result.to_csv(path + '/results_' + exe_serial + '.csv', index=False)
    features.to_csv(path + '/features_' + exe_serial + '.csv', index=False)
    precision.to_csv(path + '/precision_' + exe_serial + '.csv', index=False)        
    scores.to_csv(path + '/scores_' + exe_serial + '.csv', index=False)
    best_params.to_csv(path + '/best_params_' + exe_serial + '.csv', index=False)


    return result, precision



# %% Dev ------



def tw_fix_symbol_error():
    '''
    Fix the symbol error caused by csv file, which makes 0050 become 50.
    '''
    
    # Draft
    file = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/1_Data_Collection/2_TEJ/Export/ewprcd_data_2021-06-21_2021-06-30.csv'
    file = pd.read_csv(file)
    
    for i in range(3):
        file['SYMBOL'] = '0' + file['SYMBOL'] 


# %% Debug ------

def debug():
    

    chk = main_data[main_data['INDUSTRY_TRADE_VALUE_MA_6_LAG'].isna()]
    chk = chk[['SYMBOL']].drop_duplicates()
    
    stock_info = stk.tw_get_stock_info(path=path_temp)
    chk_main = chk.merge(stock_info, how='left', on=['SYMBOL'])


# %% Execution ------


if __name__ == '__main__':
    
    symbols = [2520, 2605, 6116, 6191, 3481, 2409, 2603]
    
    # predict_result, precision = \
    #     master(param_holder=param_holder,
    #            predict_begin=20211209,
    #             _volume_thld=1000, export_model=True, load_model=False,
    #             fast=True)
        
        
    sam.master(param_holder=param_holder,
               predict_begin=20211201,
               load_model=False,
               threshold=30000)        
        

