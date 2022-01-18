#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
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


import pickle


host = 2
host = 0


# Path .....
if host == 0:
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis'
    path_dcm = '/Users/aron/Documents/GitHub/Stock_Forecast/1_Data_Collection'
    
elif host == 2:
    path = '/home/jupyter//2_Stock_Analysis'
    path_dcm = '/home/jupyter//1_Data_Collection'


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20220118',
                 r'/home/jupyter/Arsenal/20220118',
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import codebase_ml as cbml
import arsenal as ar
import arsenal_stock as stk
# import ultra_tuner_v0_26 as ut
# import ultra_tuner_v0_261 as ut
# import ultra_tuner_v0_27_dev as ut
import ultra_tuner_v0_29_dev as ut

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
    
    
    global symbol, market
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
    
    if len(symbol) == 0:
        market_data_raw = \
            stk.get_data(
                data_begin=loc_begin, 
                data_end=data_end, 
                market=market, 
                symbol=[],
                adj=True,
                price_change=True, 
                price_limit=True, 
                trade_value=trade_value,
                restore=True
                )
    else:
        market_data_raw = \
            stk.get_data(
                data_begin=loc_begin,
                data_end=data_end, 
                market=market, 
                symbol=symbol,
                adj=True,
                price_change=True, 
                price_limit=True,
                trade_value=trade_value,
                restore=True
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
        if  min_value < -0.1:
            print(msg_min)
            
        if  max_value >= 0.1:
            print(msg_max)
        

    # Exclude Low Volume symbol ......
    market_data_raw = select_symbols()
    
    
    # First Trading Day ......
    # - This object will be used at the end of get_model_data
    global first_trading_day
    first_trading_day = market_data_raw[['SYMBOL', 'WORK_DATE']] \
            .sort_values(by=['SYMBOL', 'WORK_DATE'], ascending=True) \
            .drop_duplicates(subset=['SYMBOL']) \
            .rename(columns={'WORK_DATE':'FIRST_TRADING_DAY'})
            

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
    
    # if support_resist:
    #     # Check，確認寫法是否正確
    #     # print('add_support_resistance - days == True時有bug，全部數值一樣，導致
    #     # 沒辦法標準化？')
    #     global data_period
    #     market_data_raw, _ = \
    #         stk.add_support_resistance(df=market_data_raw, cols='CLOSE',
    #                                    rank_thld=int(data_period * 2 / 360),
    #                                    prominence=4, days=False)


    # Predict Symbols ......
    # 1. Prevent some symbol excluded by select_symbols(), but still
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
    
    calendar_proc, _, _, _ = \
        cbml.ml_data_process(
            df=calendar_proc, 
            ma=False, scale=True, lag=False,
            group_by=[],
            cols=[], 
            except_cols=['WORK_DATE', 'TRADE_DATE'],
            cols_mode='equal',
            drop_except=[],
            date_col='WORK_DATE',
            scale_method=1,
            ma_values=ma_values, 
            lag_period=predict_period
            )
        
    
    # Merge As Main Data ......
    global main_data_frame, main_data_frame_calendar
    main_data_frame = cbyz.df_cross_join(symbol_df, calendar_proc)
    
    # Remove Untrading Day
    main_data_frame = \
        main_data_frame[
            (main_data_frame['TRADE_DATE']>=1) \
            & (main_data_frame['WORK_DATE']<=predict_date[-1])] \
        .drop('TRADE_DATE', axis=1)

    market_data_raw = main_data_frame \
        .merge(market_data_raw, how='left', on=['SYMBOL', 'WORK_DATE'])

    
    # Organize
    main_data_frame = main_data_frame[['WORK_DATE', 'SYMBOL']]
    main_data_frame_calendar = main_data_frame[['WORK_DATE']] \
                                .drop_duplicates() \
                                .sort_values(by='WORK_DATE') \
                                .reset_index(drop=True)
    

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
    
    global symbol
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
    # msg = '如果y是OHLC的價格的話，這一段的normalize邏輯有點怪'
    # assert 'CLOSE' not in var_y, msg


    # 應要先獨立把y的欄位標準化，因為這一段不用MA，但後面都需要
    # - 這裡的method要用1，如果用2的話，mse會變成0.8
    global y_scaler
    loc_main, norm_orig, y_scaler = \
        cbml.df_scaler(
            df=loc_main,
            cols=var_y,
            show_progress=False,
            method=1
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
    
        loc_main, ratio_cols, _, _ = \
            cbml.ml_data_process(df=loc_main,
                                 ma=True, scale=True, lag=True, 
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
    
    loc_main, _, _, _ = \
        cbml.ml_data_process(df=loc_main, 
                              ma=True, scale=True, lag=True,
                              date_col='WORK_DATE',
                              group_by=['SYMBOL'],
                              cols=[], 
                              except_cols=except_cols,
                              drop_except=var_y,
                              cols_mode='equal',
                              ma_values=ma_values, 
                              lag_period=predict_period
                              )
        
    # except_cols = ['SYMBOL', 'WORK_DATE', 'YEAR', 
    #                'MONTH', 'WEEKDAY', 'WEEK_NUM'] \
    #                 + ratio_cols
    
    # loc_main, temp_cols, _ = \
    #     cbml.ml_data_process(df=loc_main, 
    #                          ma=True, scale=True, lag=True,
    #                          date_col='WORK_DATE',
    #                          group_by=[],
    #                          cols=[], 
    #                          except_cols=except_cols,
    #                          drop_except=var_y,
    #                          cols_mode='equal',
    #                          ma_values=ma_values, 
    #                          lag_period=predict_period
    #                          )        
        
        

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
        
        total_trade, _, _, _ = \
            cbml.ml_data_process(df=total_trade, 
                                 ma=True, scale=True, lag=True, 
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
    
    stock_info, _, _, _ = \
        cbml.ml_data_process(df=stock_info, 
                             ma=False, scale=True, lag=False,
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

    global debug3
    debug3 = loc_main.copy()
    

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
                       
        
        industry_data, _, _, _ = \
             cbml.ml_data_process(df=industry_data, 
                                  ma=True, scale=True, lag=True,
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
    # chk_min_max = cbyz.df_chk_col_min_max(df=loc_main)
    # chk_min_max = chk_min_max[chk_min_max['COLUMN']!='WORK_DATE']
    # chk_min_max = chk_min_max[(chk_min_max['MIN_VALUE']<0) \
    #                           | (chk_min_max['MAX_VALUE']>1)]
        
    # assert len(chk_min_max) == 0, 'chk_min_max failed'
    
        
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
    file1.loc[:, 'WORK_DATE'] = '20' + file1['WORK_DATE']
    file1.loc[:, 'WORK_DATE'] = file1['WORK_DATE'].str.replace("'", "")
    file1.loc[:, 'WORK_DATE'] = file1['WORK_DATE'].str.replace("/", "")
    file1 = cbyz.df_conv_col_type(df=file1, cols='WORK_DATE', to=np.int32)
    file1 = cbyz.df_conv_col_type(df=file1, cols='EX_DIVIDENDS_PRICE',
                                  to='float')    
    file1.loc[:, 'SALE_MON_DATE'] = 1
    file1 = cbyz.df_conv_na(df=file1, 
                            cols=['EX_DIVIDENDS_PRICE', 'SALE_MON_DATE'])

    # 填息
    file2 = file_raw[['SYMBOL', 'EX_DIVIDENDS_DONE']]
    file2.columns = ['SYMBOL', 'WORK_DATE']
    file2.loc[:, 'WORK_DATE'] = '20' + file2['WORK_DATE']
    file2.loc[:, 'WORK_DATE'] = file2['WORK_DATE'].str.replace("'", "")
    file2.loc[:, 'WORK_DATE'] = file2['WORK_DATE'].str.replace("/", "")
    file2 = cbyz.df_conv_col_type(df=file2, cols='WORK_DATE', to=np.int32)
    file2.loc[:, 'EX_DIVIDENDS_DONE'] = 1
    
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


def get_model_data(industry=True, trade_value=True, load_file=False):
    
    
    global shift_begin, shift_end, data_begin, data_end, ma_values
    global predict_date, predict_period, calendar, calendar_lite
    global main_data_frame, main_data_frame_calendar, market_data_raw
    global symbol
    global var_y
    global params, error_msg
    global id_keys
    global main_data
    
    
    if load_file:
    
        main_data_file = path_temp + '/model_data.csv'
        model_x_file = path_temp + '/model_x.csv'
        norm_orig_file = path_temp + '/norm_orig.csv'
        

        # 為了避免跨日的問題，多計算一天
        today = cbyz.date_get_today()
        
        main_data_mdate = cbyz.os_get_file_modify_date(main_data_file)
        main_data_mdate = cbyz.date_cal(main_data_mdate, 1, 'd')
        main_data_diff = cbyz.date_diff(today, main_data_mdate, absolute=True)        

        model_x_mdate = cbyz.os_get_file_modify_date(model_x_file)
        model_x_mdate = cbyz.date_cal(model_x_mdate, 1, 'd')
        model_x_diff = cbyz.date_diff(today, model_x_mdate, absolute=True)
        
        norm_mdate = cbyz.os_get_file_modify_date(norm_orig_file)
        norm_mdate = cbyz.date_cal(norm_mdate, 1, 'd')
        norm_diff = cbyz.date_diff(today, norm_mdate, absolute=True)        


        # Ensure files were saved recently
        if main_data_diff <= 2 and  model_x_diff <= 2 and norm_diff <= 2:
            try:
                main_data = pd.read_csv(main_data_file)
                model_x = cbyz.li_read_csv(model_x_file)
                norm_orig = pd.read_csv(norm_orig_file)
            except Exception as e:
                print(e)
            else:
                return model_data, model_x, norm_orig                
    

    # Check ......
    msg = 'get_model_data - predict_period is longer than ma values, ' \
            + 'and it will cause na.'
    assert predict_period <= min(ma_values), msg


    # Symbols ......
    symbol = cbyz.conv_to_list(symbol)
    symbol = cbyz.li_conv_ele_type(symbol, 'str')


    # Market Data ......
    # market_data_raw
    get_market_data_raw(trade_value=trade_value)
    gc.collect()
    
    
    # Load Historical Data ......
    global main_data_raw
    main_data_raw, norm_orig = \
        sam_load_data(industry=industry, trade_value=trade_value) 

    main_data = main_data_raw.copy()
    cbyz.df_chk_col_na(df=main_data_raw)
    
    
    # TODC Shareholdings Spread ......
    # sharehold = stk.tdcc_get_sharehold_spread(shift_begin, end_date=None,
    #                                           local=local) 
    
    # main_data = main_data.merge(sharehold, how='left', 
    #                           on=['SYMBOL', 'WORK_DATE'])      


    # Buffett Indicator ......
    if market == 'tw':
        
        buffett_indicator = \
            stk.cal_buffett_indicator(end_date=predict_date[0])
            
        buffett_indicator = \
            main_data_frame_calendar \
                .merge(buffett_indicator, how='left', on='WORK_DATE')  
    
        buffett_indicator, _, _, _ = \
               cbml.ml_data_process(df=buffett_indicator, 
                                    ma=False, scale=True, lag=True, 
                                    group_by=[], 
                                    cols=[],
                                    except_cols=['WORK_DATE'],
                                    drop_except=[],
                                    cols_mode='equal',
                                    date_col='WORK_DATE',
                                    ma_values=[], 
                                    lag_period=predict_period
                                    ) 
      
        cols = cbyz.df_get_cols_except(df=buffett_indicator,
                                       except_cols=['WORK_DATE'])
        main_data = main_data \
                    .merge(buffett_indicator, how='left', on=['WORK_DATE'])
                


    # Government Invest ......
    gov_invest = stk.od_tw_get_gov_invest(path=path_resource)
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
      
    daily_close = cbyz.df_fillna(df=daily_close, cols='CLOSE', 
                                 sort_keys=['SYMBOL', 'WORK_DATE'],
                                 method='both')
                
    
    # 除權息 ...
    sale_mon_data1, sale_mon_data2 = get_sale_mon_data()
    
    # Data 1 - 除權息日期及價錢 ...
    sale_mon_data1 = daily_close \
        .merge(sale_mon_data1, how='left', on=['WORK_DATE', 'SYMBOL'])
    
    sale_mon_data1['EX_DIVIDENDS_PRICE'] = \
        sale_mon_data1['EX_DIVIDENDS_PRICE'] / sale_mon_data1['CLOSE']    
        
    sale_mon_data1 = sale_mon_data1.drop('CLOSE', axis=1)
    sale_mon_data1 = cbyz.df_conv_na(df=sale_mon_data1, 
                                     cols=['EX_DIVIDENDS_PRICE', 
                                           'SALE_MON_DATE'])
    
    sale_mon_data1, _, _, _ = \
             cbml.ml_data_process(df=sale_mon_data1, 
                                  ma=False, scale=True, lag=False, 
                                  group_by=['SYMBOL'], 
                                  cols=['EX_DIVIDENDS_PRICE', 'SALE_MON_DATE'],
                                  except_cols=[],
                                  drop_except=[],
                                  cols_mode='equal',
                                  date_col='WORK_DATE',
                                  ma_values=ma_values, 
                                  lag_period=predict_period
                                  ) 
    
    # Data 2 - 填息 ...
    sale_mon_data2, _, _, _ = \
        cbml.ml_data_process(df=sale_mon_data2, 
                             ma=False, scale=True, lag=False, 
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
    if market == 'tw':
        ewtinst1c_raw = stk.tej_get_ewtinst1c(begin_date=shift_begin, 
                                              end_date=None, 
                                              symbol=symbol,
                                              trade=True)
        
        ewtinst1c = main_data_frame \
                    .merge(ewtinst1c_raw, how='left', on=['WORK_DATE', 'SYMBOL']) \
                    .merge(symbol_df, on=['SYMBOL'])
    
        cols = cbyz.df_get_cols_except(df=ewtinst1c,
                                       except_cols=['WORK_DATE', 'SYMBOL']) 
        
        ewtinst1c = cbyz.df_fillna(df=ewtinst1c, cols=cols, 
                                   sort_keys=['SYMBOL', 'WORK_DATE'], 
                                   group_by=[], method='ffill')                    
                    
            
        # 獲利率HROI、Sell、Buy用globally normalize，所以要分兩段
        hroi_cols = cbyz.df_get_cols_contains(
            df=ewtinst1c, 
            string=['_HROI', '_SELL', '_BUY']
            )
        
        # Keep Needed Symbols Only
        ewtinst1c, cols_1, _, _ = \
            cbml.ml_data_process(df=ewtinst1c, 
                                 ma=True, scale=True, lag=True, 
                                 group_by=[],
                                 cols=hroi_cols, 
                                 except_cols=[],
                                 drop_except=[],
                                 cols_mode='contains',
                                 date_col='WORK_DATE',
                                 ma_values=ma_values, 
                                 lag_period=predict_period
                                 ) 
    
        ewtinst1c, cols_2, _, _ = \
            cbml.ml_data_process(df=ewtinst1c, 
                                 ma=True, scale=True, lag=True, 
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
        
        print('ewtinst1c Check')
        print('Check - 全部填NA是否合理？')
        main_data = cbyz.df_conv_na(df=main_data, cols=cols_1 + cols_2)
        cbyz.df_chk_col_na(df=main_data, except_cols=var_y, mode='stop')


    # 月營收資料表 ......
    # 1. 當predict_date=20211101，且為dev時, 造成每一個symbol都有na，先移除
    # 1. 主要邏輯就是顯示最新的營收資料
    if market == 'tw':
        
        msg = '''Bug - sam_tej_get_ewsale，在1/18 23:00跑1/19時會出現chk_na error，但1/19 00:00過後
        再跑就正常了
        '''
        print(msg)
        
        ewsale = sam_tej_get_ewsale(begin_date=shift_begin)
        main_data = main_data \
                    .merge(ewsale, how='left', on=['SYMBOL', 'WORK_DATE'])      
    
        cbyz.df_chk_col_na(df=main_data, except_cols=var_y, mode='stop')
    
    
    # 財務報表
    # - 現在只用單季，需確認是否有缺漏
    print('財務報表現在只用單季，需確認是否有缺漏')
    # financial_statement = sam_tej_get_ewifinq()
    
    
    
    # 指數日成本 ......
    # - Increase prediction time a lot, but not increase mape obviously.
    # ewiprcd = stk.tej_get_ewiprcd()
    # main_data = main_data.merge(ewiprcd, how='left', on=['WORK_DATE'])  


    # Pytrends Data ......
    # - Increase prediction time a lot, and made mape decrease.
    # - Pytrends已經normalize過後才pivot，但後面又normalize一次
    # pytrends, pytrends_cols = get_google_treneds(begin_date=shift_begin, 
    #                                               end_date=data_end, 
    #                                               scale=True, 
    #                                               stock_type=stock_type, 
    #                                               local=local)
    
    # main_data = main_data.merge(pytrends, how='left', on=['WORK_DATE'])      


    # 台股加權指數 TW Index
    tw_index = sam_od_tw_get_index(begin_date=shift_begin, 
                                   end_date=predict_date[-1])
    
    main_data = main_data.merge(tw_index, how='left', on=['WORK_DATE'])
    
    
    # S&P 500 ......
    snp, cols = sam_od_us_get_snp_data(begin_date=shift_begin)
    main_data = main_data.merge(snp, how='left', on=['WORK_DATE'])
    
    
    # COVID-19 ......
    if market == 'tw':
        covid19, _ = cbyz.get_covid19_data()
        covid19 = main_data_frame_calendar \
                    .merge(covid19, how='left', on=['WORK_DATE'])
    
        covid19, covid19_cols, _, _ = \
                    cbml.ml_data_process(df=covid19, 
                                         ma=True, scale=True, lag=True, 
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


    # Remove date before first_trading_day
    # - 由於main_data_frame是用cross_join，所以會出現listing前的日期，但這個步驟要等到
    #   最後才執行，否則在合併某些以月或季為單位的資料時會出現NA
    global first_trading_day
    main_data = main_data \
        .merge(first_trading_day, how='left', on=['SYMBOL'])
    
    main_data = main_data[
        main_data['WORK_DATE']>=main_data['FIRST_TRADING_DAY']] \
        .drop('FIRST_TRADING_DAY', axis=1)


    # Check NA ......
    global hist_df, predict_df
    hist_df = main_data[main_data['WORK_DATE']<predict_date[0]]
    
    
    # Debug ......
    msg = ("get_model_data - 把normalize的group_by拿掉後，這個地方會出錯，但"
           "只有一筆資料有問題，暫時直接drop")
    print(msg)
    # hist_df.to_csv(path_temp + '/debug_hist_df.csv', index=False)
    cbyz.df_chk_col_na(df=hist_df, mode='stop')
    
    
    # Predict有NA是正常的，但NA_COUNT必須全部都一樣
    global chk_predict_na
    predict_df = main_data[main_data['WORK_DATE']>=predict_date[0]]
    chk_predict_na = cbyz.df_chk_col_na(df=predict_df, mode='alert')
    
    
    min_value = chk_predict_na['NA_COUNT'].min()
    max_value = chk_predict_na['NA_COUNT'].max()    
    assert min_value == max_value, 'All the NA_COUNT should be the same. '
    

    # Check min max ......
    # global chk_min_max
    # chk_min_max = cbyz.df_chk_col_min_max(df=main_data)
    
    # chk_min_max = \
    #     chk_min_max[(~chk_min_max['COLUMN'].isin(id_keys)) \
    #                 & ((chk_min_max['MIN_VALUE']<0) \
    #                    | (chk_min_max['MAX_VALUE']>1))]
    
    # assert len(chk_min_max) == 0, 'get_model_data - normalize error'


    # Export Model ......
    main_data.to_csv(path_temp + '/model_data.csv', index=False)
    cbyz.li_to_csv(model_x, path_temp + '/model_x.csv')
    norm_orig.to_csv(path_temp + '/norm_orig.csv', index=False)
        
    return main_data, model_x, norm_orig


# %% Variables ------


def sam_od_tw_get_index(begin_date, end_date):
    
    global ma_values, predict_period
    global main_data_frame_calendar
    
    loc_df = stk.od_tw_get_index(path=path_dcm)
    
    loc_df = main_data_frame_calendar \
                .merge(loc_df, how='left', on='WORK_DATE')    
    
    # Process
    loc_df, cols, _, _ = \
        cbml.ml_data_process(df=loc_df, 
                             ma=True, scale=True, lag=True, 
                             group_by=[],
                             cols=[], 
                             except_cols=['WORK_DATE'],
                             drop_except=[],
                             cols_mode='equal',
                             date_col='WORK_DATE',
                             ma_values=ma_values, 
                             lag_period=predict_period
                             )        
        
    loc_df = loc_df[(loc_df['WORK_DATE']>=begin_date) \
                    & (loc_df['WORK_DATE']<=end_date)]
        
    return loc_df


def sam_od_us_get_snp_data(begin_date):
    
    
    global ma_values, predict_period, predict_date
    global main_data_frame_calendar
    
    loc_df = stk.od_us_get_snp_data(daily_backup=True, path=path_temp)
    loc_df = loc_df.rename(columns={'WORK_DATE':'WORK_DATE_ORIG'})
    
    
    # Handle Time Lag
    loc_df = cbyz.df_date_cal(df=loc_df, amount=-1, unit='d',
                              new_cols='WORK_DATE',
                              cols='WORK_DATE_ORIG')
    
    loc_df = loc_df.drop('WORK_DATE_ORIG', axis=1)


    # Fillna .......
    # 1. 因為美股的交易時間可能和台灣不一樣，包含特殊節日等，為了避免日期無法對應，用fillna
    #    補上完整的日期
    # 2. 先執行fillna再ml_data_process比較合理
    loc_calendar = cbyz.date_get_calendar(begin_date=begin_date, 
                                          end_date=predict_date[-1])
    loc_calendar = loc_calendar[['WORK_DATE']]
    
    
    loc_df = loc_calendar.merge(loc_df, how='left', on='WORK_DATE')
    cols = cbyz.df_get_cols_except(df=loc_df, except_cols='WORK_DATE')
    loc_df = cbyz.df_fillna(df=loc_df, cols=cols, sort_keys='WORK_DATE', 
                            group_by=[], method='ffill')

    # Process
    loc_df, cols, _, _ = \
        cbml.ml_data_process(df=loc_df, 
                             ma=True, scale=True, lag=True, 
                             group_by=[],
                             cols=[], 
                             except_cols=['WORK_DATE'],
                             drop_except=[],
                             cols_mode='equal',
                             date_col='WORK_DATE',
                             ma_values=ma_values, 
                             lag_period=predict_period
                             )    
        
    # Fillter the target period        
    if begin_date != None:
        loc_df = loc_df[(loc_df['WORK_DATE'])>=begin_date]
        
        
    # 因為傳進來的begin_date是alert，所以有na是正常的
    # cbyz.df_chk_col_na(df=loc_df, mode='alert')
    # cbyz.df_chk_col_min_max(df=loc_df)
        
    return loc_df, cols



def sam_tej_get_ewsale(begin_date):

    global main_data_frame, symbol
    loc_df = stk.tej_get_ewsale(begin_date=begin_date, end_date=None, 
                                symbol=symbol, fill=True, host=host)
    
    
    # Merge will cause NA, so it must to execute df_fillna
    loc_df = main_data_frame \
        .merge(loc_df, how='left', on=['SYMBOL', 'WORK_DATE'])
    
    
    loc_df = cbyz.df_fillna(df=loc_df, cols=['D0001'], 
                               sort_keys=['SYMBOL', 'WORK_DATE'], 
                               group_by=['SYMBOL'], method='ffill')                    
                    
    loc_df, cols, _, _ = \
        cbml.ml_data_process(df=loc_df, 
                              ma=False, scale=True, lag=False, 
                              group_by=['SYMBOL'],
                              cols=['D0001'], 
                              except_cols=[],
                              cols_mode='equal',
                              drop_except=[],
                              ma_values=ma_values, 
                              lag_period=predict_period)    
    
    return loc_df


def sam_tej_get_ewifinq():
    
    
    loc_df = stk.tej_get_ewifinq(path=path_dcm, fill_date=True)
    
    return loc_df



# %% Master ------

def master(param_holder, predict_begin, export_model=True,
           threshold=30000, bt_index=0, load_data=False):
    '''
    主工作區
    '''
    
    # date_period為10年的時候會出錯
    
    # v2.00
    # - Add Ultra_Tuner
    # - Fix Ex-dividend issues
    # - Rename stock_type and stock_symbol
    # - Set df_fillna with interpolate method in arsenal_stock
    # v2.01
    # - Fix Date Issues
    # v2.02
    # - Add correlations detection > Done
    # - Fix tw_get_stock_info_twse issues
    # v2.03
    # - Add S&P 500 data
    # v2.04
    # - Add 台股指數
    # - Fix terrible date lag issues in get_model_data - Done
    # v2.041
    # - Update for new modules
    # - Export Model Data
    # v2.05
    # - Add Financial Statements

    # v2.07
    # - Remove group_by paramaters when normalizing
    # - Fix bug after removing group_by params of normalizing
    # - Add GDP and Buffett Indicator
    
    # v2.09
    # - Update the calculation method of trade value as mean of high and low > Done
    # - Add load data feature for get_model_data > Done
    # - Update cbml and df_scaler > Done
    # - Remove df_chk_col_min_max > Done


    # v2.10
    # - Rename symbols as symbol - Done
    # - Add symbol params to ewifinq - Done
    # - Update cbml for df_scaler
    
    
    # Update
    # Bug - sam_tej_get_ewsale，在1/18 23:00跑1/19時會出現chk_na error，但1/19 00:00過後
    #       再跑就正常。end_date應該要改成data_begin, 這個問題應該是today比data_begin少一天    
    # - Add financial_statement
    #   > 2021下半年還沒更新，需要改code，可以自動化更新並合併csv
    # - Test price as Y


    # Note
    # 1. 20220107 v2.06 - 原本在normalize的時候，會group by symbol，讓每一檔都和自己
    #   比較，否則高價股和低價股比感覺很虧。這個版本試著把sam_load_data中的group by
    #   改成[]。經測試過後，R2差不多，所以保留新的版本，應該可以提高計算速度。
    
    
    # - 確認TEJ財務報表的資料會不會自動更新
    # - Fix Support and resistant
    
    
    # - Add support_resist
    # - select_symbols用過去一周的總成交量檢查
    # - Short term model and long term model be overwirtted
    # 以close price normalize，評估高價股和低價股
    

    global version
    version = 2.10


    # Tracking
    # get_model_data - 把normalize的group_by拿掉後，這個地方會出錯，暫時直接drop
    
    
    # Optimization
    # - Make progarm can recover automatically. If detect na rows, then throw 
    #   message and delete na columns
    # - Combine cbyz >> detect_cycle.py, including support_resistance and 
    #    season_decompose
    # - Add 流通股數 from ewprcd
    # - Fix week_num, because it may be useful to present seasonaility
    # - 合併Yahoo Finance和TEJ的market data，兩邊都有可能缺資料。現在的方法是用interpolate，
    #   但如果begin_date剛好缺值，這檔股票就會被排除
    # - 在data中多一個欄位標註新股，因為剛上市的時候通常波動較大
    
    
    # Bug
    # 1. get_sale_mon_data - bug - 目前只有到2018
    # 2. Add covid-19 daily backup
    # 3. 在get_market_data_raw中, OPEN_CHANGE_RATIO min_value 
    #    is -0.8897097625329815，暫時移除assert
    # 5. 有些symbole會不見，像正隆
    # 2. pred price和Last price都有錯置的問題，應該是stk有bug, df_fillna的interpolate
    # 6. Handle week_num cyclical issues
    #    https://towardsdatascience.com/how-to-handle-cyclical-data-in-machine-learning-3e0336f7f97c
    
    
    # -NA issues, replace OHLC na in market data function, and add replace 
    # na with interpolation. And why 0101 not excluded?



    # Optimization .....
    # - Add week
    # 建長期投資的基本面模型
    # 5. print('add_support_resistance - days == True時有bug，全部數值一樣，導致沒辦法標準化？')
    # 6. 加上PRICE_CHANGE_ABS / PRICE_CHANGE加上STD
    # 6. 技術分析型態學
    # 8. Update CBYZ and auto-competing model
    # 9. 上櫃的也要放
    # 10. buy_signal to int
    # Update - 20211220，暫時在get_tw_index中呼叫update_tw_index，但這個方式可能會漏資料
    
    
    global bt_last_begin, data_period, predict_period, long
    global debug, dev
    global symbol, ma_values, volume_thld, market


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
    symbol = holder['symbol'][0]   
    ma_values = holder['ma_values'][0]   
    
    # Modeling
    predict_period = holder['predict_period'][0]
    long = holder['long'][0]   
    kbest = holder['kbest'][0]
    cv = holder['cv'][0]
    
    # Program
    debug = holder['debug'][0]
    
    
    # export_model = False
    # dev = True
    # threshold = 20000
    # predict_begin=20211209
    
    
    global exe_serial
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
                               shift=-(max(ma_values) * 2))
                
    # ......
    global model_data
    global model_x, var_y, id_keys
    global norm_orig
        
    
    var_y = ['OPEN_CHANGE_RATIO', 'HIGH_CHANGE_RATIO',
              'LOW_CHANGE_RATIO', 'CLOSE_CHANGE_RATIO']

    # var_y = ['OPEN', 'HIGH',
    #          'LOW', 'CLOSE']    
    
    id_keys = ['SYMBOL', 'WORK_DATE']    
    
    
    if load_data:
        try:
            pass
            # model_data = pd.read_csv(path_temp + '/model_data.csv')
            # model_x.to_csv(path_temp + '/model_x.csv', index=False)
            # norm_orig.to_csv(path_temp + '/norm_orig.csv', index=False)            
        except Exception as e:
            print(e)
    

    # 20210707 - industry可以提高提精準，trade_value會下降
    global y_scaler
    model_data, model_x, scale_orig = \
        get_model_data(industry=industry, 
                       trade_value=trade_value)
    
    
    
    # Training Model ......
    import xgboost as xgb
    # from sklearn.model_selection import GridSearchCV    
    # from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import SGDRegressor    
    
    if len(symbol) > 0 and len(symbol) < 10:
        model_params = [{'model': LinearRegression(),
                         'params': {
                             'normalize': [True, False],
                             }
                         }]         
    else:
        
        # eta 0.01、0.03的效果都很差，目前測試0.08和0.1的效果較佳
        
        # # Change Ratio
        # model_params = [
        #                 {'model': LinearRegression(),
        #                   'params': {
        #                       'normalize': [True, False],
        #                       }
        #                   },
        #                 {'model': xgb.XGBRegressor(),
        #                  'params': {
        #                     # 'n_estimators': [200],
        #                     'eta': [0.1],
        #                     # 'eta': [0.08, 0.1],
        #                     'min_child_weight': [1],
        #                      # 'min_child_weight': [0.5, 1],
        #                     'max_depth':[8],
        #                      # 'max_depth':[6, 8, 12],
        #                     'subsample':[1]
        #                   }
        #                 },
        #                 # {'model': SGDRegressor(),
        #                 #   'params': {
        #                 #       # 'max_iter': [1000],
        #                 #       # 'tol': [1e-3],
        #                 #       # 'penalty': ['l2', 'l1'],
        #                 #       }                     
        #                 #   }
        #                ] 

        
        # Price
        model_params = [
                        {'model': LinearRegression(),
                          'params': {
                              'normalize': [True, False],
                              }
                          },
                        {'model': xgb.XGBRegressor(),
                         'params': {
                            # 'n_estimators': [200],
                            'eta': [0.1],
                            # 'eta': [0.5, 0.7],
                            'min_child_weight': [0.8],
                             # 'min_child_weight': [0.5, 1],
                            'max_depth':[10],
                             # 'max_depth':[6, 8, 12],
                            'subsample':[1]
                          }
                        },
                        # {'model': SGDRegressor(),
                        #   'params': {
                        #       # 'max_iter': [1000],
                        #       # 'tol': [1e-3],
                        #       # 'penalty': ['l2', 'l1'],
                        #       }                     
                        #   }
                       ]         
        
        
        
        
    # 1. 如果selectkbest的k設得太小時，importance最高的可能都是industry，導致同產業
    #    的預測值完全相同
    global pred_result, pred_scores, pred_params, pred_features
    
    long_suffix = 'long' if long else 'short'
    compete_mode = compete_mode if bt_index == 0 else 0
    
    
    for i in range(len(var_y)):
        
        cur_y = var_y[i]
        remove_y = [var_y[j] for j in range(len(var_y)) if j != i]
        
        tuner = ut.Ultra_Tuner(id_keys=id_keys, y=cur_y, 
                               model_type='reg', suffix=long_suffix,
                               compete_mode=compete_mode,
                               train_mode=train_mode, 
                               path=path_temp)
        
        # 排除其他y，否則會出錯
        cur_model_data = model_data.drop(remove_y, axis=1)
        
        return_result, return_scores, return_params, return_features, \
                log_scores, log_params, log_features = \
                    tuner.fit(data=cur_model_data, model_params=model_params,
                              k=kbest, cv=cv, threshold=threshold, 
                              scale_orig=[],
                              export_model=True, export_log=True)
                 
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


    # Inverse Scale
    pred_result = pred_result[id_keys + var_y]
    y_inverse = pred_result[var_y]
    y_inverse = y_scaler.inverse_transform(y_inverse)
    y_inverse = pd.DataFrame(y_inverse, columns=var_y)
    
    pred_result_inverse = pd.concat([pred_result[id_keys], y_inverse],
                                    axis=1)
        
        
    # Upload to Google Sheet
    if predict_begin == bt_last_begin:
        stk.write_sheet(data=pred_features, sheet='Features')

    # Bug - 20220116 因為這裡的欄位會出現NORM_MEAN_x, NORM_STD_x, METHOD_x等df_scaler的欄位，
    # 所以暫時先用這個方式調整
    pred_result = pred_result[id_keys + var_y]          
    
    return pred_result_inverse, pred_scores, pred_params, pred_features




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
                       scale=True):
    
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
    main_data, _, _ = cbml.df_scaler(df=main_data, cols='VALUE')
    
    
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
        

def get_season(df):
    
    '''
    By Week, 直接在get_data中處理好
    
    '''
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    
    loc_df = df.copy() \
            .rename(columns={'STOCK_SYMBOL':'SYMBOL'}) \
            .sort_values(by=['SYMBOL', 'YEAR', 'WEEK_NUM']) \
            .reset_index(drop=True)
            
    # loc_df['x'] = loc_df['WORK_DATE'].apply(cbyz.ymd)

    unique_symbols = loc_df['SYMBOL'].unique().tolist()
    result = pd.DataFrame()

    for i in range(len(unique_symbols)):
        
        symbol = unique_symbols[i]
        temp_df = loc_df[loc_df['SYMBOL']==symbol]
        model_data = temp_df[['CLOSE']]

        decompose_result = seasonal_decompose(
            model_data,
            model="multiplicative",
            period=52,
            extrapolate_trend='freq'
            )
        
        trend = decompose_result.trend
        seasonal = decompose_result.seasonal
        residual = decompose_result.resid
        
        new_result = pd.concat([temp_df, trend, seasonal, residual], axis=1)
        result = result.append(new_result)
    
        decompose_result.plot()
        
    return result



def add_support_resistance(df, cols, rank_thld=10, prominence=4, days=True,
                           threshold=0.9, plot_data=False):

    '''
    1. Calculate suppport and resistance
    2. The prominence of each symbol is different, so it will cause problems
       if apply a static number. So use quantile as the divider.
    3. Update, add multiprocessing
    
    '''
    
    
    print('Bug, 回傳必要的欄位，不要直接合併整個dataframe，減少記憶體用量')
    
    from scipy.signal import find_peaks
    cols = cbyz.conv_to_list(cols)
    
    cols_support = [c + '_SUPPORT' for c in cols]
    cols_resistance = [c + '_RESISTANCE' for c in cols]
    return_cols = cols_support + cols_resistance
    
    
    # .......
    loc_df = df.copy().rename(columns={'STOCK_SYMBOL':'SYMBOL'})
    
    
    date_index = loc_df[['SYMBOL', 'WORK_DATE']]
    date_index = cbyz.df_add_rank(df=date_index, value='WORK_DATE',
                              group_by=['SYMBOL'], 
                              sort_ascending=True, 
                              rank_ascending=True,
                              rank_name='index',
                              rank_method='min', inplace=False)
    
    
    result_raw = pd.DataFrame()
    symbol = loc_df['SYMBOL'].unique().tolist()
    value_cols = []
    
    
    # Calculate ......
    for j in range(len(symbol)):
        
        symbol = symbol[j]
        temp_df = loc_df[loc_df['SYMBOL']==symbol].reset_index(drop=True)
    
        for i in range(len(cols)):
            col = cols[i]
            x = temp_df[col].tolist()
            x = np.array(x) # 轉成np.array後，在計算低點的時候才可以直接加負值
            
            # 計算高點
            peaks_top, prop_top = find_peaks(x, prominence=prominence)
            new_top = pd.DataFrame({'VALUE':[i for i in prop_top['prominences']]})
            new_top.index = peaks_top
            
            threshold_value = new_top['VALUE'].quantile(threshold)
            new_top = new_top[new_top['VALUE']>=threshold_value]
            
            new_top['SYMBOL'] = symbol
            new_top['COLUMN'] = col
            new_top['TYPE'] = 'RESISTANCE'
            
            
            # 計算低點
            peaks_btm, prop_btm = find_peaks(-x, prominence=prominence)   
            new_btm = pd.DataFrame({'VALUE':[i for i in prop_btm['prominences']]})
            new_btm.index = peaks_btm
            
            threshold_value = new_btm['VALUE'].quantile(threshold)
            new_btm = new_btm[new_btm['VALUE']>=threshold_value]            
            
            new_btm['SYMBOL'] = symbol
            new_btm['COLUMN'] = col
            new_btm['TYPE'] = 'SUPPORT'
            
            # Append
            result_raw = result_raw.append(new_top).append(new_btm)

            if j == 0:
                # Keep the column names
                loop_times = len(temp_df)
        
        if j % 100 == 0:
            print('add_support_resistance - ' + str(j) + '/' \
                  + str(len(symbol)-1))
      
        
    result = result_raw \
            .reset_index() \
            .merge(date_index, how='left', on=['SYMBOL', 'index']) \
            .drop('index', axis=1)
            
    result = result[['SYMBOL', 'WORK_DATE', 'COLUMN', 'TYPE', 'VALUE']]


    if plot_data:
        plot_close = loc_df[['SYMBOL', 'WORK_DATE', 'CLOSE']] \
                        .rename(columns={'CLOSE':'VALUE'})
        plot_close['TYPE'] = 'CLOSE'
        plot_data_df = result.append(plot_close)
        plot_data_df = cbyz.df_ymd(df=plot_data_df, cols='WORK_DATE')
        
        # single_plot = plot_data_df[plot_data_df['SYMBOL']==2456]
        # cbyz.plotly(df=plot_data_df, x='WORK_DATE', y='VALUE', groupby='TYPE')


    print('以下還沒改完')
    # .......
    if days:
        
        for i in range(len(cols)):
            col = cols[i]
            resist_col = col + '_RESISTANCE'
            support_col = col + '_SUPPORT'
            
            resist_days = col + '_RESISTANCE_DAYS'
            support_days = col + '_SUPPORT_DAYS'  
    
            result.loc[result.index, resist_days] = \
                np.where(result[resist_col].isna(), np.nan, result.index)
        
            result.loc[result.index, support_days] = \
                np.where(result[support_col].isna(), np.nan, result.index)
        
            # FIll NA ......
            # 這裡理論上不該用backward，所以補進平均值
            result = cbyz.df_shift_fill_na(df=result, 
                                            loop_times=loop_times,
                                            cols=resist_days, 
                                            group_by=['STOCK_SYMBOL'],
                                            forward=True, backward=False)  
        
            result = cbyz.df_shift_fill_na(df=result, 
                                            loop_times=loop_times,
                                            cols=support_days, 
                                            group_by=['STOCK_SYMBOL'],
                                            forward=True, backward=False)  
        
            # Calculate date difference
            result.loc[result.index, resist_col] = \
                result.index - result[resist_col]
        
            result.loc[result.index, support_col] = \
                result.index - result[support_col]
        

            # COnvert NA
            result = cbyz.df_conv_na(df=result, 
                                      cols=[resist_col, support_col])
            
            result = cbyz.df_conv_na(df=result, cols=resist_days,
                                      value=result[resist_days].mean())            
            
            result = cbyz.df_conv_na(df=result, cols=support_days,
                                      value=result[support_days].mean())        
                
    return result, return_cols




def test_support_resistance():
    

    loc_df = stk.od_tw_get_index(path=path_dcm)
    loc_df['SYMBOL'] = 1001
    
    
    add_support_resistance(df=loc_df, cols='TW_INDEX_CLOSE', 
                           rank_thld=10, prominence=4, 
                           days=True, threshold=0.9, plot_data=False)    



# %% Debug ------

def debug():
    

    file = pd.read_csv(path_temp + '/debug_hist_df.csv')
    chk = cbyz.df_chk_col_na(df=file)
    chk = chk[~chk['COLUMN'].isin(var_y)]
    chk['NA_COUNT'].max()
    
    chk2 = file[file['COVID19_TW_MA_20_LAG'].isna()]
    
    
    debug_file = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis/debug/model_data.csv'
    debug_file = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis/debug/reg_model_close_change_ratio_short_result_20220115_170956.csv'
    
    file = pd.read_csv(debug_file)    
    file['CLOSE_CHANGE_RATIO'].min()
    file['CLOSE_CHANGE_RATIO'].max()
    
    cbyz.df_chk_col_na(df=file)
    cbyz.df_chk_col_min_max(df=file)
    


# %% Execution ------


if __name__ == '__main__':
    
    symbol = [2520, 2605, 6116, 6191, 3481, 2409, 2603]
    
    # predict_result, precision = \
    #     master(param_holder=param_holder,
    #            predict_begin=20211209,
    #             _volume_thld=1000,
    #             fast=True)


    # Arguments
    args = {'bt_last_begin':[20211201],
            'predict_period': [3], 
            'data_period':[300],
            'ma_values':[[6,10,20,60]],
            'volume_thld':[300],
            'industry':[True],
            'trade_value':[True],
            'market':['tw'],
            'compete_mode':[2],
            'train_mode':[2],            
            'cv':[2],
            'fast':[True],  # 之後用train_mode代替
            'kbest':['all'],
            'dev':[True],
            'symbol':[symbol],
            'debug':[False]
            }
    
    param_holder = ar.Param_Holder(**args)
            
        
    master(param_holder=param_holder,
           predict_begin=20211201, threshold=30000)        


