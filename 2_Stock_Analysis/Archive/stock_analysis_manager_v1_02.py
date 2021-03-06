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
import pickle


local = False
local = True

stock_type = 'tw'

# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/2_Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/2_Stock_Analysis'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])        


# %% Inner Function ------


# def data_process(df, ma_group_by=[], norm_group_by=[], lag_group_by=[], 
#                  date_col='WORK_DATE', 
#                  ma=True, normalize=True, lag=True, 
#                  ma_cols=[], ma_except=[], norm_cols=[], norm_except=[],
#                  lag_cols=[], lag_except=[], drop_except=[]):

#     global ma_values, predict_period
    
#     # Convert To List ......
#     ma_group_by = cbyz.conv_to_list(ma_group_by)
#     norm_group_by = cbyz.conv_to_list(norm_group_by)
#     lag_group_by = cbyz.conv_to_list(lag_group_by)
    
#     # MA
#     ma_cols = cbyz.conv_to_list(ma_cols)
#     ma_except = cbyz.conv_to_list(ma_except)
    
#     # Normalize
#     norm_cols = cbyz.conv_to_list(norm_cols)
#     norm_except = cbyz.conv_to_list(norm_except)
    
#     # Lag
#     lag_cols = cbyz.conv_to_list(lag_cols)
#     lag_except = cbyz.conv_to_list(lag_except)
    
#     drop_except = cbyz.conv_to_list(drop_except)
    
#     date_col_list = [date_col]
#     drop_except = drop_except + date_col_list
    
#     cols = list(df.columns)    
#     main_data = df.copy()
    
    
#     # Calculate MA ......
#     if ma:
        
#         if len(ma_cols) == 0:
#             cols = cbyz.li_remove_items(cols, 
#                                         ma_group_by+ma_except+date_col_list)
#         else:
#             cols = ma_cols
            
#         main_data, ma_cols = cbyz.df_add_ma(df=main_data, cols=cols, 
#                                            group_by=ma_group_by, 
#                                            date_col=date_col, values=ma_values,
#                                            wma=False)
        
#         drop_cols = cbyz.li_remove_items(cols, drop_except)
#         main_data = main_data.drop(drop_cols, axis=1)
#         cols = ma_cols


#     # Normalize ......
#     if normalize:
        
#         if len(norm_cols) == 0:
#             norm_cols = cbyz.df_get_cols_except(df=main_data, 
#                                                 except_cols=norm_group_by + norm_except)
        
#         main_data, _, _, _ = cbyz.df_normalize(df=main_data,
#                                               cols=norm_cols,
#                                               groupby=norm_group_by,
#                                               show_progress=True)        
    
#     # Lag ......
#     if lag:
        
#         if len(lag_cols) == 0:        
#             lag_cols = cbyz.df_get_cols_except(df=main_data, 
#                                                except_cols=lag_group_by+lag_except)
        
#         main_data, _ = cbyz.df_add_shift(df=main_data, cols=lag_cols, 
#                                         shift=predict_period,
#                                         group_by=lag_group_by,
#                                         suffix='_LAG', 
#                                         remove_na=False)    

#         drop_cols = cbyz.li_remove_items(lag_cols, drop_except)
#         main_data = main_data.drop(drop_cols, axis=1)


#     cols = cbyz.df_get_cols_except(df=main_data, 
#                                    except_cols=ma_group_by \
#                                        +norm_group_by+lag_group_by)
        

#     return main_data, cols


# .............


def get_market_data_raw(industry=True, trade_value=True):
    
    
    global stock_symbol
    global market_data_raw
    global predict_period
    global stock_info_raw
    

    stock_info_raw = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    stock_info_raw = stock_info_raw[['STOCK_SYMBOL', 'CAPITAL', 
                                     'CAPITAL_LEVEL', 'ESTABLISH_DAYS', 
                                     'LISTING_DAYS', 'INDUSTRY_ONE_HOT']]    

    # Market Data ...
    # Shift one day forward to get complete PRICE_CHANGE_RATIO
    loc_begin = cbyz.date_cal(shift_begin, -1, 'd')    
    
    if len(stock_symbol) == 0:
        market_data_raw = stk.get_data(data_begin=loc_begin, 
                            data_end=data_end, 
                            stock_type=stock_type, stock_symbol=[], 
                            price_change=True, price_limit=False, 
                            trade_value=trade_value,
                            local=local)
    else:
        market_data_raw = stk.get_data(data_begin=loc_begin, 
                            data_end=data_end, 
                            stock_type=stock_type, 
                            stock_symbol=stock_symbol, 
                            price_change=True, price_limit=False,
                            trade_value=trade_value,
                            local=local)
        

    # Exclude New Symbols ......
    # Exclude the symbols that listing date shorter than data_period
    date_min = market_data_raw['WORK_DATE'].min()
    market_data_raw['MIN_DATE'] = market_data_raw \
                        .groupby(['STOCK_SYMBOL'])['WORK_DATE'] \
                        .transform('min')

    market_data_raw = market_data_raw[market_data_raw['MIN_DATE']==date_min] \
                        .drop('MIN_DATE', axis=1)


    # Exclude Low Volume Symbols ......
    global volume_thld
    market_data_raw = select_stock_symbols()

    # Add K line ......
    market_data_raw = market_data_raw \
            .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
            .reset_index(drop=True)
            
    market_data_raw = stk.add_k_line(market_data_raw)
    market_data_raw = cbyz.df_get_dummies(df=market_data_raw, 
                                          cols=['K_LINE_COLOR', 'K_LINE_TYPE'])
    
    # Add Support Resistance ......
    # print('add_support_resistance - days == True時有bug，全部數值一樣，導致沒辦法標準化？')
    # global data_period
    # market_data_raw, _ = \
    #     stk.add_support_resistance(df=market_data_raw, cols='CLOSE',
    #                                rank_thld=int(data_period * 2 / 360),
    #                                prominence=4, days=False)


    # Add Data Index ......
    # date_index = market_data_raw[['WORK_DATE']] \
    #             .drop_duplicates() \
    #             .sort_values(by='WORK_DATE') \
    #             .reset_index(drop=True) \
    #             .reset_index() \
    #             .rename(columns={'index':'DATE_INDEX'})
  
    # market_data_raw = market_data_raw.merge(date_index, how='left', on='WORK_DATE')  
    
    
    # Add Price Change Absolute ......
    market_data_raw['PRICE_CHANGE_ABS'] = abs(market_data_raw['PRICE_CHANGE'])
    market_data_raw['PRICE_CHANGE_RATIO_ABS'] = abs(market_data_raw['PRICE_CHANGE_RATIO'])    



    # Predict Symbols ......
    # 1. Prevent some symbols excluded by select_stock_symbols(), but still
    #    exists.
    global stock_symbol_df    
    all_symbols = market_data_raw['STOCK_SYMBOL'].unique().tolist()
    stock_symbol_df = pd.DataFrame({'STOCK_SYMBOL':all_symbols})
    
    # Calendar
    calendar_proc = calendar \
                    .reset_index() \
                    .rename(columns={'index':'DATE_INDEX'})
    
    calendar_proc, _, _, _, _ = \
        cbyz.ml_data_process(df=calendar_proc, 
                                  ma=False, normalize=True, lag=False, 
                                  ma_group_by=[],
                                  norm_group_by=[], 
                                  lag_group_by=[],
                                  ma_cols_contains=[], 
                                  ma_except_contains=[],
                                  norm_cols_contains=[], 
                                  norm_except_contains=['WORK_DATE', 
                                                        'TRADE_DATE'],
                                  lag_cols_contains=[], lag_except_contains=[], 
                                  drop_except_contains=[],
                                  ma_values=ma_values, 
                                  lag_period=predict_period)    
    
    # Merge As Main Data
    main_data_frame = cbyz.df_cross_join(stock_symbol_df, calendar_proc)
    main_data_frame = main_data_frame[(main_data_frame['TRADE_DATE']>=1) \
                        & (main_data_frame['WORK_DATE']<=predict_date[-1])] \
                        .drop('TRADE_DATE', axis=1)

    market_data_raw = main_data_frame.merge(market_data_raw, how='left',
                                            on=['STOCK_SYMBOL', 'WORK_DATE'])
    

# ...........


def sam_load_data(industry=True, trade_value=True):
    '''
    讀取資料及重新整理
    '''
    
    global stock_symbol
    global market_data_raw
    global predict_period
    global stock_symbol_df
    global stock_info_raw
    
        
    # Process Market Data
    loc_main = market_data_raw.drop('TOTAL_TRADE_VALUE', axis=1)
    
    loc_main, _, norm_orig, norm_group, norm_method = \
        cbyz.ml_data_process(df=loc_main, ma=True, normalize=True, lag=True, 
                            ma_group_by=['STOCK_SYMBOL'],   
                            norm_group_by=['STOCK_SYMBOL'], 
                            lag_group_by=['STOCK_SYMBOL'], 
                            ma_cols_contains=[], 
                            ma_except_contains=['WORK_DATE'],
                            norm_cols_contains=[], 
                            norm_except_contains=['WORK_DATE'],
                            lag_cols_contains=[], 
                            lag_except_contains=['WORK_DATE'], 
                            drop_except_contains=model_y,
                            ma_values=ma_values, 
                            lag_period=predict_period)
        
        
    # Drop Except會導致CLOSE_LAG, HIGH_LAG沒被排除
    y_lag = [y + '_LAG' for y in model_y]
    loc_main = loc_main.drop(y_lag, axis=1)
    
        
    # Total Market Trade
    if trade_value:
        total_trade = market_data_raw[['WORK_DATE', 'TOTAL_TRADE_VALUE']] \
            .drop_duplicates(subset=['WORK_DATE'])
        
        total_trade, _, _, _, _ = \
            cbyz.ml_data_process(df=total_trade, ma=True, normalize=True, 
                                 lag=True, ma_group_by=[],
                                 norm_group_by=[], 
                                 lag_group_by=[],
                                 ma_cols_contains=['TOTAL_TRADE_VALUE'], 
                                 ma_except_contains=[],
                                 norm_cols_contains=['TOTAL_TRADE_VALUE'], 
                                 norm_except_contains=[],
                                 lag_cols_contains=['TOTAL_TRADE_VALUE'], 
                                 lag_except_contains=[], 
                                 drop_except_contains=['WORK_DATE'],
                                 ma_values=ma_values, 
                                 lag_period=predict_period)
        
        loc_main = loc_main.merge(total_trade, how='left', on=['WORK_DATE'])  


    # Stock Info ...
    
    
    stock_info = stock_info_raw.drop(['INDUSTRY_ONE_HOT'], axis=1)
    
    stock_info, _, _, _, _ = \
        cbyz.ml_data_process(df=stock_info, ma=False, normalize=True, 
                            lag=False, ma_group_by=[],
                            norm_group_by=[], lag_group_by=[],
                            ma_cols_contains=[], ma_except_contains=[],
                            norm_cols_contains=[], 
                            norm_except_contains=['STOCK_SYMBOL'],
                            lag_cols_contains=[], lag_except_contains=[], 
                            drop_except_contains=[],
                            ma_values=ma_values, 
                            lag_period=predict_period)
    
    loc_main = loc_main.merge(stock_info, how='left', on=['STOCK_SYMBOL'])      


    # Merge Other Data ......        
    if industry:        
        stock_industry = stock_info_raw[['STOCK_SYMBOL', 'INDUSTRY_ONE_HOT']]
        stock_info_dummy = cbyz.df_get_dummies(df=stock_industry, 
                                               cols='INDUSTRY_ONE_HOT')
        
        # Industry Data ...
        print('sam_load_data - 當有新股上市時，產業資料的比例會出現大幅變化，' \
              + '評估如何處理')
        
        if trade_value:
            industry_data = market_data_raw[['STOCK_SYMBOL', 'WORK_DATE', 
                                             'CLOSE', 'VOLUME', 
                                             'PRICE_CHANGE_ABS', 
                                             'PRICE_CHANGE_RATIO_ABS', 
                                             'SYMBOL_TRADE_VALUE', 
                                             'TOTAL_TRADE_VALUE']]
        else:
            industry_data = market_data_raw[['STOCK_SYMBOL', 'WORK_DATE', 
                                             'CLOSE', 'VOLUME', 
                                             'PRICE_CHANGE_ABS', 
                                             'PRICE_CHANGE_RATIO_ABS']]

        # Merge        
        industry_data = industry_data.merge(stock_industry, on='STOCK_SYMBOL')
        
        
        if trade_value:
            
            industry_data['TRADE_VALUE'] = \
                industry_data \
                .groupby(['WORK_DATE', 'INDUSTRY_ONE_HOT'])['SYMBOL_TRADE_VALUE'] \
                .transform('sum')
    
            industry_data['TRADE_VALUE_RATIO'] = \
                industry_data['TRADE_VALUE'] / industry_data['TOTAL_TRADE_VALUE']            
            
            industry_data = industry_data \
                            .groupby(['WORK_DATE', 'INDUSTRY_ONE_HOT']) \
                            .agg({'CLOSE':'sum', 
                                  'VOLUME':'sum',
                                  'PRICE_CHANGE_ABS':'sum',
                                  'PRICE_CHANGE_RATIO_ABS':'mean',
                                  'TRADE_VALUE':'sum',
                                  'TRADE_VALUE_RATIO':'mean'}) \
                            .reset_index() \
                            .rename(columns={'CLOSE':'INDUSTRY_CLOSE'})            
        else:
            industry_data = industry_data \
                            .groupby(['WORK_DATE', 'INDUSTRY_ONE_HOT']) \
                            .agg({'CLOSE':'sum', 
                                  'VOLUME':'sum',
                                  'PRICE_CHANGE_ABS':'sum',
                                  'PRICE_CHANGE_RATIO_ABS':'mean'}) \
                            .reset_index() \
                            .rename(columns={'CLOSE':'INDUSTRY_CLOSE'})
        
        # Rename ...
        cols = cbyz.df_get_cols_except(df=industry_data,
                                       except_cols=['WORK_DATE', 'INDUSTRY_ONE_HOT'])
        
        new_cols = ['INDUSTRY_' + c for c in cols]                  
        rename_dict = cbyz.li_to_dict(cols, new_cols)
        industry_data = industry_data.rename(columns=rename_dict)
                       
        
        industry_data, _, _, _, _ = \
             cbyz.ml_data_process(df=industry_data, 
                                  ma=True, normalize=True, lag=True, 
                                  ma_group_by=['INDUSTRY_ONE_HOT'],
                                  norm_group_by=['INDUSTRY_ONE_HOT'], 
                                  lag_group_by=['INDUSTRY_ONE_HOT'],
                                  ma_cols_contains=[], 
                                  ma_except_contains=['WORK_DATE'],
                                  norm_cols_contains=[], 
                                  norm_except_contains=['WORK_DATE'],
                                  lag_cols_contains=[], 
                                  lag_except_contains=['WORK_DATE'], 
                                  drop_except_contains=[],
                                  ma_values=ma_values, 
                                  lag_period=predict_period)
        
        # Merge ...
        loc_main = loc_main \
            .merge(stock_info_dummy, how='left', on='STOCK_SYMBOL') \
            .merge(stock_industry, how='left', on='STOCK_SYMBOL') \
            .merge(industry_data, how='left', on=['WORK_DATE', 'INDUSTRY_ONE_HOT']) \
            .drop('INDUSTRY_ONE_HOT', axis=1)
        
    return loc_main, norm_orig, norm_group, norm_method 



# ...........


def get_sale_mon_data():
    
    '''
    除權息時間
    Optimize
    '''
    
    file_raw = pd.DataFrame()
    years = list(range(2018, 2022))

    for y in years:
        new_file = pd.read_excel(path_resource + '/sale_mon/SaleMonDetail_' \
                                 + str(y) + '.xlsx')
        file_raw = file_raw.append(new_file)
        
        
    new_cols = range(len(file_raw.columns))
    new_cols = ['SALE_MON_' + str(c) for c in new_cols]

    file_raw.columns = new_cols

    file_raw = file_raw[['SALE_MON_1', 'SALE_MON_4', 'SALE_MON_5', 'SALE_MON_6']]
    file_raw.columns = ['STOCK_SYMBOL', 'WORK_DATE', 
                        'EX_DIVIDENDS_PRICE', 'EX_DIVIDENDS_DONE']
    file_raw = file_raw.dropna()
    
    
    file1 = file_raw[['STOCK_SYMBOL', 'WORK_DATE', 'EX_DIVIDENDS_PRICE']]
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
    file2 = file_raw[['STOCK_SYMBOL', 'EX_DIVIDENDS_DONE']]
    file2.columns = ['STOCK_SYMBOL', 'WORK_DATE']
    file2['WORK_DATE'] = '20' + file2['WORK_DATE']
    file2['WORK_DATE'] = file2['WORK_DATE'].str.replace("'", "")
    file2['WORK_DATE'] = file2['WORK_DATE'].str.replace("/", "")
    file2 = cbyz.df_conv_col_type(df=file2, cols='WORK_DATE', to='int')
    file2['EX_DIVIDENDS_DONE'] = 1
    
    file2 = cbyz.df_conv_na(df=file2, cols=['EX_DIVIDENDS_DONE'])
    
    return file1, file2

# .............


def select_stock_symbols():

    
    global market_data_raw
    global stock_info_raw


    # Select Rules
    # 1. 先找百元以下的，才有資金可以買一整張
    # 2. 不要找疫情後才爆漲到歷史新高的

    # global predict_date
    # predict_begin = cbyz.date_cal(predict_date[0], -3, 'm')


    # data_end = cbyz.date_get_today()
    # data_begin = cbyz.date_cal(data_end, -1, 'm')


    # Exclude ETF ......
    all_symbols = stock_info_raw[['STOCK_SYMBOL']]
    df = all_symbols.merge(market_data_raw, on=['STOCK_SYMBOL']) 


    # Stock info
    # Section 1. 資本額大小 ------
    
    # 1-1. 挑選中大型股 ......
    remove_symbols = stock_info_raw[stock_info_raw['CAPITAL_LEVEL']==1]
    remove_symbols = remove_symbols[['STOCK_SYMBOL']]
    df = cbyz.df_anti_merge(df, remove_symbols, on='STOCK_SYMBOL')    
    

    # 挑選交易量大的 ......
    global volume_thld
    if volume_thld > 0:
        
        one_week_date = cbyz.date_cal(predict_date[0], -14, 'd')
        one_week_volume = df[df['WORK_DATE']>=one_week_date]
        
        one_week_volume = one_week_volume \
                            .groupby(['STOCK_SYMBOL']) \
                            .agg({'VOLUME':'min'}) \
                            .reset_index()
                            
        one_week_volume = \
            one_week_volume[one_week_volume['VOLUME']<volume_thld * 1000] \
            .drop('VOLUME', axis=1)
                                    
        if len(one_week_volume) > 0:
            df = cbyz.df_anti_merge(df, one_week_volume, on='STOCK_SYMBOL')


    
    # 3天漲超過15%  .....
    # temp_begin = cbyz.date_cal(predict_begin, -3, 'm')
    # data = df[df['WORK_DATE']>=temp_begin]    
    
    # data, cols_pre = cbyz.df_add_shift(df=data, 
    #                                    group_by=['STOCK_SYMBOL'], 
    #                                    cols=['CLOSE'], shift=3,
    #                                    remove_na=False)


    # data['TEMP_PRICE_CHANGE_RATIO'] = (data['CLOSE'] - data['CLOSE_PRE']) \
    #                         / data['CLOSE_PRE']
    
    
    # data = data[data['TEMP_PRICE_CHANGE_RATIO']<0.15]
    # data = data[['STOCK_SYMBOL']].drop_duplicates().reset_index(drop=True)

    # df = cbyz.df_anti_merge(df, data, on='STOCK_SYMBOL')

    
    return df




def select_stock_symbols_manually(data_begin, data_end):


    # Select Rules
    # 1. 先找百元以下的，才有資金可以買一整張
    # 2. 不要找疫情後才爆漲到歷史新高的


    data_end = cbyz.date_get_today()
    data_begin = cbyz.date_cal(data_end, -1, 'm')


    # Stock info
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    
    
    # Section 1. 資本額大小 ------
    
    # 1-1. 挑選中大型股 ......
    level3_symbol = stock_info[stock_info['CAPITAL_LEVEL']>=2]
    level3_symbol = level3_symbol['STOCK_SYMBOL'].tolist()
    
    data_raw = stk.get_data(data_begin=data_begin, data_end=data_end, 
                            stock_symbol=level3_symbol, 
                            price_change=True,
                            shift=0, stock_type='tw', local=local)
    
    data = data_raw[data_raw['STOCK_SYMBOL'].isin(level3_symbol)]
    

    # 1-2. 不排除 ......
    data = stk.get_data(data_begin=data_begin, data_end=data_end, 
                            stock_symbol=[], 
                            price_change=True,
                            shift=0, stock_type='tw', local=True)
    
    
    # Section 2. 依價格篩選 ......
    
    # 2-1. 不篩選 .....
    target_symbols = data[['STOCK_SYMBOL']] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    
    
    # 2-2. 低價股全篩 .....
    # 目前排除80元以上
    last_date = data['WORK_DATE'].max()
    last_price = data[data['WORK_DATE']==last_date]
    last_price = last_price[last_price['CLOSE']>80]
    last_price = last_price[['STOCK_SYMBOL']].drop_duplicates()
    
    
    target_symbols = cbyz.df_anti_merge(data, last_price, on='STOCK_SYMBOL')
    target_symbols = target_symbols[['STOCK_SYMBOL']].drop_duplicates()
    
    
    # 2-3. 3天漲超過10%  .....
    data, cols_pre = cbyz.df_add_shift(df=data, 
                                       group_by=['STOCK_SYMBOL'], 
                                       cols=['CLOSE'], shift=3,
                                       remove_na=False)
    

    data['PRICE_CHANGE_RATIO'] = (data['CLOSE'] - data['CLOSE_PRE']) \
                            / data['CLOSE_PRE']
    
    
    results_raw = data[data['PRICE_CHANGE_RATIO']>=0.15]
    
    
    summary = results_raw \
                .groupby(['STOCK_SYMBOL']) \
                .size() \
                .reset_index(name='COUNT')
                
    
    # Select Symboles ......
    target_symbols = results_raw.copy()
    target_symbols = cbyz.df_add_size(df=target_symbols,
                                      group_by='STOCK_SYMBOL',
                                      col_name='TIMES')
        
    target_symbols = target_symbols \
                    .groupby(['STOCK_SYMBOL']) \
                    .agg({'CLOSE':'mean',
                          'TIMES':'mean'}) \
                    .reset_index()
    
    target_symbols = target_symbols.merge(stock_info, how='left', 
                                          on='STOCK_SYMBOL')
    
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
    # plot_data = results.melt(id_vars='PROFIT')

    # cbyz.plotly(df=plot_data, x='PROFIT', y='value', groupby='variable', 
    #             title="", xaxes="", yaxes="", mode=1)

    
    return results_raw, stock_info




def get_google_treneds(begin_date=None, end_date=None, 
                       normalize=True, stock_type='tw', local=True):
    
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
                                       stock_type=stock_type,
                                       local=local)

    # Word List ......
    file_path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/2_Stock_Analysis/Resource/google_trends_industry.xlsx'
    file = pd.read_excel(file_path, sheet_name='words')
    
    file = file[(file['STOCK_TYPE'].str.contains(stock_type)) \
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
    
    main_data = cbyz.df_shift_fill_na(df=main_data, loop_times=len(calendar), 
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
        main_data, _, _, _ = cbyz.df_normalize(df=main_data, cols='VALUE')
    
    
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






# %% Process ------



def get_model_data(ma_values=[5,20], industry=True, trade_value=True):
    
    
    global shift_begin, shift_end, data_begin, data_end
    global predict_date, predict_period, calendar    
    global stock_symbol
    global model_y
    global market_data_raw    
    

    # Stock Info .......
    # stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)  
    identify_cols = ['STOCK_SYMBOL', 'WORK_DATE']


    # Symbols ......
    stock_symbol = cbyz.conv_to_list(stock_symbol)
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, 'str')


    get_market_data_raw(industry=True, trade_value=True)
    

    # Load Historical Data ......
    main_data, norm_orig, norm_group, norm_method  = \
        sam_load_data(industry=industry, 
                      trade_value=trade_value) 
        
    
    # Add Symbol As Categorical Data ......
    # 造成精準度下降，而且要跑很久
    # main_data.loc[main_data.index, 'SYMBOL_VAR'] = main_data['STOCK_SYMBOL']
    
    # symbol_var_cols = main_data['SYMBOL_VAR'].unique().tolist()
    # symbol_var_cols = ['SYMBOL_VAR_' + s for s in symbol_var_cols]
    
    # ma_except_cols = ma_except_cols + symbol_var_cols
    # lag_except_cols = lag_except_cols + symbol_var_cols
    

    # # TODC Shareholdings Spread ......
    # sharehold = stk.tdcc_get_sharehold_spread(shift_begin, end_date=None,
    #                                           local=local) 
    
    # main_data = main_data.merge(sharehold, how='left', 
    #                           on=['STOCK_SYMBOL', 'WORK_DATE'])      


    # Government Invest ......
    gov_invest = stk.opd_get_gov_invest(path=path_resource)
    main_data = main_data.merge(gov_invest, how='left', on=['STOCK_SYMBOL'])
    main_data = cbyz.df_conv_na(df=main_data, cols=['GOV_INVEST'])


    # 除權息資料 ......
    # Close Lag ...
    daily_close = market_data_raw[['WORK_DATE', 'STOCK_SYMBOL', 'CLOSE']]
    daily_close, _ = cbyz.df_add_shift(df=daily_close, 
                                    cols='CLOSE', shift=1,
                                    group_by=['STOCK_SYMBOL'],
                                    suffix='_LAG', 
                                    remove_na=False)
    daily_close = daily_close \
                .drop('CLOSE', axis=1) \
                .rename(columns={'CLOSE_LAG':'CLOSE'})
    
    # 除權息 ...
    sale_mon_data1, sale_mon_data2 = get_sale_mon_data()
    
    # Data 1 - 除權息日期及價錢 ...
    sale_mon_data1 = daily_close.merge(sale_mon_data1, how='left', 
                                       on=['WORK_DATE', 'STOCK_SYMBOL'])
    
    sale_mon_data1['EX_DIVIDENDS_PRICE'] = \
        sale_mon_data1['EX_DIVIDENDS_PRICE'] / sale_mon_data1['CLOSE']    
        
    sale_mon_data1 = sale_mon_data1.drop('CLOSE', axis=1)
    sale_mon_data1 = cbyz.df_conv_na(df=sale_mon_data1, 
                                     cols=['EX_DIVIDENDS_PRICE', 
                                           'SALE_MON_DATE'])
    
    sale_mon_data1, _, _, _, _ = \
             cbyz.ml_data_process(df=sale_mon_data1, 
                                  ma=False, normalize=True, lag=False, 
                                  ma_group_by=[],
                                  norm_group_by=['STOCK_SYMBOL'], 
                                  lag_group_by=[],
                                  ma_cols_contains=[], ma_except_contains=[],
                                  norm_cols_contains=[], 
                                  norm_except_contains=[],
                                  lag_cols_contains=[], lag_except_contains=[], 
                                  drop_except_contains=[],
                                  ma_values=ma_values, 
                                  lag_period=predict_period)    
    
    
    # Data 2 - 填息 ...
    sale_mon_data2, _, _, _, _ = \
        cbyz.ml_data_process(df=sale_mon_data2, 
                             ma=False, normalize=True, lag=False, 
                             ma_group_by=['STOCK_SYMBOL'],
                             norm_group_by=['STOCK_SYMBOL'], 
                             lag_group_by=['STOCK_SYMBOL'],
                             ma_cols_contains=[], 
                             ma_except_contains=['EX_DIVIDENDS_DONE'],
                             norm_cols_contains=[], 
                             norm_except_contains=[],
                             lag_cols_contains=[], 
                             lag_except_contains=['EX_DIVIDENDS_DONE'], 
                             drop_except_contains=[],
                             ma_values=ma_values, 
                             lag_period=predict_period)    
        
    main_data = main_data \
        .merge(sale_mon_data1, how='left', on=['WORK_DATE', 'STOCK_SYMBOL']) \
        .merge(sale_mon_data2, how='left', on=['WORK_DATE', 'STOCK_SYMBOL'])
    
    # Convert NA
    temp_cols = ['EX_DIVIDENDS_PRICE', 'SALE_MON_DATE', 'EX_DIVIDENDS_DONE']    
    main_data = cbyz.df_conv_na(df=main_data, cols=temp_cols)


    
    # TEJ 三大法人持股成本 ......
    ewtinst1c_raw = stk.tej_get_ewtinst1c(begin_date=shift_begin, end_date=None, 
                                      trade=True, local=local)
    
    ewtinst1c = ewtinst1c_raw.copy()
    
    # 獲利率用全部的來norm，所以要分兩段
    hroi_cols = cbyz.df_get_cols_contain(df=ewtinst1c, 
                                         string=['_HROI', '_SELL', '_BUY'])
    
    other_cols = cbyz.df_get_cols_except(df=ewtinst1c, 
                                   except_cols=hroi_cols \
                                       + ['STOCK_SYMBOL', 'WORK_DATE'])
    
    # Keep Needed Symbols Only
    ewtinst1c = ewtinst1c.merge(stock_symbol_df, on=['STOCK_SYMBOL'])
    
    ewtinst1c, _, _, _, _ = \
        cbyz.ml_data_process(df=ewtinst1c, 
                             ma=True, normalize=True, lag=True, 
                             ma_group_by=[], norm_group_by=[], 
                             lag_group_by=[],
                             ma_cols_contains=hroi_cols, 
                             ma_except_contains=[],
                             norm_cols_contains=hroi_cols, 
                             norm_except_contains=[],
                             lag_cols_contains=hroi_cols, 
                             lag_except_contains=[], 
                             drop_except_contains=[],
                             ma_values=ma_values, 
                             lag_period=predict_period)    

    ewtinst1c, cols, _, _, _ = \
        cbyz.ml_data_process(df=ewtinst1c, 
                             ma=True, normalize=True, lag=True, 
                             ma_group_by=['STOCK_SYMBOL'],
                             norm_group_by=['STOCK_SYMBOL'], 
                             lag_group_by=['STOCK_SYMBOL'],
                             ma_cols_contains=['_HAP'], 
                             ma_except_contains=[],
                             norm_cols_contains=['_HAP'], 
                             norm_except_contains=[],
                             lag_cols_contains=['_HAP'], 
                             lag_except_contains=[], 
                             drop_except_contains=[],
                             ma_values=ma_values, 
                             lag_period=predict_period)
        
        
    main_data = main_data.merge(ewtinst1c, how='left', 
                              on=['STOCK_SYMBOL', 'WORK_DATE'])  
    
    main_data = cbyz.df_conv_na(df=main_data, cols=cols)


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
    covid19 = cbyz.get_covid19_data()
    
    covid19, covid19_cols, _, _, _ = \
                cbyz.ml_data_process(df=covid19, 
                             ma=True, normalize=True, lag=True, 
                             ma_group_by=[],
                             norm_group_by=[], 
                             lag_group_by=[],
                             ma_cols_contains=other_cols, 
                             ma_except_contains=[],
                             norm_cols_contains=other_cols, 
                             norm_except_contains=[],
                             lag_cols_contains=other_cols, 
                             lag_except_contains=[], 
                             drop_except_contains=[],
                             ma_values=ma_values, 
                             lag_period=predict_period)
        
        
    main_data = main_data.merge(covid19, how='left', on='WORK_DATE')
    main_data = cbyz.df_conv_na(df=main_data, cols=covid19_cols)

    

    if predict_period > min(ma_values):
        # Update, raise error here
        print('get_model_data - predict_period is longer than ma values, ' \
              + 'and it will cause na.')
        del main_data


    # Variables ......
    model_x = cbyz.df_get_cols_except(df=main_data, 
                                      except_cols=model_y + identify_cols)
    
    
    # Model Data ......
    main_data = main_data[main_data['WORK_DATE']>=data_begin] \
                        .reset_index(drop=True)


    # Hist Data中有部份資料缺值 ......
    print('Bug - get_model_data中這裡會有9000/154698筆資料被排除')
    hist_df = main_data[main_data['WORK_DATE']<predict_date[0]]
    hist_df = hist_df.dropna(axis=0)
    
    print('Bug - get_model_data中這裡會有50/1585筆資料被排除')
    predict_df = main_data[main_data['WORK_DATE']>=predict_date[0]]
    predict_df = predict_df.dropna(subset=model_x, axis=0)    
    
    
    main_data_final = hist_df.append(predict_df)


    # Remove all data with na values ......
    # 1. Some symbols may have serveral rows with na values
    # na_df = main_data_final[identify_cols+model_x]
    # na_df = na_df[na_df['WORK_DATE']<predict_date[0]]
    # na_df = na_df[na_df.isna().any(axis=1)]
    # symbols_removed = na_df['STOCK_SYMBOL'].unique().tolist()
    
    # main_data = main_data_final[~main_data['STOCK_SYMBOL'].isin(symbols_removed)] \
    #             .reset_index(drop=True)
            

    # Check - X裡面不應該有na，但Y的預測區間會是na ......
    chk_na = cbyz.df_chk_col_na(df=main_data_final, positive_only=True, 
                                return_obj=True, alert=True, 
                                alert_obj='main_data')
    

    export_dict = {'MODEL_DATA':main_data_final,
                   'MODEL_X':model_x,
                   'NORM_ORIG':norm_orig,
                   'NORM_GROUP':norm_group}
    
    
    return export_dict



# ...............



def split_data():
    
    global model_data, model_x, model_y, model_addt_vars, predict_date    
    global stock_symbol
    
    # Model Data ......
    cur_model_data = model_data[model_data['WORK_DATE']<predict_date[0]] \
                    .reset_index(drop=True)
    
    # Predict Data ......
    cur_predict_data = model_data[model_data['WORK_DATE']>=predict_date[0]]    


    # if len(stock_symbol) == 0:
    #     cur_model_data = model_data[model_data['WORK_DATE']<predict_date[0]] \
    #                     .reset_index(drop=True)
        
    #     # Predict Data ......
    #     cur_predict_data = model_data[model_data['WORK_DATE']>=predict_date[0]]
    # else:
    #     cur_model_data = model_data[(model_data['STOCK_SYMBOL']==symbol) \
    #             & (model_data['WORK_DATE']<predict_date[0])] \
    #         .reset_index(drop=True)
    
    #     # Predict Data ......
    #     cur_predict_data = model_data[
    #         (model_data['STOCK_SYMBOL']==symbol) \
    #             & (model_data['WORK_DATE']>=predict_date[0])]
        

    global X_train, X_test, y_train, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite

            
    # Predict            
    X_predict = cur_predict_data[model_x + model_addt_vars]
    # X_predict = X_predict[X_predict['STOCK_SYMBOL'].isin(stock_symbol)]
 
    
    # Traning And Test
    X = cur_model_data[model_x + model_addt_vars]
    y = cur_model_data[model_y + model_addt_vars]      
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    
    X_train_lite = X_train.drop(model_addt_vars, axis=1)   
    X_test_lite = X_test.drop(model_addt_vars, axis=1)   
    X_predict_lite = X_predict.drop(model_addt_vars, axis=1)   
    
    
    # y_train_lite = y_train[model_y[0]]
    # y_test_lite = y_test[model_y[0]]    
    y_train_lite = []
    y_test_lite = []    
    
    for i in range(len(model_y)):
        y_train_lite.append(y_train[model_y[i]])
        y_test_lite.append(y_test[model_y[i]])        
    
    
# ..............



def get_model(y_index, cv=5, load_model=False, export_model=True, path=None):
    
 
    global shift_begin, shift_end, data_begin, data_end
    global predict_begin, predict_end    
    global model_data, predict_date, model_x, model_y, norm_orig, norm_group
   
    
    import xgboost as xgb    
    global stock_symbol, model_data, predict_date

    global X_train, y_train, X_test, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite
    


    y = model_y[y_index].lower()
    model_path = path_export + '/saved_model_' + y + '.sav'    
    
    if load_model and path != None:
        try:
            loaded_model = pickle.load(open(model_path, 'rb'))    
        except Exception as e:
            print(e)
        else:
            return loaded_model

    # 參數設定 ......
    # model_params = {
        # 'xbgoost': {
        #     'model': xgb.XGBRegressor(),
        #     'params': {
        #         'n_estimators': [200, 400],
        #         'gamma':[0, 0.5],
        #         'max_depth':[4, 6]
        #     },
        # }
    # }

    model_params = {
        'xbgoost': {
            'model': xgb.XGBRegressor(),
            'params': {
                'n_estimators': [200, 250],
                'gamma':[0, 0.5],
                'max_depth':[4, 6]
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


def predict(load_model=False, cv=5):
    
    global shift_begin, shift_end, data_begin, data_end
    global predict_begin, predict_end    
    global model_data, predict_date, model_x, model_y, norm_orig, norm_group
    
    split_data()
    
   
    # Model ......
    results = pd.DataFrame()
    rmse = pd.DataFrame()    
    features = pd.DataFrame()    

    
    for i in range(0, len(model_y)):

        model = get_model(y_index=i, cv=cv, load_model=load_model, 
                          export_model=True, path=path_export)


        # Feature Importance ......
        features_new = {'FEATURES':list(X_train_lite.columns),
                        'IMPORTANCE':model.best_estimator_.feature_importances_}
        
        features_new = pd.DataFrame(features_new)            
        features_new['Y'] = model_y[i]
        features = features.append(features_new)        

    
        # RMSE ......
        preds_test = model.predict(X_test_lite)
        rmse_new = np.sqrt(mean_squared_error(y_test_lite[i], preds_test))
        rmse_new = pd.DataFrame(data=[rmse_new], columns=['RMSE'])
        rmse_new['Y'] = model_y[i]
        rmse = rmse.append(rmse_new)

    
        # Results ......
        preds = model.predict(X_predict_lite)
        results_new = X_predict[model_addt_vars].reset_index(drop=True)
        results_new['VALUES'] = preds
        results_new['Y'] = model_y[i]
        results = results.append(results_new)        
    
    
    # Organize ......
    rmse = rmse.reset_index(drop=True)
    
    # 
    features = features \
                .sort_values(by='IMPORTANCE', ascending=False) \
                .reset_index(drop=True)
    
    # results
    results_pivot = results \
                    .pivot_table(index=['STOCK_SYMBOL', 'WORK_DATE'],
                                 columns='Y',
                                 values='VALUES') \
                    .reset_index()
    
    results = cbyz.df_normalize_restore(df=results_pivot, 
                                        original=norm_orig,
                                        groupby=norm_group)
    
    return results, rmse, features



# %% Master ------

def master(_predict_begin, _predict_end=None, 
           _predict_period=15, _data_period=180, 
           _stock_symbol=[], _stock_type='tw', _ma_values=[3,5,20,60],
           _model_y=['OPEN', 'HIGH', 'LOW', 'CLOSE'],
           _volume_thld=1000, load_model=False, cv=2):
    '''
    主工作區
    '''
    
    # date_period為10年的時候會出錯
    
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
    
    
    # Worklist .....
    # 建長期投資的基本面模型
    # 為了配合TEJ，模型使用的資料可以delay一點    
    # 2. Error Function
    # 3. TEJ transaction data
    # 4. K Line Type要get_dummy
    # 5. print('add_support_resistance - days == True時有bug，全部數值一樣，導致沒辦法標準化？')
    # 6. 加上PRICE_CHANGE_ABS / PRICE_CHANGE加上STD
    
    # Optimize ......
    # 1. Modify pipeline, calculate ma and normalize then merge

    
    global version
    version = 1.02


    # industry=True
    # trade_value=True      
    # _data_period = int(365 * 2)
    # _predict_begin = 20210802
    # _predict_end = None
    # _stock_type = 'tw'
    # # ma_values = [2,5,20,60]
    # _ma_values = [5,10,20]
    # _predict_period = 5
    # _stock_symbol = ['2002', '2330', '2454', '2385', '3043']
    # _stock_symbol = []
    # _model_y= [ 'OPEN', 'HIGH', 'LOW', 'CLOSE']
    # # _model_y = ['PRICE_CHANGE_RATIO']      
    # # _model_y= ['CLOSE']
    # _volume_thld = 1000
    # load_model = False
    # cv = 5
    
    
    global volume_thld
    volume_thld = _volume_thld
    
    global ma_values
    ma_values = _ma_values


    global shift_begin, shift_end, data_begin, data_end, data_period
    global predict_date, predict_period, calendar
    
    
    predict_period = _predict_period
    # data_shift = -(max(ma_values) * 3)
    data_shift = -(max(ma_values) * 2)
    data_period = _data_period
    
    shift_begin, shift_end, \
            data_begin, data_end, predict_date, calendar = \
                stk.get_period(data_begin=None,
                               data_end=None, 
                               data_period=data_period,
                               predict_begin=_predict_begin,
                               predict_period=predict_period,
                               shift=data_shift)  
    
    # .......
    global stock_symbol, stock_type
    stock_type = _stock_type
    stock_symbol = _stock_symbol
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, to_type='str')


    # ......
    global model_data
    global model_x, model_y, model_addt_vars
    global norm_orig, norm_group
        
    model_y = _model_y
    
    
    # 0707 - industry可以提高提精準，trade_value會下降
    data_raw = get_model_data(ma_values=ma_values, 
                              industry=True, 
                              trade_value=True)
    
    
    model_data = data_raw['MODEL_DATA']
    model_x = data_raw['MODEL_X']
    norm_orig = data_raw['NORM_ORIG']
    norm_group = data_raw['NORM_GROUP']
    model_addt_vars = ['STOCK_SYMBOL', 'WORK_DATE']
    
    
    global predict_results
    predict_results = predict(load_model=load_model, cv=cv)
    predict_results
    # features = predict_results[2]
    
    return predict_results



# if __name__ == '__main__':
    
    # master()



def check():
    
    chk = cbyz.df_chk_col_na(df=model_data_raw)    
    chk = cbyz.df_chk_col_na(df=model_data)

    # Err01
    chk = main_data[model_x]
    chk_na = cbyz.df_chk_col_na(df=chk, positive_only=True, return_obj=True,
                                alert=True, alert_obj='main_data')
    
    chk = main_data[main_data['OPEN_MA_20_LAG'].isna()]
    
    
    # Check Columns Not Normalized .......
    cols = list(data.columns)
    debug = pd.DataFrame()
    
    for c in cols:
        new_df = pd.DataFrame({'COL':[c],
                               'MIN':[data[c].min()],
                               'MAX':[data[c].max()]})
        
        debug = debug.append(new_df)
    
    debug = cbyz.df_conv_col_type(df=debug, cols=['MAX'], to='float')
    chk = debug[debug['MAX']>1]


    # Check NA ......
    chk_na = cbyz.df_chk_col_na(df=na_df, positive_only=True, return_obj=True,
                                alert=True, alert_obj='main_data')
    
        
    
# %% Archive ------


def predict_old():
    
    global shift_begin, shift_end, data_begin, data_end
    global predict_begin, predict_end    
    global model_data, predict_date, model_x, model_y, norm_orig, norm_group
    
   
    split_data()
   
    
    # global X_train, X_test, y_train, y_test, X_predict
    # global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite

    # bulk_test = cbyz.ml_bulk_run_reg_model(X_train=X_train_lite, 
    #                                         y_train=y_train_lite[0], 
    #                                         X_test=X_test_lite, 
    #                                         y_test=y_test_lite[0], 
    #                                         rf=False, kr=False, lreg=True, 
    #                                         svr=True, dt=False, sgd=True, 
    #                                         knn=False, xgb=False)
    
    
   
    # Model ......
    model_list = get_model_list()
    results = pd.DataFrame()
    rmse = pd.DataFrame()    
    features = pd.DataFrame()    
    
    
    for i in range(0, len(model_list)):

        cur_model = model_list[i]
        
        # Model results .......
        # (1) Update, should deal with multiple signal issues.
        #     Currently, only consider the first signal.
        
        # global model_results_raw
        model_results_raw = cur_model(remove_none=True)      
        
        
        if len(model_results_raw['RESULTS']) == 0:
            continue
        
        
        model_name = cur_model.__name__
        
        
        # Results ......
        temp_results = model_results_raw['RESULTS']
        temp_results['MODEL'] = model_name
        results = results.append(temp_results)
        
        
        # Features ......
        new_features = model_results_raw['FEATURES']
        new_features['MODEL'] = model_name
        features = features.append(new_features)
        
        # RMSE ......
        new_rmse = model_results_raw['RMSE']
        new_rmse['MODEL'] = model_name
        rmse = rmse.append(new_rmse)
        
    
    # Organize ......
    rmse = rmse.reset_index(drop=True)
    
    # 
    features = features \
                .sort_values(by='IMPORTANCE', ascending=False) \
                .reset_index(drop=True)
    
    # results
    results_pivot = results \
                    .pivot_table(index=['STOCK_SYMBOL', 'WORK_DATE', 'MODEL'],
                                 columns='Y',
                                 values='VALUES') \
                    .reset_index()
    
    results = cbyz.df_normalize_restore(df=results_pivot, 
                                        original=norm_orig,
                                        groupby=norm_group)
    
    return results, rmse, features
    



# %% Dev ------



def model_5(remove_none=True):
    '''
    Flatten Linear regression
    '''
    
    from sklearn.linear_model import LinearRegression
    global stock_symbol, model_x, model_y, model_addt_vars
    global model_data, predict_date

    
    # Model ........
    global X_train, X_test, y_train, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite
    
    
    # Predict ......
    features = pd.DataFrame()
    results = pd.DataFrame()
    rmse = pd.DataFrame()    
    
    
    for i in range(len(model_y)):
        
        reg = LinearRegression().fit(X_train_lite,
                                     y_train_lite[i])
        
        features_new = pd.DataFrame({'FEATURES':list(X_train_lite.columns),
                                     'IMPORTANCE':list(reg.coef_)})
        features_new['Y'] = model_y[i]
        features = features.append(features_new)
        # reg.score(x, y)
        # reg.coef_
        # reg.intercept_
        
    
        # RMSE ......
        preds_test = reg.predict(X_test_lite)
        rmse_new = np.sqrt(mean_squared_error(y_test_lite[i], preds_test))
        rmse_new = pd.DataFrame(data=[rmse_new], columns=['RMSE'])
        rmse_new['Y'] = model_y[i]
        rmse = rmse.append(rmse_new)
    
    
        # Predict ......      
        preds = reg.predict(X_predict_lite).tolist()
        # preds = cbyz.li_join_flatten(preds)
        
        results_new = X_predict[model_addt_vars].reset_index(drop=True)
        results_new['VALUES'] = preds
        results_new['Y'] = model_y[i]        
        results = results.append(results_new)

    
    # Reorganize ------
    return_dict = {'RESULTS':results,
                   'FEATURES':features,
                   'RMSE':rmse} 

    return return_dict



# ................
    




def model_6(remove_none=True):
    '''
    XGBoost flatten
    https://www.datacamp.com/community/tutorials/xgboost-in-python
    '''
    
    import xgboost as xgb    
    global stock_symbol, model_x, model_y, model_data, predict_date


    global X_train, y_train, X_test, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite
    

    # Predict ......
    features = pd.DataFrame()
    results = pd.DataFrame()
    rmse = pd.DataFrame()    
    
    
    for i in range(len(model_y)):
        
        # Dataset
        # split_data()
            
        regressor = xgb.XGBRegressor(
            n_estimators=400,
            reg_lambda=1,
            gamma=0,
            max_depth=6
        )
        
        regressor.fit(X_train_lite, y_train_lite[i])

        
        # Feature Importance ......
        features_new = {'FEATURES':list(X_train_lite.columns),
                        'IMPORTANCE':regressor.feature_importances_}
        
        features_new = pd.DataFrame(features_new)            
        features_new['Y'] = model_y[i]
        features = features.append(features_new)        

    
        # RMSE ......
        preds_test = regressor.predict(X_test_lite)
        rmse_new = np.sqrt(mean_squared_error(y_test_lite[i], preds_test))
        rmse_new = pd.DataFrame(data=[rmse_new], columns=['RMSE'])
        rmse_new['Y'] = model_y[i]
        rmse = rmse.append(rmse_new)

    
        # Results ......
        preds = regressor.predict(X_predict_lite)
        results_new = X_predict[model_addt_vars].reset_index(drop=True)
        results_new['VALUES'] = preds
        results_new['Y'] = model_y[i]
        results = results.append(results_new)
        
    
    return_dict = {'RESULTS':results,
                   'FEATURES':features,
                    'RMSE':rmse
                   } 
    

    return return_dict





# ................


def get_model_list(status=[0,1]):
    '''
    List all analysis here
    '''    

    # (1) List manually
    # (2) List by model historic performance
    # (3) List by function pattern
    
    # function_list = [model_5, model_6]
    function_list = [model_6]

    return function_list




def tw_fix_symbol_error():
    '''
    Fix the symbol error caused by csv file, which makes 0050 become 50.
    '''
    
    # Draft
    file = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/1_Data_Collection/2_TEJ/Export/ewprcd_data_2021-06-21_2021-06-30.csv'
    file = pd.read_csv(file)
    
    for i in range(3):
        file['SYMBOL'] = '0' + file['SYMBOL'] 
    
    
