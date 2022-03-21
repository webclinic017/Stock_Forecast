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


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import pickle


host = 3
# host = 2
host = 4
# host = 0


# Path .....
if host == 0:
    # Home
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis'
    path_dcm = '/Users/aron/Documents/GitHub/Stock_Forecast/1_Data_Collection'
    
    
elif host == 2:
    path = '/home/jupyter/Production/2_Stock_Analysis'
    path_dcm = '/home/jupyter/Production/1_Data_Collection'    
    
elif host == 3:
    path = '/home/jupyter/Develop/2_Stock_Analysis'
    path_dcm = '/home/jupyter/Develop/1_Data_Collection'        

elif host == 4:
    # RT
    path = r'D:\Data_Mining\GitHub共用\Stock_Forecast\2_Stock_Analysis'
    path_dcm = r'D:\Data_Mining\GitHub共用\Stock_Forecast\1_Data_Collection'


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'D:\Data_Mining\Projects\Codebase_YZ',
                 r'D:\Data_Mining\GitHub共用\Arsenal',
                 r'/Users/aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20220314',
                 r'/home/jupyter/Arsenal/20220314',
                 path + '/Function']

for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import codebase_ml as cbml
import arsenal as ar
import arsenal_stock as stk
import ultra_tuner_v1_0002 as ut

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
    
    
    global id_keys, time_key
    global symbol, market, var_y, var_y_orig
    global predict_period, time_unit
    global stock_info_raw
    global log, data_form
    global main_data_frame
    global calendar_key
    global market_data_raw, market_data
    
    
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
    
    
    # 避免重複query market_data_raw，在開發或Debug時節省GCP SQL的費用
    if 'market_data_raw' not in globals():
        
        if len(symbol) == 0:
            market_data_raw = \
                stk.get_data(
                    data_begin=loc_begin, 
                    data_end=data_end, 
                    market=market, 
                    symbol=[],
                    ratio_limit=True,
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
                    ratio_limit=True,
                    price_change=True, 
                    price_limit=True,
                    trade_value=True,
                    restore=True
                    )

    market_data = market_data_raw.copy()
    

    # Check        
    global ohlc
    for c in ohlc:
        col = c + '_CHANGE_RATIO'
        min_value = market_data[col].min()
        max_value = market_data[col].max()
        
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
    market_data = select_symbols()


    # main_data_frame ......
    # - market_data will merge calendar in this function
    # - market_data會在這裡merge calendar
    set_frame()


    # Total Trade
    # - 如果time_unit是w，欄位名稱會在df_summary後改變，所以要先刪除
    # - 需要merge calendar_key，所以要寫在set_frame()後面
    # - total_trade會在sam_load_data中用到
    # - 
    # print('Update, total_trade_ratio應該加到get_data中')
    # if trade_value:
    #     total_trade = market_data[['WORK_DATE', 'TOTAL_TRADE_VALUE']] \
    #                     .drop_duplicates()
                        
    #     total_trade = total_trade.merge(calendar_key, on='WORK_DATE')         
    #     market_data = market_data.drop('TOTAL_TRADE_VALUE', axis=1)
        


    # First Trading Day ......
    # - This object will be used at the end of get_model_data
    global first_trading_day
    first_trading_day = market_data[['SYMBOL', 'WORK_DATE']] \
            .sort_values(by=['SYMBOL', 'WORK_DATE'], ascending=True) \
            .drop_duplicates(subset=['SYMBOL']) \
            .rename(columns={'WORK_DATE':'FIRST_TRADING_DAY'})
            

    # Add K line ......
    market_data = market_data \
                    .sort_values(by=['SYMBOL', 'WORK_DATE']) \
                    .reset_index(drop=True)
            
    market_data = stk.add_k_line(market_data)
    market_data = \
        cbml.df_get_dummies(
            df=market_data, 
            cols=['K_LINE_COLOR', 'K_LINE_TYPE']
        )
    
    
    # Add Support Resistance ......
    
    # if support_resist:
    #     # Check，確認寫法是否正確
    #     # print('add_support_resistance - days == True時有bug，全部數值一樣，導致
    #     # 沒辦法標準化？')
    #     global data_period
    #     market_data, _ = \
    #         stk.add_support_resistance(df=market_data, cols='CLOSE',
    #                                    rank_thld=int(data_period * 2 / 360),
    #                                    prominence=4, days=False)

    market_data = market_data \
        .dropna(subset=['WORK_DATE'], axis=0)

    market_data = ar.df_simplify_dtypes(df=market_data)


    # Test ......
    
    # # 執行到這裡，因為加入了預測區間，所以會有NA，但所有NA的數量應該要一樣多
    # na_cols = cbyz.df_chk_col_na(df=market_data)
    # na_min = na_cols['NA_COUNT'].min()
    # na_max = na_cols['NA_COUNT'].max()
    
    # msg = 'Number of NA in each column should be the same.'
    # assert na_min == na_max, msg
    
    print('Check - Temoporaily remove df_chk_col_na')
    
    
    # Check Predict Period
    # chk = market_data.merge(predict_date, on=time_key)
    # chk = chk[time_key].drop_duplicates()
    
    # predict_date_unique = predict_date[time_key].drop_duplicates()
    # assert len(chk) == len(predict_date_unique), 'predict_date error'
    

# ...........


def sam_load_data(industry=True, trade_value=True):
    '''
    讀取資料及重新整理
    '''
    
    global id_keys, var_y, var_y_orig, symbol
    global market_data
    global predict_period, data_end
    global symbol_df, wma
    global stock_info_raw
    global data_form
    global ohlc, ohlc_ratio, ohlc_change
    global df_summary_mean, df_summary_min, df_summary_max
    global df_summary_median, df_summary_std
    
    # New Vars
    global y_scaler
    
    # Scale Market Data ......
    # - loc_main和industry都需要scal ohlc，所以直接在market_data中處理，這樣就
    #   不需要做兩次
    # - 如果y是OHLC Change Ratio的話，by WORK_DATE或SYMBOL的意義不大，反而讓運算
    #    速度變慢，唯一有影響的是一些從來沒有漲跌或跌停的Symbol
    # - 先獨立把y的欄位標準化，因為這一段不用MA，但後面都需要
    # - 這裡的method要用1，如果用2的話，mse會變成0.8
    # - 因為method是1，其他大部份都是0，所以這一段要獨立出來
    # - price和change_ratio都用method 1 scale
    # - Remove open price and open_price_change
    
    market_data, scale_orig_ratio, y_scaler_ratio = \
        cbml.df_scaler(df=market_data, 
                       cols=['HIGH_CHANGE_RATIO', 'LOW_CHANGE_RATIO',
                             'CLOSE_CHANGE_RATIO'],
                       show_progress=False, method=1)
    
    market_data, scale_orig_price, y_scaler_price = \
        cbml.df_scaler(df=market_data,
                       cols=['HIGH', 'LOW', 'CLOSE'],
                       show_progress=False, method=1)
    
    market_data, _, _ = \
        cbml.df_scaler(df=market_data,
                       cols=['OPEN', 'OPEN_CHANGE_RATIO'],
                       show_progress=False, method=1)

    scale_orig = scale_orig_ratio.append(scale_orig_price)
    
    if 'CLOSE_CHANGE_RATIO' in var_y:
        y_scaler = y_scaler_ratio
    elif 'CLOSE' in var_y:
        y_scaler = y_scaler_price
        
    pickle.dump(y_scaler, 
                open(path_temp + '/y_scaler_' + time_unit + '.sav', 'wb'))
    
    loc_main = market_data.copy()
    loc_main = loc_main.drop('TOTAL_TRADE_VALUE', axis=1)
    
    
    # {Y}_ORIG的欄位是用來計算MA
    for y in var_y:
        loc_main.loc[:, y + '_ORIG'] = loc_main[y]

    
    # Process Data
    if data_form == 1:
            
        # Scale
        # - 即使time_unit是w，也需要MA，因為df_summary只考慮當周，不會考慮
        #   更久的MA
        cols = cbyz.df_get_cols_except(
            df=loc_main, 
            except_cols=ohlc+ohlc_ratio+['SYMBOL', 'WORK_DATE']
            )
        
        loc_main, _, _ = cbml.df_scaler(df=loc_main, cols=cols,
                                        show_progress=False, method=0)
            
        # MA
        cols = \
            cbyz.df_get_cols_except(
                df=loc_main, 
                except_cols=['SYMBOL', 'WORK_DATE'] + var_y
                )
            
        loc_main, ma_cols_done = \
            cbyz.df_add_ma(df=loc_main, cols=cols,
                           group_by=['SYMBOL'], 
                           date_col='WORK_DATE',
                           values=ma_values,
                           wma=wma, 
                           show_progress=False
                           )
        loc_main = loc_main.drop(cols, axis=1)
        
        
    elif data_form == 2:
        
        except_cols = ['WORK_DATE', 'YEAR_ISO', 'MONTH',
                       'WEEKDAY', 'WEEK_NUM_ISO'] + id_keys  

        # 新股會有NA，但直接drop的話會刪到pred preiod
        loc_main.loc[:, 'REMOVE'] = \
            np.where((loc_main['WORK_DATE']<data_end) \
                     & (loc_main[var_y[-1]].isna()), 1, 0)
                
        loc_main = loc_main[loc_main['REMOVE']==0] \
                    .drop('REMOVE', axis=1)                
        
        # 新股上市的第一天，OPEN_CHANGE會是inf
        print('Check - 為什麼OPEN_CHANGE_RATIO不是inf，但OPEN_CHANGE_ABS_RATIO是')
        loc_main.loc[:, 'OPEN_CHANGE_ABS_RATIO'] = \
            np.where(loc_main['OPEN_CHANGE_ABS_RATIO']==np.inf, 
                     0, loc_main['OPEN_CHANGE_ABS_RATIO'])

        loc_main, _, _ = \
            cbml.df_scaler(
                df=loc_main,
                except_cols=except_cols,
                show_progress=False,
                method=0
                )  

        # loc_main, _ = \
        #     cbml.ml_df_to_time_series(
        #         df=loc_main, 
        #         cols=[], 
        #         except_cols=except_cols, 
        #         group_by=[],
        #         sort_keys=id_keys, 
        #         window=predict_period,
        #         drop=False)        
    
        
    
    # Total Market Trade
    if trade_value:
        
        total_trade = market_data[['WORK_DATE', 'TOTAL_TRADE_VALUE']] \
                        .drop_duplicates(subset=['WORK_DATE'])


        if data_form == 1:    
            
            # Scale Data
            total_trade, _, _ = cbml.df_scaler(df=total_trade, 
                                            cols='TOTAL_TRADE_VALUE',
                                            method=0)
            # MA
            total_trade, _ = \
                cbyz.df_add_ma(df=total_trade, cols='TOTAL_TRADE_VALUE',
                               group_by=[], date_col='WORK_DATE',
                               values=ma_values, wma=wma, 
                               show_progress=False
                               )   
                
            total_trade = total_trade.drop('TOTAL_TRADE_VALUE', axis=1)
                
        elif data_form == 2:
                
            total_trade, _, _ = \
                cbml.df_scaler(
                    df=total_trade,
                    except_cols=['WORK_DATE'],
                    show_progress=False,
                    method=0
                    )  
        
            # total_trade, _ = \
            #     cbml.ml_df_to_time_series(
            #         df=total_trade, 
            #         cols=[], 
            #         except_cols='WORK_DATE',
            #         group_by=[],
            #         sort_keys='WORK_DATE', 
            #         window=1,
            #         drop=True)    
        
        loc_main = loc_main.merge(total_trade, how='left', on=['WORK_DATE'])  


    # Stock Info ...
    stock_info = stock_info_raw.drop('INDUSTRY_ONE_HOT', axis=1)
    stock_info, _, _ = \
        cbml.df_scaler(
            df=stock_info,
            except_cols=['SYMBOL'],
            show_progress=False,
            method=0
            )      
    
    loc_main = loc_main.merge(stock_info, how='left', on=['SYMBOL'])      
    

    # Industry ......       
    # - 因為industry_data中會用到TOTAL_TRADE_VALUE，所以TOTAL_TRADE_VALUE沒辦法
    #   先獨立處理
    if industry: 
        stock_industry = stock_info_raw[['SYMBOL', 'INDUSTRY_ONE_HOT']]
        
        stock_info_dummy = \
            cbml.df_get_dummies(df=stock_industry, 
                                cols='INDUSTRY_ONE_HOT'
                                )
        
        # Industry Data And Trade Value ...
        # print('sam_load_data - 當有新股上市時，產業資料的比例會出現大幅變化，' \
        #       + '評估如何處理')
        industry_data = \
            market_data[['SYMBOL', 'WORK_DATE', 'VOLUME'] \
                        + ohlc + ohlc_change + \
                        ['SYMBOL_TRADE_VALUE', 'TOTAL_TRADE_VALUE']]

        # Merge        
        industry_data = industry_data.merge(stock_industry, on='SYMBOL')
        
        industry_data['TRADE_VALUE'] = \
            industry_data \
            .groupby(['WORK_DATE', 'INDUSTRY_ONE_HOT'])['SYMBOL_TRADE_VALUE'] \
            .transform('sum')

        industry_data['TRADE_VALUE_RATIO'] = \
            industry_data['TRADE_VALUE'] / industry_data['TOTAL_TRADE_VALUE']            
        
        industry_data = industry_data[['WORK_DATE', 'INDUSTRY_ONE_HOT'] \
                                       + ohlc + ohlc_change + \
                                       ['TRADE_VALUE', 'TRADE_VALUE_RATIO']]
        
        industry_data = industry_data \
                        .groupby(['WORK_DATE', 'INDUSTRY_ONE_HOT']) \
                        .mean() \
                        .reset_index()
        
        # Rename ...
        cols = cbyz.df_get_cols_except(
            df=industry_data,
            except_cols=['WORK_DATE', 'INDUSTRY_ONE_HOT']
            )
        
        new_cols = ['INDUSTRY_' + c for c in cols]                  
        rename_dict = cbyz.li_to_dict(cols, new_cols)
        industry_data = industry_data.rename(columns=rename_dict)
                       
        # MA ......
        cols = cbyz.df_get_cols_except(
            df=industry_data, 
            except_cols=['WORK_DATE', 'INDUSTRY_ONE_HOT']
            )
        
        industry_data, _ = \
            cbyz.df_add_ma(df=industry_data, cols=cols,
                           group_by=[], date_col='WORK_DATE',
                           values=ma_values, wma=wma, 
                           show_progress=False
                           )    
        industry_data = industry_data.drop(cols, axis=1)   
        
        # Merge ...
        # .merge(stock_info_dummy, how='left', on='SYMBOL') \
        loc_main = loc_main \
            .merge(stock_industry, how='left', on='SYMBOL') \
            .merge(industry_data, how='left', on=['WORK_DATE', 'INDUSTRY_ONE_HOT']) \
            .drop('INDUSTRY_ONE_HOT', axis=1)
        

    # backup = loc_main.copy()
    # loc_main = backup.copy()

    if time_unit == 'w':

        # Merge Market Data And Calendar
        new_loc_main = main_data_frame \
            .merge(calendar_key, how='left', on=time_key)
        
        
        # 這裡merge完後，shift_period的YEAR_ISO和WEEK_NUM_ISO中會有NA，且shift_period
        # 已經超出calendar的範圍，這是正常的
        loc_main = new_loc_main \
            .merge(loc_main, how='outer', on=['SYMBOL', 'WORK_DATE'])


        # merge完main_data_frame後，新股的WORK_DATE會是NA
        loc_main = loc_main.dropna(subset=['WORK_DATE'] + time_key, axis=0)
        loc_main = loc_main.drop('WORK_DATE', axis=1)
        
        cols = cbyz.df_get_cols_except(df=loc_main, 
                                       except_cols=id_keys + var_y)
        
        
        # - df_summary中的groupby會讓var_y消失，所以需要先獨立出來，而且也
        #   必須df_summary
        # Hyperparameter，目前y_data用mean aggregate，不確定median會不會比較好
        y_data = loc_main[id_keys + var_y]
        
        y_data = y_data \
                .groupby(id_keys) \
                .mean() \
                .reset_index()
        
        
        # y用平均就好，不要用high of high, low of low，避免漲一天跌四天
        print('skew很容易產生NA，先移除 / 還是每個skew的第一個數值都是NA？')
        
        
        loc_main, _ = \
            cbyz.df_summary(df=loc_main, cols=cols, 
                            group_by=id_keys, 
                            add_mean=df_summary_mean, 
                            add_min=df_summary_min, 
                            add_max=df_summary_max, 
                            add_median=df_summary_median,
                            add_std=df_summary_std, 
                            add_skew=False, add_count=False, quantile=[])
            
        loc_main = loc_main.merge(y_data, how='left', on=id_keys)
        del y_data
        gc.collect()
        
    elif time_unit == 'd':
        
        # 這裡merge完後，shift_period的YEAR和WEEK_NUM_ISO中會有NA，且shift_period
        # 已經超出calendar的範圍，這是正常的
        loc_main = main_data_frame \
            .merge(loc_main, how='outer', on=['SYMBOL', 'WORK_DATE'])        


    # backup2 = loc_main.copy()
    # loc_main = backup2.copy()

    # Shift Y ......
    # 1. 因為X與y之間需要做shift，原始的版本都是移動X，但是X的欄位很多，因此改成
    #    移動y，提升計算效率
    loc_main, new_cols = \
        cbyz.df_add_shift(
            df=loc_main, 
            cols=time_key + var_y, 
            shift=-predict_period, 
            group_by=['SYMBOL'], 
            sort_by=id_keys,
            suffix='', 
            remove_na=False
            )
        
    # 因為One Hot Encoding後的Industry欄位不用df_summary，也不用shift，所以最後
    # 再merge即可
    if industry:
        loc_main = loc_main.merge(stock_info_dummy, how='left', on='SYMBOL')


    # 檢查shift完後YEAR和WEEK_NUM_ISO的NA
    chk_na = cbyz.df_chk_col_na(df=loc_main, cols=time_key)
    na_min = chk_na['NA_COUNT'].min()
    na_max = chk_na['NA_COUNT'].max()
    
    assert na_min == na_max and na_min == len(symbol_df) * predict_period, \
        'Error after shift'


    # 必須在df_add_shift才dropna，否則準備往前推的WORK_DATE會被刪除；shift完後，
    # id_keys中會有NA，但這裡的NA就可以drop
    cols = cbyz.df_get_cols_except(df=loc_main, except_cols=var_y)
    
    
    # 因為計算MA及df_shift，所以開頭會有NA，time_key也因為NA，所以變成float，
    # 因此drop後要simplify_dtypes
    if data_form == 1:
        loc_main = loc_main.dropna(subset=cols, axis=0)
    elif data_form == 2:
        pass

    loc_main = ar.df_simplify_dtypes(df=loc_main)


    # Simplify dtypes
    # - YEAR and WEEK_NUM_ISO will be float here
    loc_main = ar.df_simplify_dtypes(df=loc_main)


    # Check NA ......
    # 有些新股因為上市時間較晚，在MA_LAG中會有較多的NA，所以只處理MA的欄位
    na_cols = cbyz.df_chk_col_na(df=loc_main, except_cols=var_y)
    # na_cols = cbyz.df_chk_col_na(df=loc_main)
    
    
    
    # df_summary可能造成每一欄的NA數不一樣，所以先排除time_unit = 'w'
    
    if time_unit == 'd':

        assert len(na_cols) == 0 \
            or na_cols['NA_COUNT'].min() == na_cols['NA_COUNT'].max(), \
            'All the NA_COUNT should be the same.'
    
        na_cols = na_cols['COLUMN'].tolist()
        loc_main = loc_main.dropna(subset=na_cols, axis=0)
        
    elif time_unit == 'w':
        pass
        
    # Check for min max
    # - Check, temporaily remove for data_from 2
    # chk_min_max = cbyz.df_chk_col_min_max(df=loc_main)
    # chk_min_max = chk_min_max[chk_min_max['COLUMN']!='WORK_DATE']
    # chk_min_max = chk_min_max[(chk_min_max['MIN_VALUE']<0) \
    #                           | (chk_min_max['MAX_VALUE']>1)]
        
    # assert len(chk_min_max) == 0, 'chk_min_max failed'
    
    return loc_main, scale_orig


# ...........



# .............


def select_symbols():

    '''
    Version Note
    
    1. Exclude small capital

    '''    

    global market_data
    global stock_info_raw


    # Exclude ETF ......
    all_symbols = stock_info_raw[['SYMBOL']]
    df = all_symbols.merge(market_data, on=['SYMBOL']) 


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



def set_frame():
    
    
    # Merge As Main Data ......
    global symbol_df, id_keys, time_key, time_unit
    global market_data, calendar, calendar_proc
    global predict_week, predict_date
    global main_data_frame, main_data_frame_calendar

    # New Global Variables
    global symbol_df
    global calendar_lite, calendar_proc, calendar_key


    # Predict Symbols ......
    # 1. Prevent symbols excluded by select_symbols(), but still exists.
    all_symbols = market_data['SYMBOL'].unique().tolist()
    symbol_df = pd.DataFrame({'SYMBOL':all_symbols})
    
    
    # Calendar ......
    calendar_proc = calendar[calendar['TRADE_DATE']>0] \
                    .reset_index(drop=True) \
                    .reset_index() \
                    .rename(columns={'index':'DATE_INDEX'})
                   
                    
    calendar_lite = calendar[calendar['TRADE_DATE']>0]                     
    calendar_lite = calendar_lite[time_key] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
                    
               
    # Remove untrading date
    calendar_key = calendar[calendar['TRADE_DATE']>0].reset_index(drop=True)
    if time_unit == 'd':
        calendar_key = calendar_key[['WORK_DATE']]
        
    elif time_unit == 'w':
        calendar_key = calendar_key[['WORK_DATE', 'YEAR_ISO', 'WEEK_NUM_ISO']]
    
    
    # Duplicate year and week_num, then these two columns can be variables 
    # of the model
    if time_unit == 'w':
        
        calendar_proc = calendar_proc[['YEAR_ISO', 'MONTH', 
                                       'WEEK_NUM_ISO', 'TRADE_DATE']]
        
        calendar_proc = calendar_proc \
            .drop_duplicates() \
            .reset_index(drop=True) \
            .reset_index() \
            .rename(columns={'index':'DATE_INDEX'}) 
        
        calendar_proc.loc[:, 'YEAR_DUP'] = calendar_proc['YEAR_ISO']
        calendar_proc.loc[:, 'WEEK_NUM_DUP'] = calendar_proc['WEEK_NUM_ISO']
        

    calendar_proc, _, _ = \
        cbml.df_scaler(
            df=calendar_proc,
            except_cols=id_keys + ['TRADE_DATE'],
            show_progress=False,
            method=1
            )           
    
    
    # main_data_frame ......
    main_data_frame = cbyz.df_cross_join(symbol_df, calendar_lite)
    
    
    if time_unit == 'd':
        max_date = predict_date['WORK_DATE'].max()
        main_data_frame = \
            main_data_frame[main_data_frame['WORK_DATE']<=max_date]
            
    elif time_unit == 'w':
        pass
        print('chk - 這裡是否需要篩選日期')


    # 20220208 - 移到sam_load_data    
    # market_data = main_data_frame \
    #     .merge(market_data, how='left', on=['SYMBOL', 'WORK_DATE'])

    
    # Organize
    main_data_frame = main_data_frame[id_keys]
    
    global debug
    debug = main_data_frame.copy()
    
    main_data_frame = ar.df_simplify_dtypes(df=main_data_frame)
    
    print('Check - main_data_frame_calendar是否等同calendar_lite')
    main_data_frame_calendar = calendar_lite.copy()
    # main_data_frame_calendar = main_data_frame[time_key] \
    #                             .drop_duplicates() \
    #                             .sort_values(by='WORK_DATE') \
    #                             .reset_index(drop=True)



# %% TW Variables ------


def sam_buffett_indicator():
    
    global id_keys, time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma, corr_threshold
    
    result = stk.cal_buffett_indicator(
            end_date=predict_date.loc[len(predict_date) - 1, 'WORK_DATE']
            )
    
    cols = cbyz.df_get_cols_except(df=result,
                                   except_cols='WORK_DATE')
    
    # backup = buffett_indicator.copy()
    # buffett_indicator = backup.copy()

    # Scale Data
    result, _, _ = \
        cbml.df_scaler(df=result, cols=cols, method=0)
        
    # MA
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=[], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
        
    result = result.drop(cols, axis=1)

    if time_unit == 'w':

        result = calendar_full_key \
            .merge(result, how='left', on='WORK_DATE') \
            .drop(['WORK_DATE'], axis=1)

        result, ma_cols = \
            cbyz.df_summary(
                df=result, cols=ma_cols, group_by=time_key, 
                add_mean=True, add_min=True, 
                add_max=True, add_median=True, add_std=True, 
                add_skew=False, add_count=False, quantile=[]
                )    
            

    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)
         
    # Filter exist columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)

    # Merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key)    

    return result, ma_cols


# .........


def sam_covid_19_tw():
    
    global id_keys, time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma, corr_threshold
    global df_summary_mean, df_summary_min, df_summary_max
    global df_summary_median, df_summary_std    
    
    
    result, _ = cbyz.get_covid19_data(backup=True, path=path_temp)
    
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['WORK_DATE', 'YEAR_ISO', 'WEEK_NUM_ISO']
        )
    
    # Scale Data
    result, _, _ = cbml.df_scaler(df=result, 
                                    cols=cols,
                                    method=0)
    # MA
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=[], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
        
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='WORK_DATE')
            
        result = cbyz.df_conv_na(df=result, cols=ma_cols)  
        
        result, ma_cols = \
            cbyz.df_summary(
                df=result, cols=ma_cols, group_by=time_key, 
                add_mean=df_summary_mean, 
                add_min=df_summary_min, 
                add_max=df_summary_max, 
                add_median=df_summary_median,
                add_std=df_summary_std, 
                add_skew=False, add_count=False, quantile=[]
                )

    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                     except_cols=id_keys)
        
    # Filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)
    
    
    # Merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key)    

    return result, ma_cols


# .................
  
    
def sam_ex_dividend():
    
    global id_keys, time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma, corr_threshold
    
    
    # # Close Lag ...
    # daily_close = market_data[['WORK_DATE', 'SYMBOL', 'CLOSE']]
    
    # daily_close, _ = \
    #     cbyz.df_add_shift(df=daily_close, 
    #                       sort_by=['SYMBOL', 'WORK_DATE'],
    #                       cols='CLOSE', shift=1,
    #                       group_by=['SYMBOL'],
    #                       suffix='_LAG', 
    #                       remove_na=False)
        
    # daily_close = daily_close \
    #             .drop('CLOSE', axis=1) \
    #             .rename(columns={'CLOSE_LAG':'CLOSE'})

    # daily_close = cbyz.df_fillna_chain(df=daily_close, cols='CLOSE',
    #                                    sort_keys=['SYMBOL', 'WORK_DATE'],
    #                                    group_by=['SYMBOL'],
    #                                    method=['ffill', 'bfill'])

    print('sam_ex_dividend - Only return date')
    result = stk.od_tw_get_ex_dividends()
    cols = cbyz.df_get_cols_except(df=result, 
                                   except_cols=['SYMBOL', 'WORK_DATE'])
    
    cbyz.df_chk_col_na(df=result, mode='stop')
    
    # Scale Data
    # result, _, _ = cbml.df_scaler(df=result, 
    #                                 cols=cols,
    #                                 method=0)
    
    # MA
    # result, ma_cols = \
    #     cbyz.df_add_ma(df=result, cols=cols,
    #                    group_by=[], date_col='WORK_DATE',
    #                    values=ma_values, wma=wma, 
    #                    show_progress=False
    #                    )   
        
    # result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
                .merge(calendar_full_key, how='left', on='WORK_DATE') \
                .drop('WORK_DATE', axis=1) \
                .dropna(subset=time_key, axis=0)
            
        cbyz.df_chk_col_na(df=result, mode='stop')            
            
        result = result \
                .groupby(id_keys) \
                .max() \
                .reset_index()
                
                
    # Drop Highly Correlated Features                
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)
        
    # Filter existing columns
    cols = cbyz.df_filter_exist_cols(df=result, cols=cols)    
            
    # Merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key) \
                .dropna(subset=['SYMBOL'], axis=0)

    result = cbyz.df_conv_na(df=result, cols=cols, value=0)
    result = ar.df_simplify_dtypes(df=result)

    return result, cols


# .................


def sam_tw_gov_invest(dev=False):
    
    global symbol_df, calendar_key
    global id_keys, time_key, var_y, corr_threshold
    
    result = stk.od_tw_get_gov_invest(path=path_resource)
    
    if not dev:
        result = result.merge(symbol_df, on='SYMBOL')
    else:
        # 這個結果merge後需要再fillna by ffill，為了避免和model_data沒有交集，
        # 所以增加dev mode
        test_symbol = symbol[0:len(result)]
        result['SYMBOL'] = test_symbol
    
    if time_unit == 'w':
        result = result.merge(calendar_key, how='left', on='WORK_DATE')
        result = result.drop('WORK_DATE', axis=1)
        
        
    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)
    
    
    print('是否需要return cols，並df_filter_exist_cols')
    
    return result
    

# .................
    

def sam_tw_gov_own(dev=False):
    
    global id_keys, time_key, symbol_df, calendar_key
    global corr_threshold
    
    result = stk.od_tw_get_gov_own(path=path_resource)
    
    if not dev:
        result = result.merge(symbol_df, on='SYMBOL')
    else:
        # 這個結果merge後需要再fillna by ffill，為了避免和model_data沒有交集，
        # 所以增加dev mode
        test_symbol = symbol[0:len(result)]
        result['SYMBOL'] = test_symbol
    
    if time_unit == 'w':
        result = result.merge(calendar_key, how='left', on='WORK_DATE')
        result = result.drop('WORK_DATE', axis=1)
        
    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)     
    
    print('是否需要return cols，並df_filter_exist_cols')
    
    return result
    

# .................
    

def sam_od_tw_get_fx_rate():
    
    global id_keys, time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma, corr_threshold
    
    result = stk.od_tw_get_fx_rate()
    cols = cbyz.df_get_cols_except(df=result, except_cols=['WORK_DATE'])
    result = pd.melt(result, id_vars=['WORK_DATE'], value_vars=cols)
    
    # Scale
    result, _, _ = \
        cbml.df_scaler(df=result, cols='value', show_progress=False, method=1)
        
    result = result \
            .pivot_table(index=['WORK_DATE'], columns='variable',
                         values='value') \
            .reset_index()

    result = result.reset_index(drop=True)
    
    
    # MA
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=[], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='WORK_DATE')
            
        result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                      sort_keys='WORK_DATE', 
                                      method=['ffill', 'bfill'], 
                                      group_by=[])
        
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=time_key, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=False, add_count=False, quantile=[]
            )
        
    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)   
    
    # Filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)
        
    return result, ma_cols       


# .................


def sam_od_tw_get_index(begin_date, end_date):
    
    global ma_values, predict_period, corr_threshold
    global id_keys, time_unit, time_key
    global calendar, main_data_frame_calendar
    
    # main_data_frame_calendar
    # calendar_full_key
    
    result = stk.od_tw_get_index()
    cols = cbyz.df_get_cols_except(df=result, except_cols='WORK_DATE')
    
    # 如果有NA的話，可能要先做fillna
    cbyz.df_chk_col_na(df=result, mode='stop')
    
    # Scale Data
    result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
    # MA
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=[], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
        
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        
        result = result \
            .merge(calendar_full_key, how='left', on='WORK_DATE') \
            .drop(['WORK_DATE'], axis=1)
            
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=time_key, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=False, add_count=False, quantile=[]
            )
    
    
    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)    
    
    # Filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    
    
    # Merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key)   

    return result, ma_cols


# .................


def sam_od_us_get_dji():
    
    # Dow Jones Industrial Average (^DJI)
    global id_keys, corr_threshold
    global ma_values, predict_period, predict_date
    global calendar, main_data_frame_calendar
    
    loc_df = stk.od_us_get_dji(daily_backup=True, path=path_temp)
    cols = cbyz.df_get_cols_except(df=loc_df, except_cols=['WORK_DATE'])
    
    # Handle Time Lag
    # - 20220317 - 移至stk，待確認是否會出錯，沒問題的話這一段就刪除
    # loc_df = loc_df.rename(columns={'WORK_DATE':'WORK_DATE_ORIG'})
    # loc_df = cbyz.df_date_cal(df=loc_df, amount=-1, unit='d',
    #                           new_cols='WORK_DATE',
    #                           cols='WORK_DATE_ORIG')
    
    # loc_df = loc_df.drop('WORK_DATE_ORIG', axis=1)


    # Fillna .......
    # 1. 因為美股的交易時間可能和台灣不一樣，包含特殊節日等，為了避免日期無法
    #    對應，用fillna
    #    補上完整的日期
    # - 20220317 - 移至stk，待確認是否會出錯，沒問題的話這一段就刪除
    
    # loc_calendar = cbyz.date_get_calendar(begin_date=begin_date, 
    #                                       end_date=predict_date[-1])
    # loc_calendar = calendar[['WORK_DATE']]
    # loc_df = loc_calendar.merge(loc_df, how='left', on='WORK_DATE')
    # cols = cbyz.df_get_cols_except(df=loc_df, except_cols='WORK_DATE')
    # loc_df = cbyz.df_fillna(df=loc_df, cols=cols, sort_keys='WORK_DATE', 
    #                         group_by=[], method='ffill')

    # Scale
    loc_df, _, _ = cbml.df_scaler(df=loc_df, cols=cols, method=0)
    
    # MA
    loc_df, ma_cols = \
        cbyz.df_add_ma(df=loc_df, cols=cols,
                       group_by=[], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
    loc_df = loc_df.drop(cols, axis=1)
    
    
    # Agg for Weekly Prediction
    if time_unit == 'w':
        loc_df = loc_df \
            .merge(calendar_full_key, how='left', on='WORK_DATE')
        
        loc_df, cols = cbyz.df_summary(
            df=loc_df, cols=ma_cols, group_by=time_key, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=False, add_count=False, quantile=[]
            )    
    

    # Drop Highly Correlated Features
    loc_df = cbml.df_drop_high_corr_var(df=loc_df, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # Filter existing columns
    cols = cbyz.df_filter_exist_cols(df=loc_df, cols=cols)    
        
    return loc_df, cols    


# .................


def sam_od_us_get_snp(begin_date):
    
    global id_keys, corr_threshold
    global ma_values, predict_period, predict_date
    global calendar, main_data_frame_calendar
    
    loc_df = stk.od_us_get_snp(daily_backup=True, path=path_temp)
    cols = cbyz.df_get_cols_except(df=loc_df, except_cols=['WORK_DATE'])
    
    # Handle Time Lag
    # - 20220317 - 移至stk，待確認是否會出錯，沒問題的話這一段就刪除
    # loc_df = loc_df.rename(columns={'WORK_DATE':'WORK_DATE_ORIG'})
    # loc_df = cbyz.df_date_cal(df=loc_df, amount=-1, unit='d',
    #                           new_cols='WORK_DATE',
    #                           cols='WORK_DATE_ORIG')
    
    # loc_df = loc_df.drop('WORK_DATE_ORIG', axis=1)


    # Fillna .......
    # 1. 因為美股的交易時間可能和台灣不一樣，包含特殊節日等，為了避免日期無法對應，用fillna
    #    補上完整的日期
    # - 20220317 - 移至stk，待確認是否會出錯，沒問題的話這一段就刪除
    
    # loc_calendar = cbyz.date_get_calendar(begin_date=begin_date, 
    #                                       end_date=predict_date[-1])
    # loc_calendar = calendar[['WORK_DATE']]
    
    # loc_df = loc_calendar.merge(loc_df, how='left', on='WORK_DATE')
    # cols = cbyz.df_get_cols_except(df=loc_df, except_cols='WORK_DATE')
    # loc_df = cbyz.df_fillna(df=loc_df, cols=cols, sort_keys='WORK_DATE', 
    #                         group_by=[], method='ffill')

    # Scale
    loc_df, _, _ = cbml.df_scaler(df=loc_df, cols=cols, method=0)
    
    # MA
    loc_df, ma_cols = \
        cbyz.df_add_ma(df=loc_df, cols=cols,
                       group_by=[], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
    loc_df = loc_df.drop(cols, axis=1)
    
    
    # Agg for Weekly Prediction
    if time_unit == 'w':
        loc_df = loc_df \
            .merge(calendar_full_key, how='left', on='WORK_DATE')
        
        loc_df, cols = cbyz.df_summary(
            df=loc_df, cols=ma_cols, group_by=time_key, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=False, add_count=False, quantile=[]
            )    
    

    # Drop Highly Correlated Features
    loc_df = cbml.df_drop_high_corr_var(df=loc_df, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # Filter existing columns
    cols = cbyz.df_filter_exist_cols(df=loc_df, cols=cols)    
        
    return loc_df, cols


# ...............


def sam_tej_get_ewsale(begin_date):

    
    print('還沒加df_drop_high_corr_var')
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


# ...........


def tej_get_ewgin():
    
    global id_keys, corr_threshold
    
    result = stk.tej_get_ewgin(begin_date=shift_begin, end_date=None, 
                               symbol=symbol)
    
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['SYMBOL', 'WORK_DATE']
        )
    
    # Scale Data
    result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
    # MA
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['SYMBOL'], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
        
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='WORK_DATE')
            
        result = cbyz.df_conv_na(df=result, cols=ma_cols)  
        
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=id_keys, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=False, add_count=False, quantile=[]
            )    
    
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='SYMBOL')

    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # Filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    

    return result, ma_cols   


# ...........


def sam_tej_get_ewifinq():

    global ma_values, predict_period, predict_date
    global calendar, main_data_frame_calendar
    global symbol
    
    print('還沒加df_drop_high_corr_var')    
    result = stk.tej_get_ewifinq(fill_date=True, target_type='symbol',
                                 target=symbol)

    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['SYMBOL', 'WORK_DATE']
        )
    
    # Scale Data
    result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
    
    if time_unit == 'w':
        result = result.merge(calendar_full_key, on='WORK_DATE')
        cbyz.df_chk_col_na(df=result, mode='stop')
        
    result = main_data_frame.merge(result, how='left', on=id_keys)
    return result, cols


# ..............
    

def sam_tej_get_ewtinst1():
    
    global id_keys, corr_threshold
    
    result = stk.tej_get_ewtinst1(begin_date=shift_begin, end_date=None, 
                                  symbol=symbol)
    
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['SYMBOL', 'WORK_DATE']
        )
    
    # Scale Data
    result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
    # MA
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['SYMBOL'], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
        
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='WORK_DATE')
            
        result = cbyz.df_conv_na(df=result, cols=ma_cols)  
        
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=id_keys, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=False, add_count=False, quantile=[]
            )    
    
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='SYMBOL')


    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # Filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    

    return result, ma_cols   



# .............


def sam_tej_get_ewtinst1c():
    
    global id_keys, corr_threshold
    
    result = stk.tej_get_ewtinst1c(begin_date=shift_begin, end_date=None, 
                                   symbol=symbol, trade=True)
    
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['SYMBOL', 'WORK_DATE']
        )
    
    # Scale Data
    result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
    # MA
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['SYMBOL'], date_col='WORK_DATE',
                       values=ma_values, wma=wma, 
                       show_progress=False
                       )   
        
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='WORK_DATE')
            
        result = cbyz.df_conv_na(df=result, cols=ma_cols)  
        
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=id_keys, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=False, add_count=False, quantile=[]
            )    
    
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='SYMBOL')


    # Drop Highly Correlated Features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # Filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    

    return result, ma_cols    



# %% Process ------


def get_model_data(industry=True, trade_value=True, load_file=False):
    
    
    global shift_begin, shift_end, data_begin, data_end, ma_values
    global corr_threshold
    global predict_date, predict_week, predict_period
    global calendar, calendar_lite, calendar_proc
    global calendar_key, calendar_full_key
    global main_data_frame, main_data_frame_calendar, market_data
    global symbol
    global var_y
    global params, error_msg
    global id_keys
    global main_data
    global time_unit, y_scaler
    
    
    main_data_file = path_temp + '/model_data_' + time_unit + '.csv'
    model_x_file = path_temp + '/model_x_' + time_unit + '.csv'
    scale_orig_file = path_temp + '/scale_orig_' + time_unit + '.csv'
    y_scaler_file = path_temp + '/y_scaler_' + time_unit + '.sav'    

    
    if load_file:
        
        # 為了避免跨日的問題，多計算一天
        today = cbyz.date_get_today()
        
        main_data_mdate = cbyz.os_get_file_modify_date(main_data_file)
        main_data_mdate = cbyz.date_cal(main_data_mdate, 1, 'd')
        main_data_diff = cbyz.date_diff(today, main_data_mdate, absolute=True)        

        model_x_mdate = cbyz.os_get_file_modify_date(model_x_file)
        model_x_mdate = cbyz.date_cal(model_x_mdate, 1, 'd')
        model_x_diff = cbyz.date_diff(today, model_x_mdate, absolute=True)
        
        norm_mdate = cbyz.os_get_file_modify_date(scale_orig_file)
        norm_mdate = cbyz.date_cal(norm_mdate, 1, 'd')
        norm_diff = cbyz.date_diff(today, norm_mdate, absolute=True)        


        # Ensure files were saved recently
        if main_data_diff <= 2 and  model_x_diff <= 2 and norm_diff <= 2:
            try:
                model_data = pd.read_csv(main_data_file)
                model_x = cbyz.li_read_csv(model_x_file)
                scale_orig = pd.read_csv(scale_orig_file)
                y_scaler = pickle.load(open(y_scaler_file, 'rb'))
                
            except Exception as e:
                print('get_model_data - fail to load files.')                
                print(e)
                
            else:
                print('get_model_data - load files successfully.')
                return model_data, model_x, scale_orig                
    

    # Check ......
    msg = ('get_model_data - predict_period is longer than ma values, '
            'and it will cause na.')
    assert predict_period <= min(ma_values), msg


    # Symbols ......
    symbol = cbyz.conv_to_list(symbol)
    symbol = cbyz.li_conv_ele_type(symbol, 'str')


    # Market Data ......
    # market_data
    get_market_data_raw(trade_value=trade_value)
    gc.collect()
    
    
    # Load Historical Data ......
    global main_data_raw
    main_data_raw, scale_orig = \
        sam_load_data(industry=industry, trade_value=trade_value) 
        
    main_data = main_data_raw.copy()
    cbyz.df_chk_col_na(df=main_data_raw)
    
    
    # Merge Calendar_Proc
    main_data = main_data.merge(calendar_proc, how='left', on=time_key)
    
    
    # TODC Shareholdings Spread ......
    # sharehold = stk.tdcc_get_sharehold_spread(shift_begin, end_date=None,
    #                                           local=local) 
    
    # main_data = main_data.merge(sharehold, how='left', 
    #                           on=['SYMBOL', 'WORK_DATE'])      


    # TEJ EWTINST1 - Transaction Details of Juridical Persons ......
    if market == 'tw':
        ewtinst1, cols = sam_tej_get_ewtinst1()
        main_data = main_data.merge(ewtinst1, how='left', on=id_keys)
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=['SYMBOL'])   
        del ewtinst1
        gc.collect()


    # TEJ EWGIN ......
    if market == 'tw':
        ewgin, cols = tej_get_ewgin()
        main_data = main_data.merge(ewgin, how='left', on=id_keys)
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=['SYMBOL'])          
        del ewgin
        gc.collect()


    # Ex-Dividend And Ex-Right ...
    if market == 'tw':
        data, cols = sam_ex_dividend()
        main_data = main_data.merge(data, how='left', on=id_keys)
        main_data = cbyz.df_conv_na(df=main_data, cols=cols, value=0)


    # Buffett Indicator ......
    if market == 'tw':
        
        buffett_indicator, cols = sam_buffett_indicator()
        
        # 因為部份欄位和下面的tw_index重複，所以刪除
        drop_cols = cbyz.df_get_cols_contains(df=buffett_indicator, 
                                              string=['TW_INDEX'])
        
        buffett_indicator = buffett_indicator.drop(drop_cols, axis=1)
        cols = cbyz.li_remove_items(cols, drop_cols)
        
        # Merge
        main_data = main_data \
                    .merge(buffett_indicator, how='left', on=time_key)

        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=[])


    # Government Invest ......
    if market == 'tw':
        # gov_invest = sam_tw_gov_invest(dev=True)
        gov_invest = sam_tw_gov_invest(dev=False)
        cols = cbyz.df_get_cols_except(df=gov_invest, except_cols=id_keys)
        
        # 沒有交集時就不merge，避免一整欄都是NA
        if len(gov_invest) > 0:
            main_data = main_data.merge(gov_invest, how='left', on=id_keys)
            main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                        sort_keys=id_keys, group_by=['SYMBOL'], 
                                        method='ffill')
            
            # 避免開頭的資料是NA，所以再用一次bfill
            main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                        sort_keys=id_keys, group_by=['SYMBOL'], 
                                        method='bfill')                        
    
            main_data = cbyz.df_conv_na(df=main_data, cols=cols, value=0)
    
    
    # Government Own ......
    if market == 'tw':
        # gov_own = sam_tw_gov_own(dev=True)
        gov_own = sam_tw_gov_own(dev=False)
        cols = cbyz.df_get_cols_except(df=gov_own, except_cols=id_keys)
        
        # 沒有交集時就不merge，避免一整欄都是NA
        if len(gov_own) > 0:
            main_data = main_data.merge(gov_own, how='left', on=id_keys)
            main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                        sort_keys=id_keys, group_by=['SYMBOL'], 
                                        method='ffill')
            
            # 避免開頭的資料是NA，所以再用一次bfill
            main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                        sort_keys=id_keys, group_by=['SYMBOL'], 
                                        method='bfill')    

            main_data = cbyz.df_conv_na(df=main_data, cols=cols, value=0)                   


    # ^TWII .......
    if market == 'tw':
         
        tw_index, cols = \
            sam_od_tw_get_index(
                begin_date=shift_begin,
                end_date=predict_date.loc[len(predict_date)-1, 'WORK_DATE']
                )
        
        main_data = main_data.merge(tw_index, how='left', on=time_key)
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=[])    

    
    # Fiat Currency Exchange ......
    fx_rate, cols = sam_od_tw_get_fx_rate()
    main_data = main_data.merge(fx_rate, how='left', on=time_key)

    main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                      sort_keys=time_key, 
                                      method=['ffill', 'bfill'], 
                                      group_by=[])


    # Get Dow Jones Industrial Average (^DJI) ......
    dji, cols = sam_od_us_get_snp(begin_date=shift_begin)
    main_data = main_data.merge(dji, how='left', on=time_key)
    del dji
    gc.collect()   


    # S&P 500 ......
    snp, cols = sam_od_us_get_snp(begin_date=shift_begin)
    main_data = main_data.merge(snp, how='left', on=time_key)
    del snp
    gc.collect()
    
    
    # COVID-19 ......
    if market == 'tw':
        covid_tw, cols = sam_covid_19_tw()
        
        # Future Plan
        # sam_covid_19_global()
        
        main_data = main_data.merge(covid_tw, how='left', on=time_key)
        main_data = cbyz.df_conv_na(df=main_data, cols=cols)
        
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=[])
    elif market == 'en':
        # Future Plan
        # covid_en = sam_covid_19_global()        
        pass


    
    # Monthly Revenue ......
    # 1. 當predict_date=20211101，且為dev時, 造成每一個symbol都有na，先移除
    # 2. 主要邏輯就是顯示最新的營收資料
    # if market == 'tw':
        
    #     msg = '''Bug - sam_tej_get_ewsale，在1/18 23:00跑1/19時會出現chk_na error，但1/19 00:00過後
    #     再跑就正常了
    #     '''
    #     print(msg)
        
    #     ewsale = sam_tej_get_ewsale(begin_date=shift_begin)
    #     main_data = main_data \
    #                 .merge(ewsale, how='left', on=['SYMBOL', 'WORK_DATE'])      
    
    #     cbyz.df_chk_col_na(df=main_data, except_cols=var_y, mode='stop')
    
    
    # Financial Statement
    # if market == 'tw':
        # print('目前只用單季，需確認是否有缺漏')
        # 20220218 - Dev=True，eta=0.2時，即使只保留一個欄位也會overfitting
        # financial_statement, cols = sam_tej_get_ewifinq()
        
        # main_data = main_data \
        #         .merge(financial_statement, how='left', on=id_keys)
                
        # main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
        #                                   sort_keys=time_key, 
        #                                   method=['ffill', 'bfill'], 
        #                                   group_by=[])


    # TEJ ewtinst1c - Average Holding Cost of Juridical Persons ......
    # if market == 'tw':
    #     ewtinst1c, cols = sam_tej_get_ewtinst1c()
    #     main_data = main_data.merge(ewtinst1c, how='left', on=id_keys)
    #     main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
    #                                       sort_keys=time_key, 
    #                                       method=['ffill', 'bfill'], 
    #                                       group_by=['SYMBOL'])          


    # Model Data ......


    # Remove date before first_trading_day
    # - 由於main_data_frame是用cross_join，所以會出現listing前的日期，但這個步驟要等到
    #   最後才執行，否則在合併某些以月或季為單位的資料時會出現NA
    print('是否可以移到sam_load_data最下面；暫時移除')
    # global first_trading_day
    # main_data = main_data \
    #     .merge(first_trading_day, how='left', on=['SYMBOL'])
    
    # main_data = main_data[
    #     main_data['WORK_DATE']>=main_data['FIRST_TRADING_DAY']] \
    #     .drop('FIRST_TRADING_DAY', axis=1)


    # Check NA ......
    if time_unit == 'd':
        hist_df = main_data[
            main_data['WORK_DATE']<predict_date['WORK_DATE'].min()]
        
    elif time_unit == 'w':
        hist_df = cbyz.df_anti_merge(main_data, predict_week, 
                                     on=time_key)
    
    
    # Debug ......
    msg = ("get_model_data - 把normalize的group_by拿掉後，這個地方會出錯，但"
           "只有一筆資料有問題，暫時直接drop")
    print(msg)
    
    
    # - 當symbols=[]時，這裡會有18筆NA，都是var_y的欄位，應該是新股，因此直接排除
    # - 當time_unit為w，predict_begin為20220104時，會有275筆NA，但都是5244這一
    #   檔，且都是ewtinst1c的欄位，應該也是新股的問題，直接排除
    # - 新股的ewtinst1c一定會有NA，但SNP不會是NA，導致na_min和na_max不一定相等
    # - 如果只是某幾天缺資料的話，chk_na_min和chk_na_max應該不會相等
    chk_na = cbyz.df_chk_col_na(df=hist_df)
    chk_na = chk_na[~chk_na['COLUMN'].isin(var_y)]
    
    assert_cond = len(chk_na) < 600
    if not assert_cond:
        chk_na.to_csv(path_temp + '/chk_na_id_01.csv', index=False)

    assert assert_cond, \
        'get_model_data - hist_df has ' + str(len(chk_na)) + ' NA'
    
    na_cols = chk_na['COLUMN'].tolist()
    main_data = main_data.dropna(subset=na_cols, axis=0)
    
    
    # Predict有NA是正常的，但NA_COUNT必須全部都一樣
    global chk_predict_na
    if time_unit == 'd':
        predict_df = main_data.merge(predict_date, on=time_key)
        
    elif time_unit == 'w':
        predict_df = main_data.merge(predict_week, on=time_key)
    
    chk_predict_na = cbyz.df_chk_col_na(df=predict_df, mode='alert')
    min_value = chk_predict_na['NA_COUNT'].min()
    max_value = chk_predict_na['NA_COUNT'].max()    
    
    assert len(chk_predict_na) == 0 or min_value == max_value, \
        'All the NA_COUNT should be the same.'
    

    # Check min max ......
    # global chk_min_max
    # chk_min_max = cbyz.df_chk_col_min_max(df=main_data)
    
    # chk_min_max = \
    #     chk_min_max[(~chk_min_max['COLUMN'].isin(id_keys)) \
    #                 & ((chk_min_max['MIN_VALUE']<0) \
    #                    | (chk_min_max['MAX_VALUE']>1))]
    
    # assert len(chk_min_max) == 0, 'get_model_data - normalize error'


    # Select Features ......
    # - Select featuers after completing data cleaning
    
    # Drop Highly Correlated Features
    main_data = cbml.df_drop_high_corr_var(df=main_data, 
                                            threshold=corr_threshold, 
                                            except_cols=id_keys + var_y) 
    # Select Best Features
    best_var_raw, best_var_score, best_var = \
        cbml.selectkbest(df=main_data, model_type='reg', 
                          y=var_y, X=[], except_cols=id_keys, k=60)
        
    main_data = main_data[id_keys + var_y + best_var] 


    # Variables ......
    model_x = cbyz.df_get_cols_except(df=main_data, 
                                      except_cols=var_y + id_keys)


    # Export Model ......
    main_data.to_csv(main_data_file, index=False)
    cbyz.li_to_csv(model_x, model_x_file)
    scale_orig.to_csv(scale_orig_file, index=False)
        
    return main_data, model_x, scale_orig
    


# %% Master ------


def master(param_holder, predict_begin, export_model=True,
           threshold=30000, bt_index=0, load_data=False):
    '''
    主工作區
    '''
    
    # v2.10
    # - Rename symbols as symbol
    # - Add symbol params to ewifinq
    # - Update cbml for df_scaler
    # v2.11 - 20220119
    # - The MVP version of data_for = 2
    # - Y can be price or change ratio
    # v2.112 - 20220123
    # - MVP of weekly prediction
    # v2.2 - 20220209
    # - Restore variables for weekly prediction
    # - Change the way to shift day. The original method is shift var_x, and 
    #   the new version is to shift var_y and id_keys
    # - Remove ml_data_process
    # - Remove switch of trade_value
    # v2.3 - 20220210
    # - Combine result of daily prediction and weekly prediction in BTM
    # - Add time_unit as suffix for saved_file of model_data
    # - industry_one_hot 不用df_summary    
    # - Modify dev mode and test mode
    # v2.400 - 20220214
    # - Collect fx_rate in dcm
    # - Add fx_rate to pipeline
    # - Remove open_change_ratio and open from var_y
    # v2.500 - 20220215
    # v2.501 - 20220216
    # - Replace YEAR with YEAR_ISO, and WEEK with WEEK_ISO
    # v2.502 - 20220216
    # - Restore sam_tej_get_ewifinq, but removed again
    # - Set model params for week and day seperately
    # - Move od_tw_get_ex_dividends to arsenal_stock
    # - Optimize load_data feature of get_model_data 
    # v2.0600 - 20220221
    # - 當time_unit為w時，讓predict_begin可以不是星期一
    # - week_align為True時，get_model_data最下面的assert會出錯
    # v3.0000 - 20220225
    # - 開發重心轉移至trading bot
    # - Update for ultra_tuner v0.3100
    # v3.0100 - 20220305
    # - Drop correlated columns in variable function, or it will cause 
    #   expensive to execute this in the ultra_tuner
    # - Give the common serial when predict for each y
    # - Add MLP
    # v3.0200
    # - Update for cbml.selectkbest
    # - Update for ut_v1.0001, and add epochs to model_params
    # v3.0300
    # - Add tej_get_ewgin
    # - Add tej_get_ewtinst1
    # - Add od_us_get_dji
    # - Completed test
    
    
    # v3.0400
    # df_expend_one_hot_signal


    # v3.0500
    # - Add df_vif and fix bug
    # - C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\stats
    #   \outliers_influence.py:193: RuntimeWarning: divide by zero encountered in
    #   double_scalars
    #   vif = 1. / (1. - r_squared_i)


    global version
    version = 3.0400    
    
    
    # Bug
    # SNP有NA是合理的嗎？
    #                      COLUMN  NA_COUNT
    # 0         HIGH_CHANGE_RATIO       475
    # 1          LOW_CHANGE_RATIO       475
    # 2        CLOSE_CHANGE_RATIO       475
    # 3       QFII_HAP_MA_36_MEAN       256
    # 4        QFII_HAP_MA_36_MIN       256
    # ..                      ...       ...
    # 348    SNP_VOLUME_MA_1_MEAN       446
    # 349     SNP_VOLUME_MA_1_MIN       446
    # 350     SNP_VOLUME_MA_1_MAX       446
    # 351  SNP_VOLUME_MA_1_MEDIAN       446
    # 352     SNP_VOLUME_MA_1_STD       446
    

    
    # Update
    # - Bug - sam_tej_get_ewsale，在1/18 23:00跑1/19時會出現chk_na error，
    #   但1/19 00:00過後再跑就正常。end_date應該要改成data_begin, 這個問題應該
    #   是today比data_begin少一天    
    # - Replace symbol with target, and add target_type which may be symbol
    #   or industry or 大盤
    # - Add financial_statement
    #   > 2021下半年還沒更新，需要改code，可以自動化更新並合併csv
    # - 財報的更新時間很慢，20220216的時候，2021年12月的財報還沒出來；所以
    #   在tej_get_ewtinst1c需要有提醒，列出最新的財報日期，並用assert檢查中間
    #   是否有空缺
    # - Fix Support and resistant
    # - select_symbols用過去一周的總成交量檢查
    # 以close price normalize，評估高價股和低價股
    # - Handle week_num cyclical issues
    #   https://towardsdatascience.com/how-to-handle-cyclical-data-in-machine-learning-3e0336f7f97c
    
    
    # New Dataset ......
    # - 美元利率
    # - 台幣利率
    # https://www.cbc.gov.tw/tw/np-1166-1.html
    # - 國際油價
    # - 黃金價格
    # - 確認現金流入率 / 流出率的算法，是否要買賣金額，還是只要有交易額就可以了
    # - 三大法人交易明細
    #   >> 如果買TEJ的達人方案，那也不需要額外加購三大法人持股成本
    #   >> 增加三大法人交易明細後，試著計算三大法人每個交易所的平均持有天數，
    #      再試著將平均持有天數做分群，就可以算出每一個交易所的交易風格和產業策略。
    #   >> 增加三大法人交易明細後，從20170101，累積計算後，可以知道每一個法人
    #      手上的持股狀況；
    # - Add 流通股數 from ewprcd
    # - Add 法說會日期
    # - Add 道瓊指數
    

    # Optimization ......
    # - 如何把法說會的日期往前推N天，應該interpolate
    # - Combine cbyz >> detect_cycle.py, including support_resistance and 
    #    season_decompose
    # - Fix week_num, because it may be useful to present seasonaility
    # - 合併Yahoo Finance和TEJ的market data，兩邊都有可能缺資料。現在的方法是
    #   用interpolate，
    #   但如果begin_date剛好缺值，這檔股票就會被排除
    # - 把symbol改成target，且多一個target_type，值可以是symbol或industry
    # - 技術分析型態學
    # - buy_signal to int
    #   Update - 20211220，暫時在get_tw_index中呼叫update_tw_index，但這個方式
    #   可能會漏資料    
    
    
    global bt_last_begin, data_period, predict_period, long, time_unit
    global dev, test
    global symbol, ma_values, volume_thld, market, data_form, load_model_data


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
    data_form = holder['data_form'][0]   
    time_unit = holder['time_unit'][0]
    load_model_data = holder['load_model_data'][0]
    
    # Modeling
    predict_period = holder['predict_period'][0]
    long = holder['long'][0]   
    kbest = holder['kbest'][0]
    cv = holder['cv'][0]
    
    # Program
    dev = holder['dev'][0]
    test = holder['test'][0]
    
    # export_model = False
    # dev = True
    # threshold = 20000
    # predict_begin=20211209
    
    
    global exe_serial
    exe_serial = cbyz.get_time_serial(with_time=True, remove_year_head=True)

    global log, error_msg
    global ohlc, ohlc_ratio, ohlc_change
    log = []
    error_msg = []    
    ohlc = stk.get_ohlc()
    ohlc_ratio = stk.get_ohlc(orig=False, ratio=True, change=False)
    ohlc_change = stk.get_ohlc(orig=False, ratio=False, change=True)
    
    # Keys ------
    global id_keys, time_key
    global var_y, var_y_orig
    
    if time_unit == 'w':
        id_keys = ['SYMBOL', 'YEAR_ISO', 'WEEK_NUM_ISO']
        time_key = ['YEAR_ISO', 'WEEK_NUM_ISO']
        
    elif time_unit == 'd':
        id_keys = ['SYMBOL', 'WORK_DATE']    
        time_key = ['WORK_DATE']

    
    # var_y = ['HIGH', 'LOW', 'CLOSE']
    var_y = ['HIGH_CHANGE_RATIO', 'LOW_CHANGE_RATIO', 'CLOSE_CHANGE_RATIO']
    var_y_orig = [y + '_ORIG' for y in var_y]    
    
    
    # Update, add to BTM
    global wma, corr_threshold
    wma = False

    # 原本設定為0.85，但CU DTSA 5509將Collinearity的標準設為0.7
    corr_threshold = 0.7
    # corr_threshold = 0.85
    
    
    global df_summary_mean, df_summary_min, df_summary_max
    global df_summary_median, df_summary_std
    
    
    print('有些dataset需要sum，如交易量')
    df_summary_mean = True
    df_summary_min = True
    df_summary_max = True
    df_summary_median = True
    df_summary_std = True
    
    
    # Calendar ------
    global shift_begin, shift_end, data_begin, data_end
    global predict_date, predict_week
    global calendar, calendar_full_key
    
    shift_begin, shift_end, data_begin, data_end, \
        predict_date, predict_week, calendar = \
                stk.get_period(predict_begin=predict_begin,
                               predict_period=predict_period,
                               data_period=data_period,
                               unit=time_unit, week_align=True,
                               shift=int((max(ma_values) + 20)))

    # df may have na if week_align is true
    if time_unit == 'w':
        calendar = calendar.dropna(subset=time_key, axis=0)
        
    # 有些資料可能包含非交易日，像是COVID-19，所以需要一個額外的calendar作比對
    calendar_full_key = calendar[['WORK_DATE', 'YEAR_ISO', 'WEEK_NUM_ISO']]
                
                
    # ......
    global model_data, model_x, scale_orig
    model_data, model_x, scale_orig = \
        get_model_data(industry=industry, trade_value=trade_value,
                       load_file=load_model_data)
    
    
    # Training Model ......
    import xgboost as xgb
    from sklearn.linear_model import LinearRegression
    import tensorflow as tf

    
    if len(symbol) > 0 and len(symbol) < 10:
        model_params = [{'model': LinearRegression(),
                         'params': {
                             'normalize': [True, False],
                             }
                         }]         
    else:
        
        # MLP
        # - Prevent error on host 4
        if host in [2, 3]:
            vars_len = cbyz.df_get_cols_except(df=model_data, 
                                               except_cols=id_keys + var_y)
            vars_len = len(vars_len)
            
            mlp_model = tf.keras.Sequential()
            mlp_model.add(tf.keras.layers.Dense(30, input_dim=vars_len, 
                                                activation='relu'))
            
            mlp_model.add(tf.keras.layers.Dense(30, activation='softmax'))
            mlp_model.add(tf.keras.layers.Dense(1, activation='linear'))  
            
            mlp_param = {'model': mlp_model,
                         'params': {'optimizer':'adam',
                                    'epochs':15}}
        
        
        # eta 0.01、0.03的效果都很差，目前測試0.08和0.1的效果較佳
        # data_form1 - Change Ratio
        if time_unit == 'd' and 'CLOSE_CHANGE_RATIO' in var_y:
            
            model_params = [{'model': LinearRegression(),
                              'params': {
                                  'normalize': [True, False],
                                  }
                              },
                            {'model': xgb.XGBRegressor(),
                              'params': {
                                # 'n_estimators': [200],
                                'eta': [0.1],
                                # 'eta': [0.08, 0.1],
                                'min_child_weight': [1],
                                  # 'min_child_weight': [0.5, 1],
                                'max_depth':[8],
                                  # 'max_depth':[6, 8, 12],
                                'subsample':[1]
                              }
                            }
                            ]
            
        elif time_unit == 'w' and 'CLOSE_CHANGE_RATIO' in var_y:
            
            # eta 0.1 / 0.2
            model_params = [
                            {'model': LinearRegression(),
                              'params': {
                                  'normalize': [True, False],
                                  }
                              },
                            {'model': xgb.XGBRegressor(),
                              'params': {
                                # 'n_estimators': [200],
                                # 'eta': [0.1],
                                'eta': [0.1, 0.2],
                                'min_child_weight': [1],
                                  # 'min_child_weight': [0.5, 1],
                                'max_depth':[8],
                                  # 'max_depth':[6, 8, 12],
                                'subsample':[1]
                              }
                            }
                            ]             
            
        # Prevent error on host 4
        if host in [2, 3]:
            model_params.insert(0, mlp_param)
            
        
        # # Price
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
        #                     # 'eta': [0.5, 0.7],
        #                     'min_child_weight': [0.8],
        #                      # 'min_child_weight': [0.5, 1],
        #                     'max_depth':[10],
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
        

        # data_form2 - Price
        # model_params = [
        #                 {'model': LinearRegression(),
        #                   'params': {
        #                       'normalize': [True, False],
        #                       }
        #                   },
        #                 {'model': xgb.XGBRegressor(),
        #                   'params': {
        #                     # 'n_estimators': [200],
        #                     'eta': [0.2, 0.4],
        #                     # 'eta': [0.08, 0.1],
        #                     # 'min_child_weight': [1],
        #                   'min_child_weight': [0.5, 1],
        #                     'max_depth':[6, 8],
        #                       # 'max_depth':[6, 8, 12],
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
        #                 ]         
        
        
    # - 如果selectkbest的k設得太小時，importance最高的可能都是industry，導致同
    #   產業的預測值完全相同
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
                               serial=exe_serial,
                               path=path_temp)
        
        # 排除其他y，否則會出錯
        cur_model_data = model_data.drop(remove_y, axis=1)
        
        return_result, return_scores, return_params, return_features, \
                log_scores, log_params, log_features = \
                    tuner.fit(data=cur_model_data, 
                              model_params=model_params,
                              cv=cv, threshold=threshold, 
                              scale_orig=[],
                              export_model=True, export_log=True)
                 
        if i == 0:
            pred_result = return_result.copy()
            pred_scores = return_scores.copy()
            pred_params = return_params.copy()
            pred_features = return_features.copy()
        else:
            pred_result = pred_result \
                        .merge(return_result, how='left', on=id_keys)
            pred_scores = pred_scores.append(return_scores)
            pred_params = pred_params.append(return_params)
            pred_features = pred_features.append(return_features)            

        # Prvent memory insufficient for saved data in ut
        del tuner
        gc.collect()


    # Inverse Scale
    global y_scaler        
    pred_result = pred_result[id_keys + var_y]
    y_inverse = pred_result[var_y]
    y_inverse = y_scaler.inverse_transform(y_inverse)
    y_inverse = pd.DataFrame(y_inverse, columns=var_y)
    
    pred_result_inverse = pd.concat([pred_result[id_keys], y_inverse],
                                    axis=1)
        
    # Upload to Google Sheet
    if predict_begin == bt_last_begin:
        stk.write_sheet(data=pred_features, sheet='Features')

    
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

    


def get_season(df):
    
    '''
    By Week, 直接在get_data中處理好
    
    '''
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    
    loc_df = df.copy() \
            .rename(columns={'STOCK_SYMBOL':'SYMBOL'}) \
            .sort_values(by=['SYMBOL', 'YEAR_ISO', 'WEEK_NUM_ISO']) \
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



def add_support_resistance(df, cols, rank_thld=10, prominence=4, 
                           unit='d', interval=True,
                           threshold=0.9, plot_data=False):
    '''
    1. Calculate suppport and resistance
    2. The prominence of each symbol is different, so it will cause problems
       if apply a static number. So use quantile as the divider.
    3. Update, add multiprocessing
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.
    rank_thld : TYPE, optional
        DESCRIPTION. The default is 10.
    prominence : TYPE, optional
        DESCRIPTION. The default is 4.
    interval : boolean, optional
        Add interval of peaks.
    threshold : TYPE, optional
        DESCRIPTION. The default is 0.9.
    plot_data : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    result : TYPE
        DESCRIPTION.
    return_cols : TYPE
        DESCRIPTION.

    '''
    
    print('Bug, 回傳必要的欄位，不要直接合併整個dataframe，減少記憶體用量')
    

    from scipy.signal import find_peaks
    cols = cbyz.conv_to_list(cols)

    cols_support = [c + '_SUPPORT' for c in cols]
    cols_resistance = [c + '_RESISTANCE' for c in cols]
    return_cols = cols_support + cols_resistance
    
    
    group_key = ['SYMBOL', 'COLUMN', 'TYPE']
    
    
    # .......
    loc_df = df[['SYMBOL', 'WORK_DATE'] + cols]
    
    date_index = loc_df[['SYMBOL', 'WORK_DATE']]
    date_index = cbyz.df_add_rank(df=date_index, value='WORK_DATE',
                              group_by=['SYMBOL'], 
                              sort_ascending=True, 
                              rank_ascending=True,
                              rank_name='index',
                              rank_method='min', inplace=False)
    
    result_raw = pd.DataFrame()
    
    symbol_df = loc_df[['SYMBOL']].drop_duplicates()
    symbol = symbol_df['SYMBOL'].tolist()


    # Frame ......
    begin_date = loc_df['WORK_DATE'].min()
    today = cbyz.date_get_today()

    calendar = cbyz.date_get_calendar(begin_date=begin_date,
                                      end_date=today)
    
    calendar = calendar[['WORK_DATE']]
    cols_df = pd.DataFrame({'COLUMN':cols})
    
    frame = cbyz.df_cross_join(symbol_df, calendar)
    frame = cbyz.df_cross_join(frame, cols_df)

    
    # Calculate ......
    for j in range(len(symbol)):
        
        symbol_cur = symbol[j]
        temp_df = loc_df[loc_df['SYMBOL']==symbol_cur].reset_index(drop=True)
    
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
            
            new_top['SYMBOL'] = symbol_cur
            new_top['COLUMN'] = col
            new_top['TYPE'] = 'RESISTANCE'
            
            
            # 計算低點
            # - 使用-x反轉，讓低點變成高點，所以quantile一樣threshold
            peaks_btm, prop_btm = find_peaks(-x, prominence=prominence)   
            new_btm = pd.DataFrame({'VALUE':[i for i in prop_btm['prominences']]})
            new_btm.index = peaks_btm
            
            threshold_value = new_btm['VALUE'].quantile(threshold)
            new_btm = new_btm[new_btm['VALUE']>=threshold_value]            
            
            new_btm['SYMBOL'] = symbol_cur
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
            
    result = result[['SYMBOL', 'WORK_DATE', 'COLUMN', 'TYPE']] \
            .sort_values(by=['SYMBOL', 'COLUMN', 'WORK_DATE']) \
            .reset_index(drop=True)

    # Dummy Encoding
    result = cbml.df_get_dummies(df=result, cols='TYPE', 
                                 expand_col_name=True,inplace=False)


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
    if interval:
        
        interval_df, _ = \
            cbyz.df_add_shift(df=result, cols='WORK_DATE', shift=1, 
                              sort_by=['SYMBOL', 'COLUMN', 'WORK_DATE'],
                              group_by=['SYMBOL', 'COLUMN'], 
                              suffix='_PREV', remove_na=False)
            
        # NA will cause error when calculate date difference
        interval_df = interval_df.dropna(subset=['WORK_DATE_PREV'], axis=0)
        
        interval_df = cbyz.df_conv_col_type(df=interval_df,
                                            cols=['WORK_DATE', 'WORK_DATE_PREV'],
                                            to='int')
        
        interval_df = cbyz.df_date_diff(df=interval_df, col1='WORK_DATE', 
                                    col2='WORK_DATE_PREV', name='INTERVAL', 
                                    absolute=True, inplace=False)
        
        interval_df = cbyz.df_conv_col_type(df=interval_df,
                                            cols=['INTERVAL'],
                                            to='int')
        
        interval_df = interval_df.drop('WORK_DATE_PREV', axis=1)
        
        print(('Does it make sense to use median? Or quantile is better, '
              'then I can go ahead of the market'))
        interval_df = interval_df \
                    .groupby(['SYMBOL', 'COLUMN']) \
                    .agg({'INTERVAL':'median'}) \
                    .reset_index()
            
        # Merge
        result_full = frame \
            .merge(result, how='left', on=['SYMBOL', 'WORK_DATE', 'COLUMN'])
        
        result_full = cbyz.df_conv_na(df=result_full, 
                                      cols=['TYPE_RESISTANCE', 'TYPE_SUPPORT'],
                                      value=0)
        
        result_full = result_full \
            .merge(interval_df, how='left', on=['SYMBOL', 'COLUMN'])
        
        
        result_full['RESIS_SUPPORT_SIGNAL'] = \
            np.select([result_full['TYPE_RESISTANCE']==1,
                       result_full['TYPE_SUPPORT']==1],
                      [result_full['INTERVAL'], -result_full['INTERVAL']],
                      default=np.nan)
         
        result_full['INDEX'] = \
            np.where((result_full['TYPE_RESISTANCE']==1) \
                      | (result_full['TYPE_SUPPORT']==1),
                      result_full.index, np.nan)
            
        result_full = \
            cbyz.df_fillna(df=result_full, 
                           cols=['RESIS_SUPPORT_SIGNAL', 'INDEX'], 
                           sort_keys=['SYMBOL', 'COLUMN', 'WORK_DATE'], 
                           group_by=['SYMBOL', 'COLUMN'],
                           method='ffill')
            
        result_full['RESIS_SUPPORT_SIGNAL'] = \
            np.where((result_full['TYPE_RESISTANCE']==1) \
                     | (result_full['TYPE_SUPPORT']==1),
                     result_full['RESIS_SUPPORT_SIGNAL'],
                     result_full['RESIS_SUPPORT_SIGNAL'] \
                         - (result_full.index - result_full['INDEX']))
            
    else:
        pass
            

    # if unit == 'w':
    #     interval_df['INTERVAL'] = interval_df['INTERVAL'] / 7

        
        
    return result, return_cols




def test_support_resistance():
    

    data = stk.od_tw_get_index()
    
    # data = data.rename(columns={'TW_INDEX_CLOSE':'CLOSE'})
    data['SYMBOL'] = 1001
    
    
    add_support_resistance(df=data, cols='TW_INDEX_CLOSE', 
                           rank_thld=10, prominence=4, 
                           days=True, threshold=0.9, plot_data=False)    



# %% Debug ------

def debug():
    
    pass
    


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
            'symbol':[symbol]
            }
    
    param_holder = ar.Param_Holder(**args)
        
    master(param_holder=param_holder,
           predict_begin=20211201, threshold=30000)        


