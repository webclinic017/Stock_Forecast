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



# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import pickle

host = 3
host = 2
host = 4
host = 0


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
                 r'/Users/aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20220213',
                 r'/home/jupyter/Arsenal/20220213',
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
    
    
    market_data, scale_orig_ratio, y_scaler_ratio = \
        cbml.df_scaler(df=market_data, cols=ohlc_ratio,
                       show_progress=False, method=1)
    
    market_data, scale_orig_price, y_scaler_price = \
        cbml.df_scaler(df=market_data, cols=ohlc,
                       show_progress=False, method=1)

    scale_orig = scale_orig_ratio.append(scale_orig_price)
    
    if 'CLOSE_CHANGE_RATIO' in var_y:
        y_scaler = y_scaler_ratio
    elif 'CLOSE' in var_y:
        y_scaler = y_scaler_price
        
    pickle.dump(y_scaler, open(path_temp + '/y_scaler.sav', 'wb'))
    
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
        
        except_cols = ['WORK_DATE', 'YEAR', 'MONTH',
                       'WEEKDAY', 'WEEK_NUM'] + id_keys  

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
    
        
        
    # Drop Except會導致CLOSE_LAG, HIGH_LAG沒被排除
    # - 20220212 - 因為把ml_data_process移除，所以這段應該也不需要了
    # if 'CLOSE' in var_y and data_form == 1:
    #     ohlc_str = '|'.join(ohlc)
    #     drop_cols = cbyz.df_chk_col_na(df=loc_main, positive_only=True)
        
    #     drop_cols = drop_cols[(~drop_cols['COLUMN'].isin(var_y)) \
    #                           & (~drop_cols['COLUMN'].str.contains('MA')) \
    #                           & (drop_cols['COLUMN'].str.contains(ohlc_str))]
        
    #     drop_cols = drop_cols['COLUMN'].tolist()
    #     loc_main = loc_main.drop(drop_cols, axis=1)
    

    
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
        
        
        # 這裡merge完後，shift_period的YEAR和WEEK_NUM中會有NA，且shift_period
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
            cbyz.df_summary(df=loc_main, cols=cols, group_by=id_keys, 
                            add_mean=True, add_min=True, 
                            add_max=True, add_median=True, add_std=True, 
                            add_skew=False, add_count=False, quantile=[])
            
        loc_main = loc_main.merge(y_data, how='left', on=id_keys)
        del y_data
        gc.collect()
        
    elif time_unit == 'd':
        
        # 這裡merge完後，shift_period的YEAR和WEEK_NUM中會有NA，且shift_period
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


    # 檢查shift完後YEAR和WEEK_NUM的NA
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
    # - YEAR and WEEK_NUM will be float here
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
        calendar_key = calendar_key[['WORK_DATE', 'YEAR', 'WEEK_NUM']]
    
    
    # Duplicate year and week_num, then these two columns can be variables 
    # of the model
    if time_unit == 'w':
        
        calendar_proc = calendar_proc[['YEAR', 'MONTH', 
                                       'WEEK_NUM', 'TRADE_DATE']]
        
        calendar_proc = calendar_proc \
            .drop_duplicates() \
            .reset_index(drop=True) \
            .reset_index() \
            .rename(columns={'index':'DATE_INDEX'}) 
        
        calendar_proc.loc[:, 'YEAR_DUP'] = calendar_proc['YEAR']
        calendar_proc.loc[:, 'WEEK_NUM_DUP'] = calendar_proc['WEEK_NUM']
        

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
    main_data_frame = ar.df_simplify_dtypes(df=main_data_frame)
    
    print('Check - main_data_frame_calendar是否等同calendar_lite')
    main_data_frame_calendar = calendar_lite.copy()
    # main_data_frame_calendar = main_data_frame[time_key] \
    #                             .drop_duplicates() \
    #                             .sort_values(by='WORK_DATE') \
    #                             .reset_index(drop=True)



# %% TW Variables ------


def sam_buffett_indicator():
    
    global time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma    
    
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

    # Merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key)    

    return result, ma_cols


# .........

def sam_covid_19_tw():
    
    global time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma
    
    result, _ = cbyz.get_covid19_data()
    
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['WORK_DATE', 'YEAR', 'WEEK_NUM']
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
        
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=time_key, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=False, add_count=False, quantile=[]
            )
    
    # Merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key)    

    return result, ma_cols


# .................
    

def sam_tw_gov_invest(dev=False):
    
    global time_key, symbol_df, calendar_key
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
    
    return result
    

# .................
    

def sam_tw_gov_own(dev=False):
    
    global time_key, symbol_df, calendar_key
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
    
    return result
    


# .................


def sam_od_tw_get_index(begin_date, end_date):
    
    global ma_values, predict_period
    global time_unit, time_key
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
    
    # Merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key)   

    return result, ma_cols


# .................


def sam_od_us_get_snp_data(begin_date):
    
    global ma_values, predict_period, predict_date
    global calendar, main_data_frame_calendar
    
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
    
    # loc_calendar = cbyz.date_get_calendar(begin_date=begin_date, 
    #                                       end_date=predict_date[-1])
    loc_calendar = calendar[['WORK_DATE']]
    
    loc_df = loc_calendar.merge(loc_df, how='left', on='WORK_DATE')
    cols = cbyz.df_get_cols_except(df=loc_df, except_cols='WORK_DATE')
    loc_df = cbyz.df_fillna(df=loc_df, cols=cols, sort_keys='WORK_DATE', 
                            group_by=[], method='ffill')

    # Process
    # loc_df, cols, _, _ = \
    #     cbml.ml_data_process(df=loc_df, 
    #                          ma=True, scale=True, lag=True, 
    #                          group_by=[],
    #                          cols=[], 
    #                          except_cols=['WORK_DATE'],
    #                          drop_except=[],
    #                          cols_mode='equal',
    #                          date_col='WORK_DATE',
    #                          ma_values=ma_values, 
    #                          lag_period=predict_period
    #                          )    
    
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



# %% Process ------


def get_model_data(industry=True, trade_value=True, load_file=False):
    
    
    global shift_begin, shift_end, data_begin, data_end, ma_values
    global predict_date, predict_week, predict_period
    global calendar, calendar_lite, calendar_proc
    global calendar_key, calendar_full_key
    global main_data_frame, main_data_frame_calendar, market_data
    global symbol
    global var_y
    global params, error_msg
    global id_keys
    global main_data
    global time_unit
    
    
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


        main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                   sort_keys=time_key, group_by=[], 
                                   method='ffill')
    
        main_data = cbyz.df_fillna(df=main_data, cols=cols,
                                   sort_keys=time_key, group_by=[], 
                                   method='bfill')
                

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


    # # 除權息資料 ......
    # # Close Lag ...
    # daily_close = market_data[['WORK_DATE', 'SYMBOL', 'CLOSE']]
    # daily_close, _ = cbyz.df_add_shift(df=daily_close, 
    #                                 cols='CLOSE', shift=1,
    #                                 group_by=['SYMBOL'],
    #                                 suffix='_LAG', 
    #                                 remove_na=False)
    # daily_close = daily_close \
    #             .drop('CLOSE', axis=1) \
    #             .rename(columns={'CLOSE_LAG':'CLOSE'})
      
    # daily_close = cbyz.df_fillna(df=daily_close, cols='CLOSE', 
    #                              sort_keys=['SYMBOL', 'WORK_DATE'],
    #                              method='both')
                
    
    # # 除權息 ...
    # sale_mon_data1, sale_mon_data2 = get_sale_mon_data()
    
    # # Data 1 - 除權息日期及價錢 ...
    # sale_mon_data1 = daily_close \
    #     .merge(sale_mon_data1, how='left', on=['WORK_DATE', 'SYMBOL'])
    
    # sale_mon_data1['EX_DIVIDENDS_PRICE'] = \
    #     sale_mon_data1['EX_DIVIDENDS_PRICE'] / sale_mon_data1['CLOSE']    
        
    # sale_mon_data1 = sale_mon_data1.drop('CLOSE', axis=1)
    # sale_mon_data1 = cbyz.df_conv_na(df=sale_mon_data1, 
    #                                  cols=['EX_DIVIDENDS_PRICE', 
    #                                        'SALE_MON_DATE'])
    
    # sale_mon_data1, _, _, _ = \
    #          cbml.ml_data_process(df=sale_mon_data1, 
    #                               ma=False, scale=True, lag=False, 
    #                               group_by=['SYMBOL'], 
    #                               cols=['EX_DIVIDENDS_PRICE', 'SALE_MON_DATE'],
    #                               except_cols=[],
    #                               drop_except=[],
    #                               cols_mode='equal',
    #                               date_col='WORK_DATE',
    #                               ma_values=ma_values, 
    #                               lag_period=predict_period
    #                               ) 
    
    # # Data 2 - 填息 ...
    # sale_mon_data2, _, _, _ = \
    #     cbml.ml_data_process(df=sale_mon_data2, 
    #                          ma=False, scale=True, lag=False, 
    #                          group_by=['SYMBOL'],
    #                          cols=[], 
    #                          except_cols=['EX_DIVIDENDS_DONE'],
    #                          drop_except=[],
    #                          cols_mode='equal',
    #                          date_col='WORK_DATE',
    #                          ma_values=ma_values, 
    #                          lag_period=predict_period
    #                          )
        
    # main_data = main_data \
    #     .merge(sale_mon_data1, how='left', on=['WORK_DATE', 'SYMBOL']) \
    #     .merge(sale_mon_data2, how='left', on=['WORK_DATE', 'SYMBOL'])
    
    # # Convert NA
    # temp_cols = ['EX_DIVIDENDS_PRICE', 'SALE_MON_DATE', 'EX_DIVIDENDS_DONE']    
    # main_data = cbyz.df_conv_na(df=main_data, cols=temp_cols)


    
    # # TEJ 三大法人持股成本 ......
    # if market == 'tw':
        
    #     ewtinst1c_raw = stk.tej_get_ewtinst1c(begin_date=shift_begin,
    #                                           end_date=None, 
    #                                           symbol=symbol,
    #                                           trade=True)
        
    #     ewtinst1c = main_data_frame \
    #                 .merge(ewtinst1c_raw, how='left', on=['WORK_DATE', 'SYMBOL']) \
    #                 .merge(symbol_df, on=['SYMBOL'])
    
    #     cols = cbyz.df_get_cols_except(df=ewtinst1c,
    #                                     except_cols=['WORK_DATE', 'SYMBOL']) 
        
    #     ewtinst1c = cbyz.df_fillna(df=ewtinst1c, cols=cols, 
    #                                 sort_keys=['SYMBOL', 'WORK_DATE'], 
    #                                 group_by=[], method='ffill')    

    #     # 獲利率HROI、Sell、Buy用globally normalize，所以要分兩段
    #     hroi_cols = cbyz.df_get_cols_contains(
    #         df=ewtinst1c, 
    #         string=['_HROI', '_SELL', '_BUY']
    #         )
                
    #     if data_form == 1:
            
    #         #     ewtinst1c, cols_1, _, _ = \
    #         #         cbml.ml_data_process(df=ewtinst1c, 
    #         #                              ma=True, scale=True, lag=True, 
    #         #                              group_by=[],
    #         #                              cols=hroi_cols, 
    #         #                              except_cols=[],
    #         #                              drop_except=[],
    #         #                              cols_mode='contains',
    #         #                              date_col='WORK_DATE',
    #         #                              ma_values=ma_values, 
    #         #                              lag_period=predict_period
    #         #                              ) 
            
    #         #     ewtinst1c, cols_2, _, _ = \
    #         #         cbml.ml_data_process(df=ewtinst1c, 
    #         #                              ma=True, scale=True, lag=True, 
    #         #                              group_by=['SYMBOL'],
    #         #                              cols=['_HAP'], 
    #         #                              except_cols=[],
    #         #                              drop_except=[],
    #         #                              cols_mode='contains',
    #         #                              date_col='WORK_DATE',
    #         #                              ma_values=ma_values, 
    #         #                              lag_period=predict_period
    #         #                              )
            
            
    #         ewtinst1c, cols_1, _, _ = \
    #             cbml.ml_data_process(df=ewtinst1c, 
    #                                   ma=True, scale=True, lag=True, 
    #                                   group_by=[],
    #                                   sort_by=['SYMBOL', 'WORK_DATE'],
    #                                   cols=hroi_cols, 
    #                                   except_cols=[],
    #                                   drop_except=[],
    #                                   cols_mode='contains',
    #                                   date_col='WORK_DATE',
    #                                   ma_values=ma_values, 
    #                                   lag_period=predict_period
    #                                   ) 
        
    #         ewtinst1c, cols_2, _, _ = \
    #             cbml.ml_data_process(df=ewtinst1c, 
    #                                   ma=True, scale=True, lag=True, 
    #                                   group_by=['SYMBOL'],
    #                                   sort_by=['SYMBOL', 'WORK_DATE'],
    #                                   cols=['_HAP'], 
    #                                   except_cols=[],
    #                                   drop_except=[],
    #                                   cols_mode='contains',
    #                                   date_col='WORK_DATE',
    #                                   ma_values=ma_values, 
    #                                   lag_period=predict_period
    #                                   )            
            
    #     elif data_form == 2:
        
    #         pass
            

    #     main_data = main_data \
    #         .merge(ewtinst1c, how='left', on=['SYMBOL', 'WORK_DATE'])  
        
    #     print('ewtinst1c Check')
    #     print('Check - 全部填NA是否合理？')
    #     main_data = cbyz.df_conv_na(df=main_data, cols=cols_1 + cols_2)
    #     cbyz.df_chk_col_na(df=main_data, except_cols=var_y, mode='stop')


    # # 月營收資料表 ......
    # # 1. 當predict_date=20211101，且為dev時, 造成每一個symbol都有na，先移除
    # # 1. 主要邏輯就是顯示最新的營收資料
    # if market == 'tw':
        
    #     msg = '''Bug - sam_tej_get_ewsale，在1/18 23:00跑1/19時會出現chk_na error，但1/19 00:00過後
    #     再跑就正常了
    #     '''
    #     print(msg)
        
    #     ewsale = sam_tej_get_ewsale(begin_date=shift_begin)
    #     main_data = main_data \
    #                 .merge(ewsale, how='left', on=['SYMBOL', 'WORK_DATE'])      
    
    #     cbyz.df_chk_col_na(df=main_data, except_cols=var_y, mode='stop')
    
    
    # # 財務報表
    # # - 現在只用單季，需確認是否有缺漏
    # print('財務報表現在只用單季，需確認是否有缺漏')
    # # financial_statement = sam_tej_get_ewifinq()
    
    

    # # Pytrends Data ......
    # # - Increase prediction time a lot, and made mape decrease.
    # # - Pytrends已經normalize過後才pivot，但後面又normalize一次
    # # pytrends, pytrends_cols = get_google_treneds(begin_date=shift_begin, 
    # #                                               end_date=data_end, 
    # #                                               scale=True, 
    # #                                               stock_type=stock_type, 
    # #                                               local=local)
    
    # # main_data = main_data.merge(pytrends, how='left', on=['WORK_DATE'])      


    # 台股加權指數 TW 
    if market == 'tw':
         
        tw_index, cols = \
            sam_od_tw_get_index(
                begin_date=shift_begin,
                end_date=predict_date.loc[len(predict_date)-1, 'WORK_DATE']
                )
        
        # backup = main_data.copy()
        # main_data = backup.copy()
        
        main_data = main_data.merge(tw_index, how='left', on=time_key)

        main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                   sort_keys=time_key, 
                                   group_by=[], method='ffill')
    
        main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                   sort_keys=time_key, 
                                   group_by=[], method='bfill')     

        
    
    # S&P 500 ......
    snp, cols = sam_od_us_get_snp_data(begin_date=shift_begin)
    main_data = main_data.merge(snp, how='left', on=time_key)
    
    
    # COVID-19 ......
    if market == 'tw':
        covid_tw, cols = sam_covid_19_tw()
        
        # Future Plan
        # sam_covid_19_global()
            
        main_data = main_data.merge(covid_tw, how='left', on=time_key)
        main_data = cbyz.df_conv_na(df=main_data, cols=cols)
        
        main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                   sort_keys=time_key, 
                                   group_by=[], method='ffill')
    
        main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                   sort_keys=time_key, 
                                   group_by=[], method='bfill')        

    elif market == 'en':
        # Future Plan
        # covid_en = sam_covid_19_global()        
        pass


    # Variables ......
    model_x = cbyz.df_get_cols_except(df=main_data, 
                                      except_cols=var_y + id_keys)
    
    
    # Model Data ......
    # - 20220209，已經在sam_load_data中處理完了
    # main_data = main_data[main_data['WORK_DATE']>=data_begin] \
    #                     .reset_index(drop=True)


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
    
    
    # 當symbols=[]時，這裡會有18筆NA，都是var_y的欄位，應該是新股，因此直接排除
    chk_na = cbyz.df_chk_col_na(df=hist_df)

    assert len(chk_na) < 100, 'get_model_data - Check hist_df NA'
    na_cols = chk_na['COLUMN'].tolist()
    main_data = main_data.dropna(subset=na_cols, axis=0)
    
    
    # Predict有NA是正常的，但NA_COUNT必須全部都一樣
    global chk_predict_na
    if time_unit == 'd':
        predict_df = main_data.merge(predict_date, on=time_key)
    elif time_unit == 'w':
        predict_df = main_data.merge(predict_week, on=time_key)
    
    chk_predict_na = cbyz.df_chk_col_na(df=predict_df, mode='alert')
    
    
    print('data_form=2時會出錯')
    min_value = chk_predict_na['NA_COUNT'].min()
    max_value = chk_predict_na['NA_COUNT'].max()    
    assert min_value == max_value, 'All the NA_COUNT should be the same.'
    

    # Check min max ......
    # global chk_min_max
    # chk_min_max = cbyz.df_chk_col_min_max(df=main_data)
    
    # chk_min_max = \
    #     chk_min_max[(~chk_min_max['COLUMN'].isin(id_keys)) \
    #                 & ((chk_min_max['MIN_VALUE']<0) \
    #                    | (chk_min_max['MAX_VALUE']>1))]
    
    # assert len(chk_min_max) == 0, 'get_model_data - normalize error'


    # Export Model ......
    main_data.to_csv(path_temp + '/model_data_' + time_unit + '.csv', 
                     index=False)
    
    cbyz.li_to_csv(model_x, path_temp + '/model_x_' + time_unit + '.csv')
    
    scale_orig.to_csv(path_temp + '/scale_orig_' + time_unit + '.csv',
                      index=False)
        
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
    
    
    # v2.4 - 20220214
    # - Temp    
    
    
    
    # Update
    # Bug - sam_tej_get_ewsale，在1/18 23:00跑1/19時會出現chk_na error，但1/19 00:00過後
    #       再跑就正常。end_date應該要改成data_begin, 這個問題應該是today比data_begin少一天    
    # - Replace symbol with target, and add target_type which may be symbol
    #   or industry or 大盤
    # - Add financial_statement
    #   > 2021下半年還沒更新，需要改code，可以自動化更新並合併csv


    # - 確認TEJ財務報表的資料會不會自動更新
    # - Fix Support and resistant
    # - select_symbols用過去一周的總成交量檢查
    # - Short term model and long term model be overwirtted
    # 以close price normalize，評估高價股和低價股
    

    global version
    version = 2.3


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
    # 6. Handle week_num cyclical issues
    #    https://towardsdatascience.com/how-to-handle-cyclical-data-in-machine-learning-3e0336f7f97c
    
    
    # -NA issues, replace OHLC na in market data function, and add replace 
    # na with interpolation. And why 0101 not excluded?



    # Optimization .....
    # - 如果買TEJ的達人方案，那也不需要額外加購三大法人持股成本
    # 6. 技術分析型態學
    # 10. buy_signal to int
    # Update - 20211220，暫時在get_tw_index中呼叫update_tw_index，但這個方式可能會漏資料
    
    
    global bt_last_begin, data_period, predict_period, long, time_unit
    global dev, test
    global symbol, ma_values, volume_thld, market, data_form


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
        id_keys = ['SYMBOL', 'YEAR', 'WEEK_NUM']
        time_key = ['YEAR', 'WEEK_NUM']
        
    elif time_unit == 'd':
        id_keys = ['SYMBOL', 'WORK_DATE']    
        time_key = ['WORK_DATE']
    
    var_y = ['OPEN_CHANGE_RATIO', 'HIGH_CHANGE_RATIO',
              'LOW_CHANGE_RATIO', 'CLOSE_CHANGE_RATIO']

    # var_y = ['OPEN', 'HIGH',
    #           'LOW', 'CLOSE']    
    
    var_y_orig = [y + '_ORIG' for y in var_y]    
    
    
    # Update, add to BTM
    global wma
    wma = False
    
    
    # Calendar ------
    global shift_begin, shift_end, data_begin, data_end
    global predict_date, predict_week
    global calendar, calendar_full, calendar_full_key
    
    shift_begin, shift_end, data_begin, data_end, \
        predict_date, predict_week, calendar = \
                stk.get_period(predict_begin=predict_begin,
                               predict_period=predict_period,
                               data_period=data_period,
                               unit=time_unit,
                               shift=-int((max(ma_values) + 20)))
                
    # 有些資料可能包含非交易日，像是COVID-19，所以需要一個額外的calendar作比對
    calendar_full_key = calendar[['WORK_DATE', 'YEAR', 'WEEK_NUM']]
                
                
    # ......
    global model_data, model_x, scale_orig
    model_data, model_x, scale_orig = \
        get_model_data(industry=industry, 
                       trade_value=trade_value)
    
    
    # Training Model ......
    import xgboost as xgb
    from sklearn.linear_model import LinearRegression

    
    if len(symbol) > 0 and len(symbol) < 10:
        model_params = [{'model': LinearRegression(),
                         'params': {
                             'normalize': [True, False],
                             }
                         }]         
    else:
        
        # eta 0.01、0.03的效果都很差，目前測試0.08和0.1的效果較佳
        
        # data_form1 - Change Ratio
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
                            # 'eta': [0.08, 0.1],
                            'min_child_weight': [1],
                              # 'min_child_weight': [0.5, 1],
                            'max_depth':[8],
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
        
        
    # 1. 如果selectkbest的k設得太小時，importance最高的可能都是industry，導致同
    #    產業的預測值完全相同
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
                    tuner.fit(data=cur_model_data, 
                              model_params=model_params,
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


