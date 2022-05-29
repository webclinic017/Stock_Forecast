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
    path = r'D:\GitHub\Stock_Forecast\2_Stock_Analysis'
    path_dcm = r'D:\GitHub\Stock_Forecast\1_Data_Collection'


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'D:\Data_Mining\Projects\Codebase_YZ',
                 r'D:\GitHub\Arsenal',
                 r'/home/jupyter/Arsenal/20220522',
                 path + '/Function']

for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import codebase_ml as cbml

# import arsenal as ar
# import arsenal_stock as stk
import arsenal_v0200_dev as ar
import arsenal_stock_v0200_dev as stk

import ultra_tuner_v1_0100 as ut

ar.host = host



# 自動設定區 -------
pd.set_option('display.max_columns', 30)

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'

cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])        



# %% inner function ------


def get_market_data_raw(industry=True, trade_value=True, support_resist=false):
    
    
    global id_keys, time_key
    global symbol, market, var_y, var_y_orig
    global predict_period, time_unit
    global stock_info_raw
    global log, data_form
    global main_data_frame
    global calendar_key
    global market_data_raw, market_data
    
    
    stock_info_raw = stk.tw_get_stock_info(daily_backup=True, 
                                           path=path_temp)
        
    stock_info_raw = \
        stock_info_raw[['symbol', 'capital', 'capital_level', 
                        'establish_days', 'listing_days',
                        'industry_one_hot']]

    # market data ...
    # shift one day forward to get complete price_change_ratio
    loc_begin = cbyz.date_cal(shift_begin, -1, 'd')    
    
    
    # 避免重複query market_data_raw，在開發或debug時節省gcp sql的費用
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
                    adj=True
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
                    adj=True               
                    )

    market_data = market_data_raw.copy()
    
    # industry close

    # check        
    global ohlc
    for c in ohlc:
        col = c + '_change_ratio'
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


    # exclude low volume symbol ......
    market_data = select_symbols()


    # main_data_frame ......
    # - market_data will merge calendar in this function
    # - market_data會在這裡merge calendar
    set_frame()


    # first trading day ......
    # - this object will be used at the end of get_model_data
    global first_trading_day
    first_trading_day = market_data[['symbol', 'work_date']] \
            .sort_values(by=['symbol', 'work_date'], ascending=True) \
            .drop_duplicates(subset=['symbol']) \
            .rename(columns={'work_date':'first_trading_day'})
            

    # add k line ......
    market_data = market_data \
                    .sort_values(by=['symbol', 'work_date']) \
                    .reset_index(drop=True)
            
    market_data = stk.add_k_line(market_data)
    market_data = \
        cbml.df_get_dummies(
            df=market_data, 
            cols=['k_line_color', 'k_line_type']
        )
    
    
    # add support resistance ......
    
    # if support_resist:
    #     # check，確認寫法是否正確
    #     # print('add_support_resistance - days == True時有bug，全部數值一樣，導致
    #     # 沒辦法標準化？')
    #     global data_period
    #     market_data, _ = \
    #         stk.add_support_resistance(df=market_data, cols='close',
    #                                    rank_thld=int(data_period * 2 / 360),
    #                                    prominence=4, days=false)

    market_data = market_data \
        .dropna(subset=['work_date'], axis=0)

    market_data = ar.df_simplify_dtypes(df=market_data)


    # test ......

    # 執行到這裡，因為加入了預測區間，所以會有na，但所有na的數量應該要一樣多
    cbyz.df_chk_col_na(df=market_data, mode='stop')
    

    # check predict period
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
    global symbol_df, wma, corr_threshold
    global stock_info_raw
    global data_form
    global ohlc, ohlc_ratio, ohlc_change
    global df_summary_mean, df_summary_min, df_summary_max
    global df_summary_median, df_summary_std
    
    # new vars
    global y_scaler
    
    # scale market data ......
    # - loc_main和industry都需要scal ohlc，所以直接在market_data中處理，這樣就
    #   不需要做兩次
    # - 如果y是ohlc change ratio的話，by work_date或symbol的意義不大，反而讓運算
    #    速度變慢，唯一有影響的是一些從來沒有漲跌或跌停的symbol
    # - 先獨立把y的欄位標準化，因為這一段不用ma，但後面都需要
    # - 這裡的method要用1，如果用2的話，mse會變成0.8
    # - 因為method是1，其他大部份都是0，所以這一段要獨立出來
    # - price和change_ratio都用method 1 scale
    # - remove open price and open_price_change
    
    
    # 20220429 commented
    # market_data, scale_orig_ratio, y_scaler_ratio = \
    #     cbml.df_scaler(df=market_data, 
    #                    cols=['high_change_ratio', 'low_change_ratio',
    #                          'close_change_ratio'],
    #                    show_progress=false, method=1)
        
    ratio_cols = ['high_change_ratio', 'low_change_ratio', 
                  'close_change_ratio']        
        
    y_scaler_ratio = {}
    scale_orig_ratio = pd.dataframe()
    
    for c in ratio_cols:
        market_data, new_log, new_scaler = \
            cbml.df_scaler_v2(df=market_data, cols=c, except_cols=[],
                              method=0, alpha=0.05, export_scaler=True,
                              show_progress=True)
        
        y_scaler_ratio = {**y_scaler_ratio, **new_scaler}
        scale_orig_ratio = scale_orig_ratio.append(new_log)
        
        
    price_cols = ['high', 'low', 'close']
    
    y_scaler_price = {}
    scale_orig_price = pd.dataframe()
    
    for c in price_cols:
        market_data, new_log, new_scaler = \
            cbml.df_scaler_v2(df=market_data, cols=c, except_cols=[],
                              method=0, alpha=0.05, export_scaler=True,
                              show_progress=True)
        
        y_scaler_price = {**y_scaler_price, **new_scaler}
        scale_orig_price = scale_orig_price.append(new_log)
            
    
    open_cols = ['open', 'open_change_ratio']
    for c in open_cols:
        market_data, _, _ = \
            cbml.df_scaler_v2(df=market_data, cols=c, except_cols=[],
                              method=0, alpha=0.05, export_scaler=True,
                              show_progress=True)
        
    scale_orig = scale_orig_ratio.append(scale_orig_price)
    
    if 'close_change_ratio' in var_y:
        y_scaler = y_scaler_ratio
    elif 'close' in var_y:
        y_scaler = y_scaler_price
        
    pickle.dump(y_scaler, 
                open(path_temp + '/y_scaler_' + time_unit + '.sav', 'wb'))
    
    loc_main = market_data.copy()
    loc_main = loc_main.drop('total_trade_value', axis=1)
    
    
    # {y}_orig的欄位是用來計算ma
    for y in var_y:
        loc_main.loc[:, y + '_orig'] = loc_main[y]

    
    # process data
    if data_form == 1:
            
        # scale
        # - 即使time_unit是w，也需要ma，因為df_summary只考慮當周，不會考慮
        #   更久的ma
        cols = cbyz.df_get_cols_except(
            df=loc_main, 
            except_cols=ohlc+ohlc_ratio+['symbol', 'work_date']
            )

        loc_main, _, _ = cbml.df_scaler_v2(df=loc_main, cols=cols, 
                                           except_cols=[], method=0,
                                           alpha=0.05, export_scaler=false,
                                           show_progress=True)        
        
        # ma
        # 20220504 - remove
        # cols = \
        #     cbyz.df_get_cols_except(
        #         df=loc_main, 
        #         except_cols=['symbol', 'work_date'] + var_y
        #         )
            
        # loc_main, ma_cols_done = \
        #     cbyz.df_add_ma(df=loc_main, cols=cols,
        #                    group_by=['symbol'], 
        #                    date_col='work_date',
        #                    values=ma_values,
        #                    wma=wma, 
        #                    show_progress=false
        #                    )
        # loc_main = loc_main.drop(cols, axis=1)
        
        
    elif data_form == 2:
        
        except_cols = ['work_date', 'year_iso', 'month',
                       'weekday', 'week_num_iso'] + id_keys  

        # 新股會有na，但直接drop的話會刪到pred preiod
        loc_main.loc[:, 'remove'] = \
            np.where((loc_main['work_date']<data_end) \
                     & (loc_main[var_y[-1]].isna()), 1, 0)
                
        loc_main = loc_main[loc_main['remove']==0] \
                    .drop('remove', axis=1)                
        
        # 新股上市的第一天，open_change會是inf
        print('check - 為什麼open_change_ratio不是inf，但open_change_abs_ratio是')
        loc_main.loc[:, 'open_change_abs_ratio'] = \
            np.where(loc_main['open_change_abs_ratio']==np.inf, 
                     0, loc_main['open_change_abs_ratio'])

        loc_main, _, _ = \
            cbml.df_scaler(
                df=loc_main,
                except_cols=except_cols,
                show_progress=false,
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
        #         drop=false)        
    
        
    
    # total market trade
    if trade_value:
        
        total_trade = market_data[['work_date', 'total_trade_value']] \
                        .drop_duplicates(subset=['work_date'])

        if data_form == 1:    

            # # scale data
            # total_trade, _, _ = cbml.df_scaler(df=total_trade, 
            #                                    cols='total_trade_value',
            #                                    method=0)
            # # ma
            # total_trade, _ = \
            #     cbyz.df_add_ma(df=total_trade, cols='total_trade_value',
            #                    group_by=[], date_col='work_date',
            #                    values=ma_values, wma=wma, 
            #                    show_progress=false
            #                    )   
            #
            # total_trade = total_trade.drop('total_trade_value', axis=1)
            
            total_trade, new_scale_log, _ = \
                cbml.df_scaler_v2(df=total_trade,
                                  cols='total_trade_value', 
                                  except_cols=[], method=0,
                                  alpha=0.05, export_scaler=false,
                                  show_progress=True)               
            
        elif data_form == 2:
                
            total_trade, _, _ = \
                cbml.df_scaler(
                    df=total_trade,
                    except_cols=['work_date'],
                    show_progress=false,
                    method=0
                    )  
        
            # total_trade, _ = \
            #     cbml.ml_df_to_time_series(
            #         df=total_trade, 
            #         cols=[], 
            #         except_cols='work_date',
            #         group_by=[],
            #         sort_keys='work_date', 
            #         window=1,
            #         drop=True)    
        
        loc_main = loc_main.merge(total_trade, how='left', on=['work_date'])  


    # stock info ...
    stock_info = stock_info_raw.drop('industry_one_hot', axis=1)
        
    stock_info, new_scale_log, _ = \
        cbml.df_scaler_v2(df=stock_info, cols=[], except_cols=id_keys,
                          method=0, alpha=0.05, export_scaler=false,
                          show_progress=True)        
        
    info_cols = cbyz.df_get_cols_except(df=stock_info, except_cols=id_keys)        
    loc_main = loc_main.merge(stock_info, how='left', on='symbol')
    

    # industry ......       
    # - 因為industry_data中會用到total_trade_value，所以total_trade_value沒辦法
    #   先獨立處理
    if industry: 
        stock_industry = stock_info_raw[['symbol', 'industry_one_hot']]
        
        stock_info_dummy = \
            cbml.df_get_dummies(df=stock_industry, 
                                cols='industry_one_hot'
                                )
        
        # industry data and trade value ...
        # print('sam_load_data - 當有新股上市時，產業資料的比例會出現大幅變化，' \
        #       + '評估如何處理')
        industry_data = \
            market_data[['symbol', 'work_date', 'volume'] \
                        + ohlc + ohlc_change + \
                        ['symbol_trade_value', 'total_trade_value']]

        # merge        
        industry_data = industry_data.merge(stock_industry, on='symbol')
        
        industry_data['trade_value'] = \
            industry_data \
            .groupby(['work_date', 'industry_one_hot'])['symbol_trade_value'] \
            .transform('sum')

        industry_data['trade_value_ratio'] = \
            industry_data['trade_value'] / industry_data['total_trade_value']            
        
        industry_data = industry_data[['work_date', 'industry_one_hot'] \
                                       + ohlc + ohlc_change + \
                                       ['trade_value', 'trade_value_ratio']]
        
        industry_data = industry_data \
                        .groupby(['work_date', 'industry_one_hot']) \
                        .mean() \
                        .reset_index()
        
        # rename ...
        cols = cbyz.df_get_cols_except(
            df=industry_data,
            except_cols=['work_date', 'industry_one_hot']
            )
        
        new_cols = ['industry_' + c for c in cols]                  
        rename_dict = cbyz.li_to_dict(cols, new_cols)
        industry_data = industry_data.rename(columns=rename_dict)
                   
        
        # ma ......
        # cols = cbyz.df_get_cols_except(
        #     df=industry_data, 
        #     except_cols=['work_date', 'industry_one_hot']
        #     )
        
        # industry_data, _ = \
        #     cbyz.df_add_ma(df=industry_data, cols=cols,
        #                    group_by=[], date_col='work_date',
        #                    values=ma_values, wma=wma, 
        #                    show_progress=false
        #                    )    
        # industry_data = industry_data.drop(cols, axis=1)   
        
        
        # merge ...
        # .merge(stock_info_dummy, how='left', on='symbol') \
        loc_main = loc_main \
            .merge(stock_industry, how='left', on='symbol') \
            .merge(industry_data, how='left', on=['work_date', 'industry_one_hot']) \
            .drop('industry_one_hot', axis=1)
        



    if time_unit == 'w':

        # merge market data and calendar
        new_loc_main = main_data_frame \
            .merge(calendar_key, how='left', on=time_key)
        
        
        # 這裡merge完後，shift_period的year_iso和week_num_iso中會有na，且shift_period
        # 已經超出calendar的範圍，這是正常的
        loc_main = new_loc_main \
            .merge(loc_main, how='outer', on=['symbol', 'work_date'])


        # merge完main_data_frame後，新股的work_date會是na
        loc_main = loc_main.dropna(subset=['work_date'] + time_key, axis=0)
        loc_main = loc_main.drop('work_date', axis=1)
        
        cols = cbyz.df_get_cols_except(
                df=loc_main, 
                except_cols=id_keys + var_y + info_cols
                )
        
        
        # - df_summary中的groupby會讓var_y消失，所以需要先獨立出來，而且也
        #   必須df_summary
        # hyperparameter，目前y_data用mean aggregate，不確定median會不會比較好
        y_data = loc_main[id_keys + var_y]
        
        y_data = y_data \
                .groupby(id_keys) \
                .mean() \
                .reset_index()
        
        
        # y用平均就好，不要用high of high, low of low，避免漲一天跌四天
        print('skew很容易產生na，先移除 / 還是每個skew的第一個數值都是na？')
        
        loc_main, _ = \
            cbyz.df_summary(df=loc_main, cols=cols, 
                            group_by=id_keys, 
                            add_mean=df_summary_mean, 
                            add_min=df_summary_min, 
                            add_max=df_summary_max, 
                            add_median=df_summary_median,
                            add_std=df_summary_std, 
                            add_skew=false, add_count=false, quantile=[])

        
    elif time_unit == 'd':
        
        # 這裡merge完後，shift_period的year和week_num_iso中會有na，且shift_period
        # 已經超出calendar的範圍，這是正常的
        loc_main = main_data_frame \
            .merge(loc_main, how='outer', on=['symbol', 'work_date'])        




    # ma
    # - 20220504 add
    cols = \
        cbyz.df_get_cols_except(
            df=loc_main, 
            except_cols=['symbol', 'work_date', 'year_week_iso'] \
                + var_y + info_cols
            )
        
            
    global debug_df
    debug_df = loc_main.copy()
    
    loc_main, ma_cols_done = \
        cbyz.df_add_ma(df=loc_main, cols=cols,
                       group_by=['symbol'], 
                       date_col=time_key,
                       values=ma_values,
                       wma=wma, 
                       show_progress=false
                       )
    loc_main = loc_main.drop(cols, axis=1)
    
        
    loc_main = loc_main.merge(y_data, how='left', on=id_keys)
    del y_data
    gc.collect()




    # shift y ......
    # 1. 因為x與y之間需要做shift，原始的版本都是移動x，但是x的欄位很多，因此改成
    #    移動y，提升計算效率
    loc_main, new_cols = \
        cbyz.df_add_shift(
            df=loc_main, 
            cols=time_key + var_y, 
            shift=-predict_period, 
            group_by=['symbol'], 
            sort_by=id_keys,
            suffix='', 
            remove_na=false
            )
    
    
    # industry dummy variables
    # - 因為one hot encoding後的industry欄位不用df_summary，也不用shift，所以最後
    #   再merge即可
    # - 20220506, industry的dummary variable可能不重要，因為趨勢會變，重要的只是
    # cash flow and volume of each industry
    # if industry:
    #     loc_main = loc_main.merge(stock_info_dummy, how='left', on='symbol')


    # 檢查shift完後year和week_num_iso的na
    chk_na = cbyz.df_chk_col_na(df=loc_main, cols=time_key)
    na_min = chk_na['na_count'].min()
    na_max = chk_na['na_count'].max()
    
    assert na_min == na_max and na_min == len(symbol_df) * predict_period, \
        'error after shift'


    # 必須在df_add_shift才dropna，否則準備往前推的work_date會被刪除；shift完後，
    # id_keys中會有na，但這裡的na就可以drop
    cols = cbyz.df_get_cols_except(df=loc_main, except_cols=var_y)
    
    
    # 因為計算ma及df_shift，所以開頭會有na，time_key也因為na，所以變成float，
    # 因此drop後要simplify_dtypes
    if data_form == 1:
        loc_main = loc_main.dropna(subset=cols, axis=0)
    elif data_form == 2:
        pass

    loc_main = ar.df_simplify_dtypes(df=loc_main)


    # simplify dtypes
    # - year and week_num_iso will be float here
    loc_main = ar.df_simplify_dtypes(df=loc_main)


    # check na ......
    # 有些新股因為上市時間較晚，在ma_lag中會有較多的na，所以只處理ma的欄位
    na_cols = cbyz.df_chk_col_na(df=loc_main, except_cols=var_y)
    # na_cols = cbyz.df_chk_col_na(df=loc_main)
    

    # drop highly correlated features
    loc_main = cbml.df_drop_high_corr_var(df=loc_main, threshold=corr_threshold, 
                                          except_cols=id_keys + var_y)
    
    
    # df_summary可能造成每一欄的na數不一樣，所以先排除time_unit = 'w'
    
    if time_unit == 'd':

        assert len(na_cols) == 0 \
            or na_cols['na_count'].min() == na_cols['na_count'].max(), \
            'all the na_count should be the same.'
    
        na_cols = na_cols['column'].tolist()
        loc_main = loc_main.dropna(subset=na_cols, axis=0)
        
    elif time_unit == 'w':
        pass
        
    # check for min max
    # - check, temporaily remove for data_from 2
    # chk_min_max = cbyz.df_chk_col_min_max(df=loc_main)
    # chk_min_max = chk_min_max[chk_min_max['column']!='work_date']
    # chk_min_max = chk_min_max[(chk_min_max['min_value']<0) \
    #                           | (chk_min_max['max_value']>1)]
        
    # assert len(chk_min_max) == 0, 'chk_min_max failed'
    
    return loc_main, scale_orig


# .............


def select_symbols():

    '''
    version note
    
    1. exclude small capital

    '''    

    global market_data
    global stock_info_raw


    # exclude etf ......
    all_symbols = stock_info_raw[['symbol']]
    df = all_symbols.merge(market_data, on=['symbol']) 


    # exclude low volume in the past 7 days
    global volume_thld
    global data_end
    loc_begin = cbyz.date_cal(data_end, -7, 'd')
    
    low_volume = df[(df['work_date']>=loc_begin) & (df['work_date']<=data_end)]
    low_volume = low_volume \
                .groupby(['symbol']) \
                .agg({'volume':'min'}) \
                .reset_index()
        
    low_volume = low_volume[low_volume['volume']<=volume_thld * 1000]
    low_volume = low_volume[['symbol']].drop_duplicates()
    
    global low_volume_symbols
    low_volume_symbols = low_volume['symbol'].tolist()
    
    # 為了避免low_volume_symbols的數量過多，導致計算效率差，因此採用df做anti_merge，
    # 而不是直接用list
    if len(low_volume_symbols) > 0:
        df = cbyz.df_anti_merge(df, low_volume, on='symbol')
        
    # add log
    log_msg = 'low_volume_symbols - ' + str(len(low_volume_symbols))
    log.append(log_msg)        
    
    return df



def set_frame():
    
    
    # merge as main data ......
    global symbol_df, id_keys, time_key, time_unit
    global market_data, calendar, calendar_proc
    global predict_week, predict_date
    global main_data_frame, main_data_frame_calendar

    # new global variables
    global symbol_df
    global calendar_lite, calendar_proc, calendar_key


    # predict symbols ......
    # 1. prevent symbols excluded by select_symbols(), but still exists.
    all_symbols = market_data['symbol'].unique().tolist()
    symbol_df = pd.dataframe({'symbol':all_symbols})
    
    
    # calendar ......
    calendar_proc = calendar[calendar['trade_date']>0] \
                    .reset_index(drop=True) \
                    .reset_index() \
                    .rename(columns={'index':'date_index'})
                   
                    
    calendar_lite = calendar[calendar['trade_date']>0]                     
    calendar_lite = calendar_lite[time_key] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
                    
               
    # remove untrading date
    calendar_key = calendar[calendar['trade_date']>0].reset_index(drop=True)
    if time_unit == 'd':
        calendar_key = calendar_key[['work_date']]
        
    elif time_unit == 'w':
        calendar_key = calendar_key[['work_date', 'year_week_iso']]
    
    
    # duplicate year and week_num, then these two columns can be variables 
    # of the model
    if time_unit == 'w':
        
        calendar_proc = calendar_proc[['year_week_iso', 'year_iso', 'month', 
                                       'week_num_iso']]
        
        # 如果加入year和month，可能會有同一周，但是跨年度的問題
        calendar_proc = calendar_proc \
            .drop_duplicates(subset=['year_week_iso', 'week_num_iso']) \
            .reset_index(drop=True) \
            .reset_index() \
            .rename(columns={'index':'date_index'}) 

    calendar_proc, _, _ = \
        cbml.df_scaler(
            df=calendar_proc,
            except_cols=id_keys,
            show_progress=false,
            method=1
            )           
    
    
    # main_data_frame ......
    main_data_frame = cbyz.df_cross_join(symbol_df, calendar_lite)
    
    
    if time_unit == 'd':
        max_date = predict_date['work_date'].max()
        main_data_frame = \
            main_data_frame[main_data_frame['work_date']<=max_date]
            
    elif time_unit == 'w':
        pass
        print('chk - 這裡是否需要篩選日期')


    # organize
    main_data_frame = main_data_frame[id_keys]
    
    
    main_data_frame = ar.df_simplify_dtypes(df=main_data_frame)
    
    print('check - main_data_frame_calendar是否等同calendar_lite')
    main_data_frame_calendar = calendar_lite.copy()
    # main_data_frame_calendar = main_data_frame[time_key] \
    #                             .drop_duplicates() \
    #                             .sort_values(by='work_date') \
    #                             .reset_index(drop=True)



# %% tw variables ------


def sam_buffett_indicator():
    
    global id_keys, time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma, corr_threshold
    global scale_log
    
    result = stk.cal_buffett_indicator(
            end_date=predict_date.loc[len(predict_date) - 1, 'work_date']
            )
    
    cols = cbyz.df_get_cols_except(df=result, except_cols='work_date')


    if time_unit == 'w':

        result = calendar_full_key \
            .merge(result, how='left', on='work_date') \
            .drop(['work_date'], axis=1)

        result, cols = \
            cbyz.df_summary(
                df=result, cols=cols, group_by=time_key, 
                add_mean=True, add_min=false, 
                add_max=false, add_median=false, add_std=True, 
                add_skew=false, add_count=false, quantile=[]
                )    
            
            
    # scale data
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=cols, except_cols=[],
                          method=0, alpha=0.05, export_scaler=false,
                          show_progress=True)
    
    scale_log = scale_log.append(new_scale_log)
    
        
    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=[], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
        
    result = result.drop(cols, axis=1)            
            

    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)
         
    # filter exist columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)

    # merge ......
    result = main_data_frame_calendar.merge(result, how='left', on=time_key)    

    return result, ma_cols


# .........


def sam_covid_19_tw():
    
    global id_keys, time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma, corr_threshold
    global df_summary_mean, df_summary_min, df_summary_max
    global df_summary_median, df_summary_std    
    global scale_log
                
    
    result, _ = cbyz.get_covid19_data(backup=True, path=path_temp)
    
    cols = cbyz.df_get_cols_except(
        df=result,
        except_cols=['work_date', 'year_week_iso']
        )
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='work_date')
            
        result = cbyz.df_conv_na(df=result, cols=cols)  
        
        result, cols = \
            cbyz.df_summary(
                df=result, cols=cols, group_by=time_key, 
                add_mean=df_summary_mean, 
                add_min=df_summary_min, 
                add_max=df_summary_max, 
                add_median=df_summary_median,
                add_std=df_summary_std, 
                add_skew=false, add_count=false, quantile=[]
                )
    
    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=[], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
        
    result = result.drop(cols, axis=1)


    # scale data
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=ma_cols, except_cols=[],
                          method=0, alpha=0.05, export_scaler=false,
                          show_progress=True)
        
    scale_log = scale_log.append(new_scale_log)        
        
        
    
    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)
        
    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)
    
    
    # merge ......
    result = main_data_frame_calendar.merge(result, how='left', on=time_key)    
    gc.collect()

    return result, ma_cols


# .................
  
    
def sam_ex_dividend():
    
    global id_keys, time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma, corr_threshold
    
    
    # # close lag ...
    # daily_close = market_data[['work_date', 'symbol', 'close']]
    
    # daily_close, _ = \
    #     cbyz.df_add_shift(df=daily_close, 
    #                       sort_by=['symbol', 'work_date'],
    #                       cols='close', shift=1,
    #                       group_by=['symbol'],
    #                       suffix='_lag', 
    #                       remove_na=false)
        
    # daily_close = daily_close \
    #             .drop('close', axis=1) \
    #             .rename(columns={'close_lag':'close'})

    # daily_close = cbyz.df_fillna_chain(df=daily_close, cols='close',
    #                                    sort_keys=['symbol', 'work_date'],
    #                                    group_by=['symbol'],
    #                                    method=['ffill', 'bfill'])

    print('sam_ex_dividend - only return date')
    result = stk.od_tw_get_ex_dividends()
    cols = cbyz.df_get_cols_except(df=result, 
                                   except_cols=['symbol', 'work_date'])
    
    cbyz.df_chk_col_na(df=result, mode='stop')
    
    # scale data
    # result, _, _ = cbml.df_scaler(df=result, 
    #                                 cols=cols,
    #                                 method=0)
    
    # ma
    # result, ma_cols = \
    #     cbyz.df_add_ma(df=result, cols=cols,
    #                    group_by=[], date_col='work_date',
    #                    values=ma_values, wma=wma, 
    #                    show_progress=false
    #                    )   
        
    # result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
                .merge(calendar_full_key, how='left', on='work_date') \
                .drop('work_date', axis=1) \
                .dropna(subset=time_key, axis=0)
            
        cbyz.df_chk_col_na(df=result, mode='stop')            
            
        result = result \
                .groupby(id_keys) \
                .max() \
                .reset_index()
                
                
    # drop highly correlated features                
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)
        
    # filter existing columns
    cols = cbyz.df_filter_exist_cols(df=result, cols=cols)    
            
    # merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key) \
                .dropna(subset=['symbol'], axis=0)

    result = cbyz.df_conv_na(df=result, cols=cols, value=0)
    result = ar.df_simplify_dtypes(df=result)

    return result, cols


# .................


def sam_tw_gov_invest(dev=false):
    
    global symbol_df, calendar_key
    global id_keys, time_key, var_y, corr_threshold
    
    result = stk.od_tw_get_gov_invest(path=path_resource)
    
    if not dev:
        result = result.merge(symbol_df, on='symbol')
    else:
        # 這個結果merge後需要再fillna by ffill，為了避免和model_data沒有交集，
        # 所以增加dev mode
        test_symbol = symbol[0:len(result)]
        result['symbol'] = test_symbol
    
    if time_unit == 'w':
        result = result.merge(calendar_key, how='left', on='work_date')
        result = result.drop('work_date', axis=1)
        
        
    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)
    
    
    print('是否需要return cols，並df_filter_exist_cols')
    
    return result
    

# .................
    

def sam_tw_gov_own(dev=false):
    
    global id_keys, time_key, symbol_df, calendar_key
    global corr_threshold
    
    result = stk.od_tw_get_gov_own(path=path_resource)
    
    if not dev:
        result = result.merge(symbol_df, on='symbol')
    else:
        # 這個結果merge後需要再fillna by ffill，為了避免和model_data沒有交集，
        # 所以增加dev mode
        test_symbol = symbol[0:len(result)]
        result['symbol'] = test_symbol
    
    if time_unit == 'w':
        result = result.merge(calendar_key, how='left', on='work_date')
        result = result.drop('work_date', axis=1)
        
    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)     
    
    print('是否需要return cols，並df_filter_exist_cols')
    
    return result
    

# .................
    

def sam_od_tw_get_fx_rate():
    
    global id_keys, time_key
    global calendar_full_key, main_data_frame_calendar
    global ma_values, wma, corr_threshold
    global scaler_log
    
    result = stk.od_tw_get_fx_rate()
    cols = cbyz.df_get_cols_except(df=result, except_cols=['work_date'])
    result = pd.melt(result, id_vars=['work_date'], value_vars=cols,
                     var_name='currency', value_name='fx_rate')

    
    if time_unit == 'w':
        
        result = result \
            .merge(calendar_full_key, how='left', on='work_date')
            
        result = result.dropna(subset=time_key, axis=0)            
            
        result = cbyz.df_fillna_chain(df=result, cols='fx_rate',
                                      sort_keys='work_date', 
                                      method=['ffill', 'bfill'], 
                                      group_by=[])
        
        result, cols = cbyz.df_summary(
            df=result, cols='fx_rate', group_by=time_key + ['currency'], 
            add_mean=True, add_min=false, 
            add_max=false, add_median=false, add_std=True, 
            add_skew=false, add_count=false, quantile=[]
            )


    # pivot
    result = result \
            .pivot_table(index=time_key, columns='currency',
                         values=cols) \
            .reset_index()

    result = cbyz.df_flatten_columns(df=result)
    cols = cbyz.df_get_cols_except(df=result, except_cols=id_keys)

    
    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=[], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
    result = result.drop(cols, axis=1)
    
    
    # scale
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=ma_cols, except_cols=[], method=0,
                          alpha=0.05, export_scaler=false, 
                          show_progress=True)
        
    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)   
    
    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)
        
    return result, ma_cols       


# .................


def sam_od_tw_get_index(begin_date, end_date):
    
    global ma_values, predict_period, corr_threshold
    global id_keys, time_unit, time_key
    global calendar, main_data_frame_calendar
    global scale_log
    
    # main_data_frame_calendar
    # calendar_full_key
    
    result = stk.od_tw_get_index()
    cols = cbyz.df_get_cols_except(df=result, except_cols='work_date')
    
    # 如果有na的話，可能要先做fillna
    cbyz.df_chk_col_na(df=result, mode='stop')
    

    
    if time_unit == 'w':
        
        result = result \
            .merge(calendar_full_key, how='left', on='work_date') \
            .drop(['work_date'], axis=1)
            
        result, cols = cbyz.df_summary(
                df=result, cols=cols, group_by=time_key, 
                add_mean=True, add_min=false, 
                add_max=false, add_median=false, add_std=True, 
                add_skew=false, add_count=false, quantile=[]
            )


    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=[], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
        
    result = result.drop(cols, axis=1)

    # scale data
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=ma_cols, except_cols=[], method=0,
                          alpha=0.05, export_scaler=false, show_progress=True)
    
    scale_log = scale_log.append(new_scale_log)
    

    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)    
    
    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    
    
    # merge ......
    result = main_data_frame_calendar \
                .merge(result, how='left', on=time_key)   

    return result, ma_cols


# .................


def sam_od_us_get_dji():
    
    # dow jones industrial average (^dji)
    global id_keys, corr_threshold
    global ma_values, predict_period, predict_date
    global calendar, main_data_frame_calendar
    global sam_od_us_get_dji
    global scale_log
    
    loc_df = stk.od_us_get_dji(daily_backup=True, path=path_temp)
    cols = cbyz.df_get_cols_except(df=loc_df, except_cols=['work_date'])
    
    # handle time lag
    # - 20220317 - 移至stk，待確認是否會出錯，沒問題的話這一段就刪除
    # loc_df = loc_df.rename(columns={'work_date':'work_date_orig'})
    # loc_df = cbyz.df_date_cal(df=loc_df, amount=-1, unit='d',
    #                           new_cols='work_date',
    #                           cols='work_date_orig')
    
    # loc_df = loc_df.drop('work_date_orig', axis=1)


    # fillna .......
    # 1. 因為美股的交易時間可能和台灣不一樣，包含特殊節日等，為了避免日期無法
    #    對應，用fillna
    #    補上完整的日期
    # - 20220317 - 移至stk，待確認是否會出錯，沒問題的話這一段就刪除
    
    # loc_calendar = cbyz.date_get_calendar(begin_date=begin_date, 
    #                                       end_date=predict_date[-1])
    # loc_calendar = calendar[['work_date']]
    # loc_df = loc_calendar.merge(loc_df, how='left', on='work_date')
    # cols = cbyz.df_get_cols_except(df=loc_df, except_cols='work_date')
    # loc_df = cbyz.df_fillna(df=loc_df, cols=cols, sort_keys='work_date', 
    #                         group_by=[], method='ffill')


    # agg for weekly prediction
    if time_unit == 'w':
        loc_df = loc_df \
            .merge(calendar_full_key, how='left', on='work_date')
        
        loc_df, cols = cbyz.df_summary(
            df=loc_df, cols=cols, group_by=time_key, 
            add_mean=True, add_min=false, 
            add_max=false, add_median=false, add_std=True, 
            add_skew=false, add_count=false, quantile=[]
            )    
    
    # ma
    loc_df, ma_cols = \
        cbyz.df_add_ma(df=loc_df, cols=cols,
                       group_by=[], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
    loc_df = loc_df.drop(cols, axis=1)
    
    
    # scale
    loc_df, new_scale_log, _ = \
        cbml.df_scaler_v2(df=loc_df, cols=ma_cols, except_cols=[],
                          method=0, alpha=0.05, 
                          export_scaler=false, show_progress=True)
        
    scale_log = scale_log.append(new_scale_log)        


    # drop highly correlated features
    loc_df = cbml.df_drop_high_corr_var(df=loc_df, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=loc_df, cols=ma_cols)    
        
    return loc_df, ma_cols    


# .................


def sam_od_us_get_snp(begin_date):
    
    global id_keys, corr_threshold
    global ma_values, predict_period, predict_date
    global calendar, main_data_frame_calendar
    global scale_log
    
    loc_df = stk.od_us_get_snp(daily_backup=True, path=path_temp)
    cols = cbyz.df_get_cols_except(df=loc_df, except_cols=['work_date'])
    
    # handle time lag
    # - 20220317 - 移至stk，待確認是否會出錯，沒問題的話這一段就刪除
    # loc_df = loc_df.rename(columns={'work_date':'work_date_orig'})
    # loc_df = cbyz.df_date_cal(df=loc_df, amount=-1, unit='d',
    #                           new_cols='work_date',
    #                           cols='work_date_orig')
    
    # loc_df = loc_df.drop('work_date_orig', axis=1)


    # fillna .......
    # - 因為美股的交易時間可能和台灣不一樣，包含特殊節日等，為了避免日期無法
    #   對應，用fillna補上完整的日期
    # - 20220317 - 移至stk，待確認是否會出錯，沒問題的話這一段就刪除
    
    # loc_calendar = cbyz.date_get_calendar(begin_date=begin_date, 
    #                                       end_date=predict_date[-1])
    # loc_calendar = calendar[['work_date']]
    
    # loc_df = loc_calendar.merge(loc_df, how='left', on='work_date')
    # cols = cbyz.df_get_cols_except(df=loc_df, except_cols='work_date')
    # loc_df = cbyz.df_fillna(df=loc_df, cols=cols, sort_keys='work_date', 
    #                         group_by=[], method='ffill')

    
    # agg for weekly prediction
    if time_unit == 'w':
        loc_df = loc_df.merge(calendar_full_key, how='left', on='work_date')
        
        loc_df, cols = cbyz.df_summary(
                df=loc_df, cols=cols, group_by=time_key, 
                add_mean=True, add_min=false, 
                add_max=false, add_median=True, add_std=True, 
                add_skew=false, add_count=false, quantile=[]
            )    


    # scale
    loc_df, new_scale_log, _ = \
        cbml.df_scaler_v2(df=loc_df, cols=cols, except_cols=[], method=0,
                          alpha=0.05, export_scaler=false, show_progress=True) 
        
    scale_log = scale_log.append(new_scale_log)        
        
    
    # ma
    loc_df, ma_cols = \
        cbyz.df_add_ma(df=loc_df, cols=cols,
                       group_by=[], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
    loc_df = loc_df.drop(cols, axis=1)

    

    # drop highly correlated features
    loc_df = cbml.df_drop_high_corr_var(df=loc_df, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=loc_df, cols=ma_cols)    
        
    return loc_df, ma_cols


# ...............


def sam_tdcc_get_sharehold_spread():
    
    global main_data_frame, time_unit
    global id_keys, corr_threshold
    global symbol
    global shift_begin    
    global scale_log
    global ma_values
    
    assert time_unit == 'w', 'update for time_unit not being week'
    result = stk.tdcc_get_sharehold_spread(begin_date=shift_begin,
                                           end_date=none,
                                           ratio_interval=10,
                                           threshold=60)
    
    cols = cbyz.df_get_cols_except(df=result, except_cols=id_keys)
    


    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['symbol'], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
    result = result.drop(cols, axis=1)

    
    # scale data
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=ma_cols, except_cols=[], method=0,
                          alpha=0.05, export_scaler=false, show_progress=True) 
        
    scale_log = scale_log.append(new_scale_log)       


    # combine
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='symbol')

    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    cols = cbyz.df_filter_exist_cols(df=result, cols=cols)            
    
    return result, cols


# ...............


def sam_tej_ewsale(begin_date):

    
    global main_data_frame, symbol
    global time_unit, time_key, id_keys
    global scale_log
    
    result = stk.tej_ewsale(begin_date=begin_date, end_date=none, 
                            symbol=symbol, fill=True, host=host,
                            time_unit=time_unit)
    
    # 這三個欄位應該是新的，所以裡面會有一大堆na
    # result = result.drop(['d0005', 'd0006', 'd0007'], axis=1)
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=id_keys
        )
    
    
    # merge will cause na, so it must to execute df_fillna
    result = main_data_frame.merge(result, how='left', on=id_keys)
    
    result = cbyz.df_fillna_chain(df=result, cols=cols,
                                  sort_keys=id_keys, 
                                  group_by=['symbol'],
                                  method=['ffill', 'bfill'])
    
    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['symbol'], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
    result = result.drop(cols, axis=1)
    
    
    # scale
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=ma_cols, except_cols=[],
                          method=0, alpha=0.05, 
                          export_scaler=false, show_progress=True)
        
    scale_log = scale_log.append(new_scale_log)        

        
    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    cols = cbyz.df_filter_exist_cols(df=result, cols=cols)               

    return result, cols


# ...........


def sam_tej_ewgin():
    
    global id_keys, corr_threshold
    global symbol
    global shift_begin
    global scale_log
    
    result = stk.tej_ewgin(begin_date=shift_begin, end_date=none, 
                           symbol=symbol)
    
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['symbol', 'work_date']
        )
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='work_date')
            
        result = cbyz.df_conv_na(df=result, cols=cols)  
        
        result, cols = cbyz.df_summary(
            df=result, cols=cols, group_by=id_keys, 
            add_mean=True, add_min=false, 
            add_max=false, add_median=false, add_std=True, 
            add_skew=false, add_count=false, quantile=[]
            )    


    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['symbol'], date_col=time_key,
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
        
    result = result.drop(cols, axis=1)


    # scale data
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=ma_cols, except_cols=[],
                          method=0, alpha=0.05, export_scaler=false,
                          show_progress=True)
    
    scale_log = scale_log.append(new_scale_log)


    # merge and fill na
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='symbol')

    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    

    return result, ma_cols   


# ...........


def sam_tej_ewifinq():

    global ma_values, predict_period, predict_date
    global calendar, main_data_frame_calendar
    global symbol
    global scale_log
    global corr_threshold
    global time_unit
    
    
    result = stk.tej_ewifinq(time_unit=time_unit, target_type='symbol',
                             target=symbol)

    cols = cbyz.df_get_cols_except(df=result, except_cols=id_keys)
    
    result = cbyz.df_fillna_chain(df=result, cols=cols,
                                  sort_keys=id_keys,
                                  method=['ffill'], 
                                  group_by='symbol')        
    
    cbyz.df_chk_col_na(df=result, mode='stop')
    
        
    # scale data
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=cols,
                          except_cols=id_keys,
                          method=0, alpha=0.05, export_scaler=false,
                          show_progress=True)
    
    scale_log = scale_log.append(new_scale_log)            
        
        
    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys)         
        
    result = main_data_frame.merge(result, how='left', on=id_keys)
    cols = cbyz.df_get_cols_except(df=result, except_cols=id_keys)
    
    return result, cols


# ..............
    


def sam_tej_ewtinst1():
    
    global id_keys, corr_threshold
    global symbol
    global shift_begin
    global scale_log
    
    result = stk.tej_ewtinst1(begin_date=shift_begin, end_date=none, 
                              symbol=symbol)
    
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['symbol', 'work_date']
        )
    
    
    if time_unit == 'w':
        result = result.merge(calendar_full_key, how='left', on='work_date')
        result = cbyz.df_conv_na(df=result, cols=cols)  
        
        result, cols = cbyz.df_summary(
            df=result, cols=cols, group_by=id_keys, 
            add_mean=True, add_min=false, 
            add_max=false, add_median=false, add_std=True, 
            add_skew=false, add_count=false, quantile=[]
            )    
    
    
    # calculate ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                        group_by=['symbol'], date_col=time_key,
                        values=ma_values, wma=wma, 
                        show_progress=false
                        )       
    
    result = result.drop(cols, axis=1)    


    # scale data
    result, new_scale_log, _ = \
        cbml.df_scaler_v2(df=result, cols=ma_cols, except_cols=[],
                          method=0, alpha=0.05, export_scaler=false,
                          show_progress=True)
    
    scale_log = scale_log.append(new_scale_log)
    
    
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='symbol')
    
        
    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    
    gc.collect()

    return result, ma_cols   



# ..............
    

def sam_tej_ewprcd_pe_ratio():
    
    global id_keys, corr_threshold
    global symbol
    global symbol_df
    global shift_begin
    
    
    result = stk.tej_ewprcd(begin_date=shift_begin, end_date=none,
                            symbol=symbol, trade=false, adj=false,
                            price=false, outstanding=false, pe_ratio=True)
    
    result = result.merge(symbol_df, on='symbol')
    
    global debug_pe_ratio
    debug_pe_ratio = result.copy()    
    
    
    # - all pe_ratio of 6598 are null.
    # - the 10th quantile of pe_ratio is zero, so fill na with 0.
    result = cbyz.df_conv_na(df=result, cols='pe_ratio', value=0)
    
    cbyz.df_chk_col_na(df=result, mode='stop')
    cols = cbyz.df_get_cols_except(df=result,
                                   except_cols=['symbol', 'work_date'])
    
    
    # change data type
    # - the original data types are object, and they can't be converted 
    #   before drop null values.
    result = cbyz.df_conv_col_type(df=result, cols=cols, to='float')    
    
    
    # scale data
    result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['symbol'], date_col='work_date',
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
        
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='work_date')
            
        result = cbyz.df_conv_na(df=result, cols=ma_cols)  
        
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=id_keys, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=false, add_count=false, quantile=[]
            )    
    
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='symbol')


    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    

    return result, ma_cols   


# .............


def sam_tej_ewtinst1_hold():
    
    global id_keys, corr_threshold
    global symbol
    global predict_date
    global shift_begin, data_end
    
    
    hold_cols = ['qfii_ex1_hold', 'fund_ex_hold', 'dlr_ex_hold']
    
    
    # stock outstanding / shares outstanding
    outstanding = stk.tej_ewprcd(begin_date=shift_begin, end_date=data_end,
                                 symbol=symbol, trade=false, adj=false,
                                 price=false, outstanding=True)
    
    # trans details
    result = stk.tej_ewtinst1_hold(end_date=predict_date.loc[0, 'work_date'],
                                   symbol=symbol, dev=false)
    
    
    # calculate spreadholding ratio
    result = result.merge(outstanding, how='left', on=['symbol', 'work_date'])
    result = cbyz.df_fillna_chain(df=result, cols='outstanding_shares',
                                  sort_keys=['symbol', 'work_date'],
                                  method=['ffill', 'bfill'], 
                                  group_by='symbol')
    for c in hold_cols:
        result[c + '_ratio'] = result[c] / result['outstanding_shares']
    
    result = result.drop('outstanding_shares', axis=1)


    # get columns list
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['symbol', 'work_date']
        )
    
    
    # scale data
    result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['symbol'], date_col='work_date',
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
        
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='work_date')
            
        result = cbyz.df_conv_na(df=result, cols=ma_cols)  
        
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=id_keys, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=false, add_count=false, quantile=[]
            )    
    
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='symbol')


    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols) 

    return result, ma_cols
    


# def sam_tej_ewtinst1_hold_20220408():
    
#     global id_keys, corr_threshold
#     global symbol
#     global predict_date
    
#     result = stk.tej_ewtinst1_hold(end_date=predict_date.loc[0, 'work_date'],
#                                    symbol=symbol, dev=false)
    
#     cols = cbyz.df_get_cols_except(
#         df=result, 
#         except_cols=['symbol', 'work_date']
#         )
    
#     # scale data
#     result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
#     # ma
#     result, ma_cols = \
#         cbyz.df_add_ma(df=result, cols=cols,
#                        group_by=['symbol'], date_col='work_date',
#                        values=ma_values, wma=wma, 
#                        show_progress=false
#                        )   
        
#     result = result.drop(cols, axis=1)
    
    
#     if time_unit == 'w':
#         result = result \
#             .merge(calendar_full_key, how='left', on='work_date')
            
#         result = cbyz.df_conv_na(df=result, cols=ma_cols)  
        
#         result, ma_cols = cbyz.df_summary(
#             df=result, cols=ma_cols, group_by=id_keys, 
#             add_mean=True, add_min=True, 
#             add_max=True, add_median=True, add_std=True, 
#             add_skew=false, add_count=false, quantile=[]
#             )    
    
#     result = main_data_frame.merge(result, how='left', on=id_keys)
#     result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
#                                   sort_keys=time_key,
#                                   method=['ffill', 'bfill'], 
#                                   group_by='symbol')


#     # drop highly correlated features
#     result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
#                                         except_cols=id_keys) 

#     # filter existing columns
#     ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols) 

#     return result, ma_cols





# .............


def sam_tej_ewtinst1c():
    
    global id_keys, corr_threshold
    global shift_begin
    
    result = stk.tej_ewtinst1c(begin_date=shift_begin, end_date=none, 
                                   symbol=symbol, trade=True)
    
    cols = cbyz.df_get_cols_except(
        df=result, 
        except_cols=['symbol', 'work_date']
        )
    
    # scale data
    result, _, _ = cbml.df_scaler(df=result, cols=cols, method=0)
    
    # ma
    result, ma_cols = \
        cbyz.df_add_ma(df=result, cols=cols,
                       group_by=['symbol'], date_col='work_date',
                       values=ma_values, wma=wma, 
                       show_progress=false
                       )   
        
    result = result.drop(cols, axis=1)
    
    
    if time_unit == 'w':
        result = result \
            .merge(calendar_full_key, how='left', on='work_date')
            
        result = cbyz.df_conv_na(df=result, cols=ma_cols)  
        
        result, ma_cols = cbyz.df_summary(
            df=result, cols=ma_cols, group_by=id_keys, 
            add_mean=True, add_min=True, 
            add_max=True, add_median=True, add_std=True, 
            add_skew=false, add_count=false, quantile=[]
            )    
    
    result = main_data_frame.merge(result, how='left', on=id_keys)
    result = cbyz.df_fillna_chain(df=result, cols=ma_cols,
                                  sort_keys=time_key,
                                  method=['ffill', 'bfill'], 
                                  group_by='symbol')


    # drop highly correlated features
    result = cbml.df_drop_high_corr_var(df=result, threshold=corr_threshold, 
                                        except_cols=id_keys) 

    # filter existing columns
    ma_cols = cbyz.df_filter_exist_cols(df=result, cols=ma_cols)    

    return result, ma_cols    



# %% process ------


def get_model_data(industry=True, trade_value=True, load_file=false):
    
    
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


        # ensure files were saved recently
        if main_data_diff <= 2 and  model_x_diff <= 2 and norm_diff <= 2:
            try:
                model_data = pd.read_csv(main_data_file)
                model_x = cbyz.li_read_csv(model_x_file)
                scale_orig = pd.read_csv(scale_orig_file)
                y_scaler = pickle.load(open(y_scaler_file, 'rb'))
                
            except exception as e:
                print('get_model_data - fail to load files.')                
                print(e)
                
            else:
                print('get_model_data - load files successfully.')
                return model_data, model_x, scale_orig                
    

    # check ......
    msg = ('get_model_data - predict_period is longer than ma values, '
            'and it will cause na.')
    assert predict_period <= min(ma_values), msg


    # symbols ......
    symbol = cbyz.conv_to_list(symbol)
    symbol = cbyz.li_conv_ele_type(symbol, 'str')


    # market data ......
    # market_data
    get_market_data_raw(trade_value=trade_value)
    gc.collect()
    
    
    # load historical data ......
    global main_data_raw
    main_data_raw, scale_orig = \
        sam_load_data(industry=industry, trade_value=trade_value) 
        
    main_data = main_data_raw.copy()
    cbyz.df_chk_col_na(df=main_data_raw)
    
    
    # merge calendar_proc
    main_data = main_data.merge(calendar_proc, how='left', on=time_key)
    
    
    # todc shareholdings spread ......
    if market == 'tw' and shift_begin >= 20210625 :
        
        print('sam_tdcc_get_sharehold_spread')
        sharehold, cols = sam_tdcc_get_sharehold_spread()
        main_data = main_data.merge(sharehold, how='left', on=id_keys)
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=['symbol'])   
        del sharehold
        gc.collect()


    # # tej ewprcd pe ratio ......
    if market == 'tw':
        pe_ratio, cols = sam_tej_ewprcd_pe_ratio()
        main_data = main_data.merge(pe_ratio, how='left', on=id_keys)
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=['symbol'])   
        del pe_ratio
        gc.collect()
    
    
    
    # monthly revenue ......
    # 1. 當predict_date=20211101，且為dev時, 造成每一個symbol都有na，先移除
    # 2. 主要邏輯就是顯示最新的營收資料
    if market == 'tw':
        
        msg = '''bug - sam_tej_ewsale，在1/18 23:00跑1/19時會出現chk_na error，
        但1/19 00:00過後再跑就正常了
        '''
        print(msg)
        
        ewsale, cols = sam_tej_ewsale(begin_date=shift_begin)
        main_data = main_data.merge(ewsale, how='left', on=id_keys)      
        
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=['symbol'])           
    
        del ewsale
        gc.collect()


    # # tej ewtinst1 - transaction details of juridical persons ......
    # if market == 'tw':
    #     ewtinst1, cols = sam_tej_ewtinst1()
    #     main_data = main_data.merge(ewtinst1, how='left', on=id_keys)
    #     main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
    #                                       sort_keys=time_key, 
    #                                       method=['ffill', 'bfill'], 
    #                                       group_by=['symbol'])   
    #     del ewtinst1
    #     gc.collect()


    # # tej ewgin ......
    # if market == 'tw':
    #     ewgin, cols = sam_tej_ewgin()
    #     main_data = main_data.merge(ewgin, how='left', on=id_keys)
    #     main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
    #                                       sort_keys=time_key, 
    #                                       method=['ffill', 'bfill'], 
    #                                       group_by=['symbol'])          
    #     del ewgin
    #     gc.collect()


    # # ex-dividend and ex-right ...
    # if market == 'tw':
    #     data, cols = sam_ex_dividend()
    #     main_data = main_data.merge(data, how='left', on=id_keys)
    #     main_data = cbyz.df_conv_na(df=main_data, cols=cols, value=0)


    # buffett indicator ......
    if market == 'tw':
        
        buffett_indicator, cols = sam_buffett_indicator()
        
        # 因為部份欄位和下面的tw_index重複，所以刪除
        drop_cols = cbyz.df_get_cols_contains(df=buffett_indicator, 
                                              string=['tw_index'])
        
        buffett_indicator = buffett_indicator.drop(drop_cols, axis=1)
        cols = cbyz.li_remove_items(cols, drop_cols)
        
        # merge
        main_data = main_data \
                    .merge(buffett_indicator, how='left', on=time_key)

        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=[])


    # government invest ......
    if market == 'tw':
        # gov_invest = sam_tw_gov_invest(dev=True)
        gov_invest = sam_tw_gov_invest(dev=false)
        cols = cbyz.df_get_cols_except(df=gov_invest, except_cols=id_keys)
        
        # 沒有交集時就不merge，避免一整欄都是na
        if len(gov_invest) > 0:
            main_data = main_data.merge(gov_invest, how='left', on=id_keys)
            main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                        sort_keys=id_keys, group_by=['symbol'], 
                                        method='ffill')
            
            # 避免開頭的資料是na，所以再用一次bfill
            main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                        sort_keys=id_keys, group_by=['symbol'], 
                                        method='bfill')                        
    
            main_data = cbyz.df_conv_na(df=main_data, cols=cols, value=0)
    
    
    # government own ......
    if market == 'tw':
        # gov_own = sam_tw_gov_own(dev=True)
        gov_own = sam_tw_gov_own(dev=false)
        cols = cbyz.df_get_cols_except(df=gov_own, except_cols=id_keys)
        
        # 沒有交集時就不merge，避免一整欄都是na
        if len(gov_own) > 0:
            main_data = main_data.merge(gov_own, how='left', on=id_keys)
            main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                        sort_keys=id_keys, group_by=['symbol'], 
                                        method='ffill')
            
            # 避免開頭的資料是na，所以再用一次bfill
            main_data = cbyz.df_fillna(df=main_data, cols=cols, 
                                        sort_keys=id_keys, group_by=['symbol'], 
                                        method='bfill')    

            main_data = cbyz.df_conv_na(df=main_data, cols=cols, value=0)                   


    # ^twii .......
    if market == 'tw':
         
        tw_index, cols = \
            sam_od_tw_get_index(
                begin_date=shift_begin,
                end_date=predict_date.loc[len(predict_date)-1, 'work_date']
                )
        
        main_data = main_data.merge(tw_index, how='left', on=time_key)
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=[])    

    
    # fiat currency exchange ......
    fx_rate, cols = sam_od_tw_get_fx_rate()
    main_data = main_data.merge(fx_rate, how='left', on=time_key)

    main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                      sort_keys=time_key, 
                                      method=['ffill', 'bfill'], 
                                      group_by=[])


    # get dow jones industrial average (^dji) ......
    dji, cols = sam_od_us_get_dji()
    main_data = main_data.merge(dji, how='left', on=time_key)
    del dji
    gc.collect()   


    # s&p 500 ......
    snp, cols = sam_od_us_get_snp(begin_date=shift_begin)
    main_data = main_data.merge(snp, how='left', on=time_key)
    del snp
    gc.collect()
    
    
    # covid-19 ......
    if market == 'tw':
        covid_tw, cols = sam_covid_19_tw()
        
        # future plan
        # sam_covid_19_global()
        
        main_data = main_data.merge(covid_tw, how='left', on=time_key)
        main_data = cbyz.df_conv_na(df=main_data, cols=cols)
        
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=[])
    elif market == 'en':
        # future plan
        # covid_en = sam_covid_19_global()        
        pass


    
    # financial statement
    if market == 'tw':
        print('目前只用單季，需確認是否有缺漏')
        # 20220218 - dev=True，eta=0.2時，即使只保留一個欄位也會overfitting
        financial_statement, cols = sam_tej_ewifinq()
        
        main_data = main_data \
                .merge(financial_statement, how='left', on=id_keys)
                
        main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
                                          sort_keys=time_key, 
                                          method=['ffill', 'bfill'], 
                                          group_by=[])


    # tej ewtinst1c - average holding cost of juridical persons ......
    # if market == 'tw':
    #     ewtinst1c, cols = sam_tej_ewtinst1c()
    #     main_data = main_data.merge(ewtinst1c, how='left', on=id_keys)
    #     main_data = cbyz.df_fillna_chain(df=main_data, cols=cols,
    #                                       sort_keys=time_key, 
    #                                       method=['ffill', 'bfill'], 
    #                                       group_by=['symbol'])          


    # model data ......


    # remove date before first_trading_day
    # - 由於main_data_frame是用cross_join，所以會出現listing前的日期，但這個步驟要等到
    #   最後才執行，否則在合併某些以月或季為單位的資料時會出現na
    print('是否可以移到sam_load_data最下面；暫時移除')
    # global first_trading_day
    # main_data = main_data \
    #     .merge(first_trading_day, how='left', on=['symbol'])
    
    # main_data = main_data[
    #     main_data['work_date']>=main_data['first_trading_day']] \
    #     .drop('first_trading_day', axis=1)


    # check na ......
    if time_unit == 'd':
        hist_df = main_data[
            main_data['work_date']<predict_date['work_date'].min()]
        
    elif time_unit == 'w':
        hist_df = cbyz.df_anti_merge(main_data, predict_week, 
                                     on=time_key)
    
    
    # debug ......
    msg = ("get_model_data - 把normalize的group_by拿掉後，這個地方會出錯，但"
           "只有一筆資料有問題，暫時直接drop")
    print(msg)
    
    
    # - 當symbols=[]時，這裡會有18筆na，都是var_y的欄位，應該是新股，因此直接排除
    # - 當time_unit為w，predict_begin為20220104時，會有275筆na，但都是5244這一
    #   檔，且都是ewtinst1c的欄位，應該也是新股的問題，直接排除
    # - 新股的ewtinst1c一定會有na，但snp不會是na，導致na_min和na_max不一定相等
    # - 如果只是某幾天缺資料的話，chk_na_min和chk_na_max應該不會相等
    chk_na = cbyz.df_chk_col_na(df=hist_df, except_cols=var_y)

    assert_cond = false
    if (isinstance(chk_na, pd.dataframe) and len(chk_na) < 600) \
        or (chk_na == none):
        assert_cond = True
        
    if not assert_cond:
        chk_na.to_csv(path_temp + '/chk_na_id_01.csv', index=false)

    assert assert_cond, \
        'get_model_data - hist_df has ' + str(len(chk_na)) + ' na'
    
    if isinstance(chk_na, pd.dataframe):
        na_cols = chk_na['column'].tolist()
        main_data = main_data.dropna(subset=na_cols, axis=0)
    
    
    # predict有na是正常的，但na_count必須全部都一樣
    global chk_predict_na, predict_df
    if time_unit == 'd':
        predict_df = main_data.merge(predict_date, on=time_key)
        
    elif time_unit == 'w':
        predict_df = main_data.merge(predict_week, on=time_key)
    
    chk_predict_na = cbyz.df_chk_col_na(df=predict_df, mode='alert')
    
    if isinstance(chk_predict_na, pd.dataframe):
        min_value = chk_predict_na['na_count'].min()
        max_value = chk_predict_na['na_count'].max()    
    
        # chk_predict_na == none, chk_predict_na應該不可能是none？
        assert min_value == max_value, \
            'all the na_count should be the same.'
    

    # check min max ......
    # global chk_min_max
    # chk_min_max = cbyz.df_chk_col_min_max(df=main_data)
    
    # chk_min_max = \
    #     chk_min_max[(~chk_min_max['column'].isin(id_keys)) \
    #                 & ((chk_min_max['min_value']<0) \
    #                    | (chk_min_max['max_value']>1))]
    
    # assert len(chk_min_max) == 0, 'get_model_data - normalize error'

    
    assert  'trade_date' not in main_data.columns, \
        'bug - trade_date should not exsits'

    print('1732- 201750有兩筆資料')
    chk_dup = main_data \
            .copy() \
            .groupby(['symbol', 'year_week_iso']) \
            .size() \
            .reset_index(name='count')
    
    chk_dup = chk_dup[chk_dup['count']>1]
    assert len(chk_dup) < 20, 'chk_dup error'
    
    main_data = main_data \
                .drop_duplicates(subset=['symbol', 'year_week_iso']) \
                .reset_index()

        
    cbyz.df_chk_duplicates(df=main_data, cols=id_keys)


    # select features ......
    # - select featuers after completing data cleaning
    
    # drop highly correlated features
    main_data = cbml.df_drop_high_corr_var(df=main_data, 
                                            threshold=corr_threshold, 
                                            except_cols=id_keys + var_y) 
    # select best features
    print('disable selectkbest to test random forest')
    # best_var_raw, best_var_score, best_var = \
    #     cbml.selectkbest(df=main_data, model_type='reg', 
    #                       y=var_y, x=[], except_cols=id_keys, k=60)
        
    # main_data = main_data[id_keys + var_y + best_var] 


    # variables ......
    model_x = cbyz.df_get_cols_except(df=main_data, 
                                      except_cols=var_y + id_keys)


    # export model ......
    main_data.to_csv(main_data_file, index=false)
    cbyz.li_to_csv(model_x, model_x_file)
    scale_orig.to_csv(scale_orig_file, index=false)
        
    return main_data, model_x, scale_orig
    


# %% master ------


def master(param_holder, predict_begin, export_model=True,
           threshold=30000, bt_index=0, load_data=false):
    '''
    主工作區
    '''
    
    # v3.0000 - 20220225
    # - 開發重心轉移至trading bot
    # - update for ultra_tuner v0.3100
    # v3.0100 - 20220305
    # - drop correlated columns in variable function, or it will cause 
    #   expensive to execute this in the ultra_tuner
    # - give the common serial when predict for each y
    # - add mlp
    # v3.0200
    # - update for cbml.selectkbest
    # - update for ut_v1.0001, and add epochs to model_params
    # v3.0300
    # - add tej_ewgin
    # - add tej_ewtinst1
    # - add od_us_get_dji
    # - completed test
    # v3.0400
    # - stable
    # v3.0500
    # - rename tej function in stk
    # - add tej_ewtins1 _hold
    # - try random forest, refer to 〈dtsa 5509 week 5 - ensemble methods.〉
    #   random forest may fit better than xgboost if dataset contains 
    #   many features.
    # v3.0600
    # - update stk
    # v3.0601
    # - add tej_ewtins1_hold ratio, getting data from ewprcd
    # - fix bug of snp and dji
    # v3.0700 - 20220418
    # - remove tej_ewtinst1_hold - done
    # - add pe_ratio from ewprcd - done
    # v3.0701 - 20220422
    # - update cbml and test wma
    # v3.0800 - 20220422
    # - replace time_key from ['year_iso', 'week_num_iso'] to ['year_week_iso'],
    #   then can be apply df_add_ma    
    # - time unit是week時，原本是先ma再summary，改成先summary再ma
    # - fix bug - 目前ma時的單位是d，parameter is week，但兩者共用相同的ma_values
    # v3.0801 - 20220513
    # - fix bug for stk.tej_ewtinst1    
    # - fix sam_tej_ewifinq
    # v3.0802 - 20220516
    # - fix sam_tej_ewsale
    # v3.0803 - 20220518
    # - upadate inverse_transform for y
    # v3.0804 - 20220518
    # - upadate y_scaler
    # v3.0803 - 20220517
    # - fix model_data duplicated issues
    # v3.0804 - 20220520
    # - fix scaler
    # - fix model_data duplicated issues
    # v3.0805 - 20220524
    # - update for cbml.df_scaler_inverse_v2
    # v3.0806 - 20220526
    # - update for cbml and ut
    # v3.0807 - 20220526
    # convert columns to lowercase

    # v3.09
    # - change number of host

    
    # v3.080x
    # - add vif
    # - 是不是要註冊第二個tej帳號
    # - visualize the tej_ewtinst1_hold and stock price
    # - 把加了financial statement的model data丟到automl
    # - cpi指數
    
    
    # - add shareholding spread, then train by automl to see the importance
    # - 如果把ewtinst1和amtop1混著用，可能會有一個問題是在外資交易所交易的不一定是真外資
        
    # - update scale function修改df_scaler的時候，發現確實需要scaled by symbol，	
    #   否則全部資料全部丟進去scaled，normaltest的p value目前是0    	
    # https://machinelearningmastery.com/how-to-transform-data-to-fit-the-normal-distribution/    


    # - 是不是可以跟tej買董監事持股的資料，然後減掉shareholding spread
    # - 董監事持股open data
    #   https://data.gov.tw/dataset/22811
    # - 大戶持股比例是否持續下降
    #   https://meet.bnext.com.tw/blog/view/9641?
    
    # - snp與每個產業的correlation
    # - 景氣對策信號
    # - 製造業循環
    #   https://www.macromicro.me/collections/3261/sector-industrial/25709/manufacturing-cycle
    # - 量價分析；volume / mean price
    #   > 量價曲線（volume price trend，vpt）
    #   > https://money.udn.com/money/story/12040/5399621
    #   > https://wealth.businessweekly.com.tw/m/garticle.aspx?id=artl00300599
    #   > https://www.wealth.com.tw/articles/81b51a4e-6549-4f6e-a11d-eb80a9a88e91
    # - log transform for volume; add log transform in df_scaler
    # - test spreadholding data, and decide to buy it or not
    # - 加總三大法人總持股數
    # - 是不是應該修改loss function，不要用mse
    
    # - add transaction volume divide outstanding
    # - optimize, y是不是可以用log transform，底數要設多少？
    # - bug, ut score features log not sorted
    # - update, add training time to ut
    # - apply ma on financial statement to prevent overfitting
    # - 底扣價
    # - 隱含波動率implied volatility (iv); 歷史波動率 historical volatility (hv)
    #   > https://slashtraders.com/tw/blog/market-volatility-hv-iv/#hv
    #   > https://www.cw.com.tw/article/5101233
    #   > https://stackoverflow.com/questions/61289020/fast-implied-volatility-calculation-in-python
    # - new incident
    # - 可以從三大法人買賣明細自己推三大法人持股成本
    # - update for new arsenal_stock
    
    # - 如果你想要拉回買進，就要趁著「量縮價穩的箱型整理時期」，因此檢視公司的月
    #   營收與季報的趨勢很重要。因為法人與主力在箱型區間震盪整理時，做價做量容易與
    #   每月營收、每季財報或利多消息時間搭配完美，股價容易呈現波段式階梯上漲，表示
    #   著主力籌碼換手積極，在換手過程中進行低買高賣，有效降低持股成本。
    #   https://wealth.businessweekly.com.tw/m/garticle.aspx?id=artl003005990
    
    
    
    # df_expend_one_hot_signal


    # - add df_vif and fix bug
    # - c:\programdata\anaconda3\lib\site-packages\statsmodels\stats
    #   \outliers_influence.py:193: runtimewarning: divide by zero encountered in
    #   double_scalars
    #   vif = 1. / (1. - r_squared_i)


    global version
    version = 3.0807
    
    
    # bug
    # snp有na是合理的嗎？
    #                      column  na_count
    # 0         high_change_ratio       475
    # 1          low_change_ratio       475
    # 2        close_change_ratio       475
    # 3       qfii_hap_ma_36_mean       256
    # 4        qfii_hap_ma_36_min       256
    # ..                      ...       ...
    # 348    snp_volume_ma_1_mean       446
    # 349     snp_volume_ma_1_min       446
    # 350     snp_volume_ma_1_max       446
    # 351  snp_volume_ma_1_median       446
    # 352     snp_volume_ma_1_std       446
    

    
    # update
    # - bug - sam_tej_ewsale，在1/18 23:00跑1/19時會出現chk_na error，
    #   但1/19 00:00過後再跑就正常。end_date應該要改成data_begin, 這個問題應該
    #   是today比data_begin少一天    
    # - replace symbol with target, and add target_type which may be symbol
    #   or industry or 大盤
    # - add financial_statement
    #   > 2021下半年還沒更新，需要改code，可以自動化更新並合併csv
    # - 財報的更新時間很慢，20220216的時候，2021年12月的財報還沒出來；所以
    #   在tej_ewtinst1c需要有提醒，列出最新的財報日期，並用assert檢查中間
    #   是否有空缺
    # - fix support and resistant
    # - select_symbols用過去一周的總成交量檢查
    # 以close price normalize，評估高價股和低價股
    # - handle week_num cyclical issues
    #   https://towardsdatascience.com/how-to-handle-cyclical-data-in-machine-learning-3e0336f7f97c
    
    
    # new dataset ......
    # - 美元利率
    # - 台幣利率
    # https://www.cbc.gov.tw/tw/np-1166-1.html
    # - 國際油價
    # - 黃金價格
    # - 確認現金流入率 / 流出率的算法，是否要買賣金額，還是只要有交易額就可以了
    # - 三大法人交易明細
    #   >> 如果買tej的達人方案，那也不需要額外加購三大法人持股成本
    #   >> 增加三大法人交易明細後，試著計算三大法人每個交易所的平均持有天數，
    #      再試著將平均持有天數做分群，就可以算出每一個交易所的交易風格和產業策略。
    #   >> 增加三大法人交易明細後，從20170101，累積計算後，可以知道每一個法人
    #      手上的持股狀況；
    # - add 流通股數 from ewprcd
    # - add 法說會日期
    # - add 道瓊指數
    

    # optimization ......
    # - 如何把法說會的日期往前推n天，應該interpolate
    # - combine cbyz >> detect_cycle.py, including support_resistance and 
    #    season_decompose
    # - fix week_num, because it may be useful to present seasonaility
    # - 合併yahoo finance和tej的market data，兩邊都有可能缺資料。現在的方法是
    #   用interpolate，
    #   但如果begin_date剛好缺值，這檔股票就會被排除
    # - 把symbol改成target，且多一個target_type，值可以是symbol或industry
    # - 技術分析型態學
    # - buy_signal to int
    #   update - 20211220，暫時在get_tw_index中呼叫update_tw_index，但這個方式
    #   可能會漏資料    
    
    
    global bt_last_begin, data_period, predict_period, long, time_unit
    global dev, test
    global symbol, volume_thld, market, data_form, load_model_data
    global ma_values, wma
    

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
    wma = holder['wma'][0]
    ma_values = holder['ma_values'][0]   
    data_form = holder['data_form'][0]   
    time_unit = holder['time_unit'][0]
    load_model_data = holder['load_model_data'][0]
    
    # modeling
    predict_period = holder['predict_period'][0]
    long = holder['long'][0]   
    kbest = holder['kbest'][0]
    cv = holder['cv'][0]
    
    # program
    dev = holder['dev'][0]
    test = holder['test'][0]
    
    # export_model = false
    # dev = True
    # threshold = 20000
    # predict_begin=20211209
    
    
    global exe_serial
    exe_serial = cbyz.get_time_serial(with_time=True, remove_year_head=True)

    global log, scale_log, error_msg
    global ohlc, ohlc_ratio, ohlc_change
    log = []
    scale_log = pd.dataframe()
    error_msg = []    
    ohlc = stk.get_ohlc()
    ohlc_ratio = stk.get_ohlc(orig=false, ratio=True, change=false)
    ohlc_change = stk.get_ohlc(orig=false, ratio=false, change=True)
    
    # keys ------
    global id_keys, time_key
    global var_y, var_y_orig
    
    if time_unit == 'w':
        id_keys = ['symbol', 'year_week_iso']
        time_key = ['year_week_iso']        
        
    elif time_unit == 'd':
        id_keys = ['symbol', 'work_date']    
        time_key = ['work_date']

    
    # var_y = ['high', 'low', 'close']
    var_y = ['high_change_ratio', 'low_change_ratio', 'close_change_ratio']
    var_y_orig = [y + '_orig' for y in var_y]    
    
    
    # update, add to btm
    global corr_threshold

    # 原本設定為0.85，但cu dtsa 5509將collinearity的標準設為0.7
    corr_threshold = 0.7
    # corr_threshold = 0.85
    
    
    global df_summary_mean, df_summary_min, df_summary_max
    global df_summary_median, df_summary_std
    
    
    print('some columns should be aggregated by sum, like volume')
    df_summary_mean = True
    df_summary_min = false
    df_summary_max = false
    df_summary_median = false
    df_summary_std = True
    
    
    # calendar ------
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
                

    # df may have na if week_align is True
    if time_unit == 'w':
        calendar = calendar.dropna(subset=time_key, axis=0)
        
    # 有些資料可能包含非交易日，像是covid-19，所以需要一個額外的calendar作比對
    calendar_full_key = calendar[['work_date', 'year_week_iso']]
                
                
    # ......
    global model_data, model_x, scale_orig
    model_data, model_x, scale_orig = \
        get_model_data(industry=industry, trade_value=trade_value,
                       load_file=load_model_data)
    
    
    # training model ......
    import xgboost as xgb
    from sklearn.linear_model import linearregression
    from sklearn.ensemble import randomforestregressor
    import tensorflow as tf

    
    if len(symbol) > 0 and len(symbol) < 10:
        model_params = [{'model': linearregression(),
                         'params': {
                             'normalize': [True, false],
                             }
                         }]         
    else:
        
        # mlp
        # - prevent error on host 4
        if host in [2, 3]:
            vars_len = cbyz.df_get_cols_except(df=model_data, 
                                               except_cols=id_keys + var_y)
            vars_len = len(vars_len)
            
            mlp_model = tf.keras.sequential()
            mlp_model.add(tf.keras.layers.dense(30, input_dim=vars_len, 
                                                activation='relu'))
            
            mlp_model.add(tf.keras.layers.dense(30, activation='softmax'))
            mlp_model.add(tf.keras.layers.dense(1, activation='linear'))  
            
            mlp_param = {'model': mlp_model,
                         'params': {'optimizer':'adam',
                                    'epochs':15}}
        
        
        # eta 0.01、0.03的效果都很差，目前測試0.08和0.1的效果較佳
        # data_form1 - change ratio
        if time_unit == 'd' and 'close_change_ratio' in var_y:
            
            model_params = [{'model': linearregression(),
                              'params': {
                                  'normalize': [True, false],
                                  }
                              },
                            {'model': xgb.xgbregressor(),
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
            
        elif time_unit == 'w' and 'close_change_ratio' in var_y:
            
            # eta 0.1 / 0.2
            model_params = [
                            {'model': randomforestregressor(),
                               'params': {
                                   'max_depth': [6],
                                   'n_estimators': [100]
                                 }                     
                             },                     
                            {'model': linearregression(),
                              'params': {
                                  'normalize': [false],
                                  }
                              },
                            {'model': xgb.xgbregressor(),
                              'params': {
                                'n_estimators': [200],
                                'eta': [0.2, 0.25],
                                # 'eta': [0.2, 0.3],
                                'min_child_weight': [1],
                                  # 'min_child_weight': [0.5, 1],
                                  'max_depth':[8],
                                  # 'max_depth':[8, 10],
                                  # 'subsample':[0.7, 1]
                              }
                            }
                            ]        
            
        # prevent error on host 4
        if host in [2, 3]:
            model_params.insert(0, mlp_param)
            
     
        
    # - 如果selectkbest的k設得太小時，importance最高的可能都是industry，導致同
    #   產業的預測值完全相同
    global pred_result, pred_scores, pred_params, pred_features
    global y_scaler
    
    long_suffix = 'long' if long else 'short'
    compete_mode = compete_mode if bt_index == 0 else 0
    
    
    for i in range(len(var_y)):
        
        cur_y = var_y[i]
        remove_y = [var_y[j] for j in range(len(var_y)) if j != i]
        cur_scaler = y_scaler[cur_y]
        
        tuner = ut.ultra_tuner(id_keys=id_keys, y=cur_y, 
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

        # prvent memory insufficient for saved data in ut
        del tuner
        gc.collect()


    # inverse scale ......
    pred_result = pred_result[id_keys + var_y]
    pred_result_inverse = cbml.df_scaler_inverse_v2(df=pred_result,
                                                    scaler=y_scaler)
        
    # upload to google sheet
    if predict_begin == bt_last_begin:
        stk.write_sheet(data=pred_features, sheet='features')

    
    return pred_result_inverse, pred_scores, pred_params, pred_features




def update_history():
    
    # v0.0 - mess version
    # v0.1 - fix mess version
    # v0.2 - 1. add calendar    
    # - support days True
    # v0.3 
    # - fix induxtry
    # - add daily backup for stock info
    # - update resistance and support function in stk
    # v0.4
    # - fix trade value
    # - add tej data
    # - precision stable version
    # v0.5
    # - add symbol vars
    # v0.6
    # - add todc shareholding spread data, but temporaily commented
    # - add tej 指數日行情
    # v0.7
    # - add google trends
    # - disable pytrends and ewiprcd
    # v0.8
    # - optimize data processing
    # v0.9
    # - add industry in excel
    # v1.0
    # - add ml_data_process
    # v1.0.1
    # - add auto model tuning
    # v1.03
    # - merge predict and tuning function from uber eats order forecast
    # - add small capitall back
    # - add tej market data function, multiply 1000
    # v1.04
    # - add price change for ohlc
    # - fix industry bug
    # v1.05
    # - update for new df_normalize
    # - test price_change_ratio as y
    # - add check for min max
    # v1.06
    # - change local variable as host
    # v1.07
    # - update for cbyz and cbml
    # - add low_volume_symbols
    # v1.08
    # - add 每月投報率 data    
    # v2.00
    # - add ultra_tuner
    # - fix ex-dividend issues
    # - rename stock_type and stock_symbol
    # - set df_fillna with interpolate method in arsenal_stock
    # v2.01
    # - fix date issues
    # v2.02
    # - add correlations detection > done
    # - fix tw_get_stock_info_twse issues
    # v2.03
    # - add s&p 500 data
    # v2.04
    # - add 台股指數
    # - fix terrible date lag issues in get_model_data - done
    # v2.041
    # - update for new modules
    # - export model data
    # v2.05
    # - add financial statements
    # v2.07
    # - remove group_by paramaters when normalizing
    # - fix bug after removing group_by params of normalizing
    # - add gdp and buffett indicator
    # v2.09
    # - update the calculation method of trade value as mean of high and low > done
    # - add load data feature for get_model_data > done
    # - update cbml and df_scaler > done
    # - remove df_chk_col_min_max > done
    # v2.10
    # - rename symbols as symbol
    # - add symbol params to ewifinq
    # - update cbml for df_scaler
    # v2.11 - 20220119
    # - the mvp version of data_for = 2
    # - y can be price or change ratio
    # v2.112 - 20220123
    # - mvp of weekly prediction
    # v2.2 - 20220209
    # - restore variables for weekly prediction
    # - change the way to shift day. the original method is shift var_x, and 
    #   the new version is to shift var_y and id_keys
    # - remove ml_data_process
    # - remove switch of trade_value
    # v2.3 - 20220210
    # - combine result of daily prediction and weekly prediction in btm
    # - add time_unit as suffix for saved_file of model_data
    # - industry_one_hot 不用df_summary    
    # - modify dev mode and test mode
    # v2.400 - 20220214
    # - collect fx_rate in dcm
    # - add fx_rate to pipeline
    # - remove open_change_ratio and open from var_y
    # v2.500 - 20220215
    # v2.501 - 20220216
    # - replace year with year_iso, and week with week_iso
    # v2.502 - 20220216
    # - restore sam_tej_ewifinq, but removed again
    # - set model params for week and day seperately
    # - move od_tw_get_ex_dividends to arsenal_stock
    # - optimize load_data feature of get_model_data 
    # v2.0600 - 20220221
    # - 當time_unit為w時，讓predict_begin可以不是星期一
    # - week_align為True時，get_model_data最下面的assert會出錯
    
    pass




# %% check ------


def check():
    
    pass

# %% manually analyze ------


def select_symbols_manually(data_begin, data_end):


    # select rules
    # 1. 先找百元以下的，才有資金可以買一整張
    # 2. 不要找疫情後才爆漲到歷史新高的

    global market
    data_end = cbyz.date_get_today()
    data_begin = cbyz.date_cal(data_end, -1, 'm')


    # stock info
    stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    
    
    # section 1. 資本額大小 ------
    
    # 1-1. 挑選中大型股 ......
    level3_symbol = stock_info[stock_info['capital_level']>=2]
    level3_symbol = level3_symbol['symbol'].tolist()
    
    data_raw = stk.get_data(data_begin=data_begin, data_end=data_end, 
                            symbol=level3_symbol, 
                            price_change=True,
                            shift=0, stock_type=market)
    
    data = data_raw[data_raw['symbol'].isin(level3_symbol)]
    

    # 1-2. 不排除 ......
    data = stk.get_data(data_begin=data_begin, data_end=data_end, 
                            symbol=[], 
                            price_change=True,
                            shift=0, stock_type=market)
    
    
    # section 2. 依價格篩選 ......
    
    # 2-1. 不篩選 .....
    target_symbols = data[['symbol']] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    
    
    # 2-2. 低價股全篩 .....
    # 目前排除80元以上
    last_date = data['work_date'].max()
    last_price = data[data['work_date']==last_date]
    last_price = last_price[last_price['close']>80]
    last_price = last_price[['symbol']].drop_duplicates()
    
    
    target_symbols = cbyz.df_anti_merge(data, last_price, on='symbol')
    target_symbols = target_symbols[['symbol']].drop_duplicates()
    
    
    # 2-3. 3天漲超過10%  .....
    data, cols_pre = cbyz.df_add_shift(df=data, 
                                       group_by=['symbol'], 
                                       cols=['close'], shift=3,
                                       remove_na=false)
    

    data['price_change_ratio'] = (data['close'] - data['close_pre']) \
                            / data['close_pre']
    
    
    results_raw = data[data['price_change_ratio']>=0.15]
    
    
    summary = results_raw \
                .groupby(['symbol']) \
                .size() \
                .reset_index(name='count')
                
    
    # select symboles ......
    target_symbols = results_raw.copy()
    target_symbols = cbyz.df_add_size(df=target_symbols,
                                      group_by='symbol',
                                      col_name='times')
        
    target_symbols = target_symbols \
                    .groupby(['symbol']) \
                    .agg({'close':'mean',
                          'times':'mean'}) \
                    .reset_index()
    
    target_symbols = target_symbols.merge(stock_info, how='left', 
                                          on='symbol')
    
    target_symbols = target_symbols \
                        .sort_values(by=['times', 'close'],
                                     ascending=[false, True]) \
                        .reset_index(drop=True)
                        
    target_symbols = target_symbols[target_symbols['close']<=100] \
                            .reset_index(drop=True)


    # export ......
    time_serial = cbyz.get_time_serial(with_time=True)
    target_symbols.to_csv(path_export + '/target_symbols_' \
                          + time_serial + '.csv',
                          index=false, encoding='utf-8-sig')

    # target_symbols.to_excel(path_export + '/target_symbols_' \
    #                         + time_serial + '.xlsx',
    #                         index=false)

    # plot ......       
    # plot_data = results.melt(id_keys='profit')

    # cbyz.plotly(df=plot_data, x='profit', y='value', groupby='variable', 
    #             title="", xaxes="", yaxes="", mode=1)

    
    return results_raw, stock_info



def check_price_limit():
    
    loc_stock_info = stk.tw_get_stock_info(daily_backup=True, path=path_temp)
    loc_stock_info = loc_stock_info[['symbol', 'capital_level']]
    
    
    loc_market = stk.get_data(data_begin=20190101, 
                        data_end=20210829, 
                        stock_type='tw', symbol=[], 
                        price_change=True, price_limit=True, 
                        trade_value=True)
    
    loc_main = loc_market.merge(loc_stock_info, how='left', 
                                on=['symbol'])

    # check limit up ......
    chk_limit = loc_main[~loc_main['capital_level'].isna()]
    chk_limit = chk_limit[chk_limit['limit_up']==1]

    chk_limit_summary = chk_limit \
            .groupby(['capital_level']) \
            .size() \
            .reset_index(name='count')


    # check volume ......
    #    over_1000  count
    # 0          0    115
    # 1          1    131    
    chk_volum = loc_main[loc_main['capital_level']==1]
    chk_volum = chk_volum \
                .groupby(['symbol']) \
                .agg({'volume':'min'}) \
                .reset_index()
                
    chk_volum['over_1000'] = np.where(chk_volum['volume']>=1000, 1, 0)
    chk_volum_summary = chk_volum \
                        .groupby(['over_1000']) \
                        .size() \
                        .reset_index(name='count')


# %% suspend ------


def get_google_treneds(begin_date=none, end_date=none, 
                       scale=True):
    
    global market
    print('get_google_treneds - 增加和get_stock_info一樣的daily backup')

    # begin_date = 20210101
    # end_date = 20210710
    
    
    # 因為google trends有假日的資料，所以應該要作平均
    # 全部lag 1
    # 存檔

    # 避免shift的時候造成na
    temp_begin = cbyz.date_cal(begin_date, -20, 'd')
    temp_end = cbyz.date_cal(end_date, 20, 'd')
    
    calendar = stk.get_market_calendar(begin_date=temp_begin, 
                                       end_date=temp_end, 
                                       stock_type=market)

    # word list ......
    file_path = '/users/aron/documents/github/data/stock_forecast/2_stock_analysis/resource/google_trends_industry.xlsx'
    file = pd.read_excel(file_path, sheet_name='words')
    
    file = file[(file['stock_type'].str.contains(market)) \
                & (~file['word'].isna()) \
                & (file['remove'].isna())]
    
    words_df = file.copy()
    words_df = words_df[['id', 'word']]
    words = words_df['word'].unique().tolist()
    
    
    # pytrends ......
    trends = cbyz.pytrends_multi(begin_date=begin_date, end_date=end_date, 
                                 words=words, chunk=180, unit='d', hl='zh-tw', 
                                 geo='tw')    
    
    trends = cbyz.df_date_simplify(df=trends, cols=['date']) \
                .rename(columns={'date':'work_date', 
                                 'variable':'word'})
    
    # merge data......
    main_data = calendar[['work_date', 'trade_date']]
    main_data = cbyz.df_cross_join(main_data, words_df)
    main_data = main_data.merge(trends, how='left', on=['work_date', 'word'])
    
    main_data['word_trend'] = 'word_' + main_data['id'].astype('str')
    main_data = cbyz.df_conv_na(df=main_data, cols='value')
    
    main_data = main_data \
                .sort_values(by=['word_trend', 'work_date']) \
                .reset_index(drop=True)
    
    
    # trade date ......
    # here are some na values
    main_data['next_trade_date'] = np.where(main_data['trade_date']==1,
                                           main_data['work_date'],
                                           np.nan)
    
    main_data = cbyz.df_shift_fillna(df=main_data, loop_times=len(calendar), 
                                     group_by='word_trend',
                                     cols='next_trade_date', forward=false)
    
    main_data = main_data \
                .groupby(['next_trade_date', 'word_trend']) \
                .agg({'value':'mean'}) \
                .reset_index() \
                .sort_values(by=['word_trend', 'next_trade_date'])

    # add lag. the main idea is that the news will be reflected on the stock 
    # price of tomorrow.                
    main_data, _ = cbyz.df_add_shift(df=main_data, cols='next_trade_date', 
                                    shift=1, group_by=[], suffix='_lag', 
                                    remove_na=false)
    
    main_data = main_data \
        .drop('next_trade_date', axis=1) \
        .rename(columns={'next_trade_date_lag':'work_date'}) \
        .dropna(subset=['work_date'], axis=0) \
        .reset_index(drop=True)
    
    
    cbyz.df_chk_col_na(df=main_data)
    
    
    # ......
    main_data, _, _ = cbml.df_scaler(df=main_data, cols='value')
    
    
    # pivot
    main_data = main_data.pivot_table(index='work_date',
                                    columns='word_trend', 
                                    values='value') \
                .reset_index()
                
                
                
    # calculate ma ......
    
    # bug
  # file "/users/aron/documents/github/codebase_yz/codebase_yz.py", line 2527, in df_add_ma
    # temp_main['variable'] = temp_main['variable'].astype(int)    
    
    global ma_values
    cols = cbyz.df_get_cols_except(df=main_data, except_cols=['work_date'])
    main_data, ma_cols = cbyz.df_add_ma(df=main_data, cols=cols, group_by=[], 
                                       date_col='work_date', values=ma_values,
                                       wma=false)
    
    # convert na
    cols = cbyz.df_get_cols_except(df=main_data, except_cols=['work_date'])
    main_data = cbyz.df_conv_na(df=main_data, cols=cols)
    
    
    return main_data, cols


# %% archive ------



# %% dev ------

    


def get_season(df):
    
    '''
    by week, 直接在get_data中處理好
    
    '''
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    
    loc_df = df.copy() \
            .rename(columns={'stock_symbol':'symbol'}) \
            .sort_values(by=['symbol', 'year_iso', 'week_num_iso']) \
            .reset_index(drop=True)
            
    # loc_df['x'] = loc_df['work_date'].apply(cbyz.ymd)

    unique_symbols = loc_df['symbol'].unique().tolist()
    result = pd.dataframe()

    for i in range(len(unique_symbols)):
        
        symbol = unique_symbols[i]
        temp_df = loc_df[loc_df['symbol']==symbol]
        model_data = temp_df[['close']]

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
                           threshold=0.9, plot_data=false):
    '''
    1. calculate suppport and resistance
    2. the prominence of each symbol is different, so it will cause problems
       if apply a static number. so use quantile as the divider.
    3. update, add multiprocessing
    

    parameters
    ----------
    df : type
        description.
    cols : type
        description.
    rank_thld : type, optional
        description. the default is 10.
    prominence : type, optional
        description. the default is 4.
    interval : boolean, optional
        add interval of peaks.
    threshold : type, optional
        description. the default is 0.9.
    plot_data : type, optional
        description. the default is false.

    returns
    -------
    result : type
        description.
    return_cols : type
        description.

    '''
    
    print('bug, 回傳必要的欄位，不要直接合併整個dataframe，減少記憶體用量')
    

    from scipy.signal import find_peaks
    cols = cbyz.conv_to_list(cols)

    cols_support = [c + '_support' for c in cols]
    cols_resistance = [c + '_resistance' for c in cols]
    return_cols = cols_support + cols_resistance
    
    
    group_key = ['symbol', 'column', 'type']
    
    
    # .......
    loc_df = df[['symbol', 'work_date'] + cols]
    
    date_index = loc_df[['symbol', 'work_date']]
    date_index = cbyz.df_add_rank(df=date_index, value='work_date',
                              group_by=['symbol'], 
                              sort_ascending=True, 
                              rank_ascending=True,
                              rank_name='index',
                              rank_method='min', inplace=false)
    
    result_raw = pd.dataframe()
    
    symbol_df = loc_df[['symbol']].drop_duplicates()
    symbol = symbol_df['symbol'].tolist()


    # frame ......
    begin_date = loc_df['work_date'].min()
    today = cbyz.date_get_today()

    calendar = cbyz.date_get_calendar(begin_date=begin_date,
                                      end_date=today)
    
    calendar = calendar[['work_date']]
    cols_df = pd.dataframe({'column':cols})
    
    frame = cbyz.df_cross_join(symbol_df, calendar)
    frame = cbyz.df_cross_join(frame, cols_df)

    
    # calculate ......
    for j in range(len(symbol)):
        
        symbol_cur = symbol[j]
        temp_df = loc_df[loc_df['symbol']==symbol_cur].reset_index(drop=True)
    
        for i in range(len(cols)):
            col = cols[i]
            x = temp_df[col].tolist()
            x = np.array(x) # 轉成np.array後，在計算低點的時候才可以直接加負值
            
            # 計算高點
            peaks_top, prop_top = find_peaks(x, prominence=prominence)
            new_top = pd.dataframe({'value':[i for i in prop_top['prominences']]})
            new_top.index = peaks_top
            
            threshold_value = new_top['value'].quantile(threshold)
            new_top = new_top[new_top['value']>=threshold_value]
            
            new_top['symbol'] = symbol_cur
            new_top['column'] = col
            new_top['type'] = 'resistance'
            
            
            # 計算低點
            # - 使用-x反轉，讓低點變成高點，所以quantile一樣threshold
            peaks_btm, prop_btm = find_peaks(-x, prominence=prominence)   
            new_btm = pd.dataframe({'value':[i for i in prop_btm['prominences']]})
            new_btm.index = peaks_btm
            
            threshold_value = new_btm['value'].quantile(threshold)
            new_btm = new_btm[new_btm['value']>=threshold_value]            
            
            new_btm['symbol'] = symbol_cur
            new_btm['column'] = col
            new_btm['type'] = 'support'
            
            # append
            result_raw = result_raw.append(new_top).append(new_btm)

            if j == 0:
                # keep the column names
                loop_times = len(temp_df)
        
        if j % 100 == 0:
            print('add_support_resistance - ' + str(j) + '/' \
                  + str(len(symbol)-1))
      
        
    result = result_raw \
            .reset_index() \
            .merge(date_index, how='left', on=['symbol', 'index']) \
            .drop('index', axis=1)
            
    result = result[['symbol', 'work_date', 'column', 'type']] \
            .sort_values(by=['symbol', 'column', 'work_date']) \
            .reset_index(drop=True)

    # dummy encoding
    result = cbml.df_get_dummies(df=result, cols='type', 
                                 expand_col_name=True,inplace=false)


    if plot_data:
        plot_close = loc_df[['symbol', 'work_date', 'close']] \
                        .rename(columns={'close':'value'})
        plot_close['type'] = 'close'
        plot_data_df = result.append(plot_close)
        plot_data_df = cbyz.df_ymd(df=plot_data_df, cols='work_date')
        
        # single_plot = plot_data_df[plot_data_df['symbol']==2456]
        # cbyz.plotly(df=plot_data_df, x='work_date', y='value', groupby='type')


    print('以下還沒改完')
    # .......
    if interval:
        
        interval_df, _ = \
            cbyz.df_add_shift(df=result, cols='work_date', shift=1, 
                              sort_by=['symbol', 'column', 'work_date'],
                              group_by=['symbol', 'column'], 
                              suffix='_prev', remove_na=false)
            
        # na will cause error when calculate date difference
        interval_df = interval_df.dropna(subset=['work_date_prev'], axis=0)
        
        interval_df = cbyz.df_conv_col_type(df=interval_df,
                                            cols=['work_date', 'work_date_prev'],
                                            to='int')
        
        interval_df = cbyz.df_date_diff(df=interval_df, col1='work_date', 
                                    col2='work_date_prev', name='interval', 
                                    absolute=True, inplace=false)
        
        interval_df = cbyz.df_conv_col_type(df=interval_df,
                                            cols=['interval'],
                                            to='int')
        
        interval_df = interval_df.drop('work_date_prev', axis=1)
        
        print(('does it make sense to use median? or quantile is better, '
              'then i can go ahead of the market'))
        interval_df = interval_df \
                    .groupby(['symbol', 'column']) \
                    .agg({'interval':'median'}) \
                    .reset_index()
            
        # merge
        result_full = frame \
            .merge(result, how='left', on=['symbol', 'work_date', 'column'])
        
        result_full = cbyz.df_conv_na(df=result_full, 
                                      cols=['type_resistance', 'type_support'],
                                      value=0)
        
        result_full = result_full \
            .merge(interval_df, how='left', on=['symbol', 'column'])
        
        
        result_full['resis_support_signal'] = \
            np.select([result_full['type_resistance']==1,
                       result_full['type_support']==1],
                      [result_full['interval'], -result_full['interval']],
                      default=np.nan)
         
        result_full['index'] = \
            np.where((result_full['type_resistance']==1) \
                      | (result_full['type_support']==1),
                      result_full.index, np.nan)
            
        result_full = \
            cbyz.df_fillna(df=result_full, 
                           cols=['resis_support_signal', 'index'], 
                           sort_keys=['symbol', 'column', 'work_date'], 
                           group_by=['symbol', 'column'],
                           method='ffill')
            
        result_full['resis_support_signal'] = \
            np.where((result_full['type_resistance']==1) \
                     | (result_full['type_support']==1),
                     result_full['resis_support_signal'],
                     result_full['resis_support_signal'] \
                         - (result_full.index - result_full['index']))
            
    else:
        pass
            

    # if unit == 'w':
    #     interval_df['interval'] = interval_df['interval'] / 7

        
        
    return result, return_cols




def test_support_resistance():
    

    data = stk.od_tw_get_index()
    
    # data = data.rename(columns={'tw_index_close':'close'})
    data['symbol'] = 1001
    
    
    add_support_resistance(df=data, cols='tw_index_close', 
                           rank_thld=10, prominence=4, 
                           days=True, threshold=0.9, plot_data=false)    



def check_ewtinst1():
    
    hold_cols = ['qfii_ex1_hold', 'fund_ex_hold', 'dlr_ex_hold']    
    
    
    result = stk.tej_ewtinst1(begin_date=20160101, end_date=none, 
                              symbol=[])
    
    
    unique_work_date = result[['work_date']].drop_duplicates()
    
    
    
    # stock outstanding / shares outstanding
    outstanding = stk.tej_ewprcd(begin_date=20170101, end_date=20220331,
                                 symbol=symbol, trade=false, adj=false,
                                 price=false, outstanding=True)
    
    # trans details
    # result = stk.tej_ewtinst1_hold(end_date=20220331,
    #                                symbol=[], dev=false)
    # result.to_csv(path_temp + '/ewtinst1_hold.csv', index=false)
    
    
    # calculate spreadholding ratio
    result = result.merge(outstanding, how='left', on=['symbol', 'work_date'])
    result = cbyz.df_fillna_chain(df=result, cols='outstanding_shares',
                                  sort_keys=['symbol', 'work_date'],
                                  method=['ffill', 'bfill'], 
                                  group_by='symbol')
    
    for c in hold_cols:
        result[c + '_ratio'] = result[c] / result['outstanding_shares']
    
    
    result = result.dropna(subset=['qfii_ex1_hold'], axis=0)
    result['work_date'].max()
    
    
    result['qfii_ex1_hold_ratio'].max()
    result['fund_ex_hold_ratio'].max()
    result['dlr_ex_hold_ratio'].max()
    
    
    result['qfii_ex1_hold_ratio'].mean()    
    result['fund_ex_hold_ratio'].mean()    
    result['dlr_ex_hold_ratio'].mean()    
    
    chk = result[result['symbol']=='2603']
    chk['qfii_ex1_hold_ratio'].max()
    
    
def dev():
    
    data = stk.get_data(data_begin=20210101,
                    data_end=20220331, 
                    market=market, 
                    symbol=symbol,
                    ratio_limit=True,
                    price_change=True, 
                    price_limit=True,
                    trade_value=True,
                    adj=True               
                    )
    
    data.loc[0:1000, 'test'] = True
    data = cbyz.df_conv_na(df=data, cols='test', value=false)
    
    data = cbml.df_vif(df=data, y=['close', 'high'],
                       # except_cols=['symbol', 'year_week_iso', 'work_date'],
                       except_cols=[],
                       limit=5, method='max')
    
    pass    
    


# %% debug ------

def debug():
    
    file = pd.read_csv(path_temp + '/model_data_w.csv')
    file.drop_duplicates()
    
    chk = file[file['symbol']==1101]
    chk = chk[chk['year_week_iso']==201737]    
    
    chk = chk.t    
    chk2 = chk[chk[0]!=chk[1]]
    
    

# %% execution ------


if __name__ == '__main__':
    
    symbol = [2520, 2605, 6116, 6191, 3481, 2409, 2603]
    
    # predict_result, precision = \
    #     master(param_holder=param_holder,
    #            predict_begin=20211209,
    #             _volume_thld=1000,
    #             fast=True)


    # arguments
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
    
    param_holder = ar.param_holder(**args)
        
    master(param_holder=param_holder,
           predict_begin=20211201, threshold=30000)        


