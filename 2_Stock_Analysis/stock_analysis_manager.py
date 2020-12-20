#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:08 2020

@author: Aron
"""



# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/2_Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar







# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 



def initialize(path):

    # 新增工作資料夾
    global path_resource, path_function, path_temp, path_export
    path_resource = path + '/Resource'
    path_function = path + '/Function'
    path_temp = path + '/Temp'
    path_export = path + '/Export'
    
    
    cbyz.create_folder(path=[path_resource, path_function, 
                             path_temp, path_export])        
    return ''








# 基本面
# 技術面
# 籌碼面：最難，但影響最大
# 消息面：Twitter


# > MIT資產管理課程，他舉了一個擺鐘的例子，當把很多本來有誤差的擺鐘放在一起，過一段時間後全部同會同步。
# > 當金融市場越來越有效率的時候，獲利的機會就越來越少
# > 大家越來越懂的時候，大家做的市場會越來越一樣
# 他們做加密貨幣，因為歷史太短了，大家都沒有充份的資料；基本面也有影響，但影響不像其他金融市場沒那麼大
# 因此，他認為要做加密貨幣的量化交易，現在是黃金時間，這個黃金時期可能還會持續一段時間。

# > 因為Crypto現在有很多交易所，且交易所的量大小都不一樣，所以當掛一樣的價錢的，量小的交易所
# 可能就買不到，整體而言的資料會更雜亂。
# Crypto相對傳統金融市場比較沒有效率，但因為沒有效率，所以才有機會。

# 量化交易的規模越大，難度也會越高，會有滑價等問題。100塊賺10塊，和100億賺10億的差別。
# 這可能也是我進場的機會。
# > 文藝復興最後也有這個問題，因為錢太多了，所以他們的行為本身就會影響市場。
# > 做交易的人，只要資料夠大，他就是用自己的籌碼在影響市場。所以過了某條線之後，有錢人越來越有錢。


# 很多人問說，如果你這麼厲害，為什麼不要自己賺就好，他之所以會出來開公司，也是為了用銀行的錢
# 進行1億賺1千萬的交易，而不是小打小鬧，用1萬賺1千塊

# 寶博兩支產品都有放，21:20




# EP61
# 比特幣從一開始到現在有3個循環，每一次的循環都一模一樣，只是最高多了幾十倍
#　2.定期買入比特幣，每天買入60元台幣


def load_data(begin_date, end_date=None, period=None, 
              stock_symbol=None):
    '''
    讀取資料及重新整理
    '''
    
    
    if stock_symbol==None:
        target = ar.stk_get_list(stock_type='tw', 
                              stock_info=False, 
                              update=False,
                              local=True)    
        # Dev
        target = target.iloc[0:10, :]
        stock_symbol = target['STOCK_SYMBOL'].tolist()
    
    
    data_raw = ar.stk_get_data(begin_date=begin_date, end_date=end_date, 
                   stock_type='tw', stock_symbol=stock_symbol, 
                   local=True)    
    
    return data_raw




def get_hold_stock():
    
    
    return ''



def get_buy_signal(data, hold_stocks=None):
    
    loc_data = data.copy()
    loc_data['SIGNAL_THRESHOLD'] = loc_data.groupby('STOCK_SYMBOL')['CLOSE'] \
        .transform('max')

    loc_data['SIGNAL_THRESHOLD'] = loc_data['SIGNAL_THRESHOLD'] * 0.95        
        
    loc_data['PREV_PRICE'] = loc_data \
                            .groupby(['STOCK_SYMBOL'])['CLOSE'] \
                            .shift(-1) \
                            .reset_index(drop=True)        
        
     
        
    loc_data.loc[loc_data['PREV_PRICE'] > loc_data['SIGNAL_THRESHOLD'], 
                 'SIGNAL'] = True
    
    loc_data = loc_data[loc_data['SIGNAL']==True] \
            .reset_index(drop=True)
    
    return loc_data


# ..............


def get_sell_signal(data, hold_stocks=None):
    
    return ''


# .............
    

def model_dev1(data_end, data_begin=None, data_period=60,
              forecast_end=None, forecast_period=30,
              stock_symbol=None,
              remove_none=True):
    '''
    remove_none    : boolean. Remove row without any positive signal.
    '''
    
    
    periods = ar.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    
    
    loc_data = load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)
    
    
    loc_data['LAST_NUM'] = loc_data['STOCK_SYMBOL'].str.slice(start=-1)
    loc_data['LAST_NUM'] = loc_data['LAST_NUM'].astype(int)
    
    
    # 
    loc_data['CLOSE_STR'] = loc_data['CLOSE'].astype(str)
    loc_data['CLOSE_STR'] = loc_data['CLOSE_STR'].str.slice(start=-1)
    loc_data['CLOSE_STR'] = loc_data['CLOSE_STR'].astype(int)
    
    
    # 
    loc_data['SIGNAL_DAY'] = loc_data['WORK_DATE'].apply(cbyz.ymd).dt.weekday
    loc_data['SIGNAL_DAY'] = loc_data['SIGNAL_DAY'].astype(int)


    # Signal
    loc_data.loc[loc_data['LAST_NUM']==loc_data['SIGNAL_DAY'],
                 'BUY_SIGNAL'] = True
    
    loc_data.loc[loc_data['CLOSE_STR']==loc_data['SIGNAL_DAY'],
                 'SELL_SIGNAL'] = True    
    
    
    loc_data = cbyz.df_conv_na(loc_data, 
                             cols=['BUY_SIGNAL', 'SELL_SIGNAL'],
                             value=False)
    
    loc_data['MODEL_ID'] = 'DEV'
    
    
    # Remove row without any positive signal.
    if remove_none == True:
        loc_data = loc_data[(loc_data['BUY_SIGNAL']==True) |
                            (loc_data['SELL_SIGNAL']==True)] \
            .reset_index(drop=True)
    
    loc_data = loc_data[['STOCK_SYMBOL', 'MODEL_ID', 'WORK_DATE',
                       'BUY_SIGNAL', 'SELL_SIGNAL']]   


    # Worklist
    # 1.Forecast the trend in the next N days.
    # 2.If forecasted ROI reach goal, then buy_signal will be true.      

    return loc_data



# ...............
    

def model_dev2(data_end, data_begin=None, data_period=60,
              forecast_end=None, forecast_period=30,
              stock_symbol=None,
              remove_none=True):
    '''
    remove_none    : boolean. Remove row without any positive signal.
    '''
    
    
    periods = ar.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    
    
    loc_data = load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)
    
    
    loc_data['LAST_NUM'] = loc_data['STOCK_SYMBOL'].str.slice(start=-1)
    loc_data['LAST_NUM'] = loc_data['LAST_NUM'].astype(int)
    
    
    # 
    loc_data['CLOSE_STR'] = loc_data['CLOSE'].astype(str)
    loc_data['CLOSE_STR'] = loc_data['CLOSE_STR'].str.slice(start=-1)
    loc_data['CLOSE_STR'] = loc_data['CLOSE_STR'].astype(int)
    
    
    # 
    loc_data['SIGNAL_DAY'] = loc_data['WORK_DATE'].apply(cbyz.ymd).dt.weekday
    loc_data['SIGNAL_DAY'] = loc_data['SIGNAL_DAY'].astype(int)


    # Signal
    loc_data.loc[loc_data['LAST_NUM']==loc_data['SIGNAL_DAY'],
                 'BUY_SIGNAL'] = True
    
    loc_data.loc[loc_data['CLOSE_STR']==loc_data['SIGNAL_DAY'],
                 'SELL_SIGNAL'] = True    
    
    
    loc_data = cbyz.df_conv_na(loc_data, 
                             cols=['BUY_SIGNAL', 'SELL_SIGNAL'],
                             value=False)
    
    loc_data['MODEL_ID'] = 'DEV'
    
    
    # Remove row without any positive signal.
    if remove_none == True:
        loc_data = loc_data[(loc_data['BUY_SIGNAL']==True) |
                            (loc_data['SELL_SIGNAL']==True)] \
            .reset_index(drop=True)
    
    loc_data = loc_data[['STOCK_SYMBOL', 'MODEL_ID', 'WORK_DATE',
                       'BUY_SIGNAL', 'SELL_SIGNAL']]   


    # Worklist
    # 1.Forecast the trend in the next N days.
    # 2.If forecasted ROI reach goal, then buy_signal will be true.      

    return loc_data




# ...............


def model_dev3(**args):
    
    print('model_dev3')
    
    return 'model_dev3'

# ..............


def get_model_list(status=[0,1]):
    '''
    List all analysis here
    '''    

    # (1) List manually
    # (2) List by model historic performance
    # (3) List by function pattern
    
    # function_list = [model_dev1, model_dev2, model_dev3]
    function_list = [model_1]
    
    
    return function_list




# ...............


def analyze_center(data):
    '''
    List all analysis here
    '''    
    
    analyze_results = get_top_price(data)
    

    # Results format
    # (1) Only stock passed test will show in the results
    # STOCK_SYMBOL
    # MODEL_ID, or MODEL_ID
    
    return analyze_results






# %% Master ------
    

# Update, 增加 停損點
def master(today=None, hold_stocks=None, roi=10, limit=90):
    '''
    主工作區
    roi:     percent
    limit:   days
    '''
    
    global stock_data
    stock_data = load_data()
    
    global analyze_results
    analyze_results = analyze_center(data=stock_data)
    
    
    # v0
    # buy_signal = get_buy_signal(data=stock_data,
    #                             hold_stocks=hold_stocks)
    
    # sell_signal = get_sell_signal(data=analyze_results,
    #                               hold_stocks=hold_stocks)
    
    # master_results = {'RESULTS':analyze_results,
    #                   'BUY_SIGNAL':buy_signal,
    #                   'SELL_SIGNAL':sell_signal}
    
    return analyze_results



def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()



# %% Dev ---------


def get_top_price(data):
    '''
    Dev
    '''
    
    # data = stock_data.copy()
    loc_data = data.copy()
    # loc_data = loc_data[['STOCK_SYMBOL', 'CLOSE']]
    
    top_price = data.groupby('STOCK_SYMBOL')['CLOSE'] \
                .aggregate(max) \
                .reset_index() \
                .rename(columns={'CLOSE':'MAX_PRICE'})

        
    results_pre = loc_data.merge(top_price, 
                         how='left', 
                         on='STOCK_SYMBOL')
    
    
    # Temp ---------
    # Add a test here
    cur_price = results_pre \
        .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE'],
                     ascending=[True, False]) \
        .drop_duplicates(subset=['STOCK_SYMBOL'])
        
    cur_price = cur_price[['STOCK_SYMBOL', 'CLOSE']] \
        .rename(columns={'CLOSE':'CUR_PRICE'})
     
    # Reorganize -------        
    results = top_price.merge(cur_price, 
                         on='STOCK_SYMBOL')
    
    
    results.loc[results['CUR_PRICE'] > results['MAX_PRICE'] * 0.95,
                'BUY_SIGNAL'] = True
    
    
    results.loc[results['CUR_PRICE'] < results['MAX_PRICE'] * 0.3,
                'SELL_SIGNAL'] = True
    
    
    
    results = results[(~results['BUY_SIGNAL'].isna()) |
                      (~results['SELL_SIGNAL'].isna())] \
        .reset_index(drop=True)

        
    results['MODEL_ID'] = 'TM01'
    results = results[['STOCK_SYMBOL', 'MODEL_ID',
                       'BUY_SIGNAL', 'SELL_SIGNAL']]

    return results   





def model_1(data=None, data_end=None, data_begin=None, 
            data_period=150, forecast_end=None, forecast_period=30,
              stock_symbol=None,
              remove_none=True):

    
    from sklearn.linear_model import LinearRegression
    
    periods = ar.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    
    
    loc_data = load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)    
    
    
    loc_data['PRICE_PRE'] = loc_data \
        .groupby('STOCK_SYMBOL')['CLOSE'] \
                            .shift(forecast_period)
    
    model_data = loc_data[~loc_data['PRICE_PRE'].isna()]


    # real_price = loc_data[loc_data['WORK_DATE']==periods['DATA_END']]
    
    
    # Predict data ......
    forecast_data_pre = ar.df_add_rank(loc_data,
                                       group_key='STOCK_SYMBOL',
                                       value='WORK_DATE',
                                       reverse=True)
        
    forecast_data = forecast_data_pre[(forecast_data_pre['RANK']>=0) \
                                  & (forecast_data_pre['RANK']<forecast_period)]
    
    # Date
    forecast_date = cbyz.get_time_seq(begin_date=periods['FORECAST_BEGIN'],
                                    periods=forecast_period,
                                         unit='d',
                                         simplify_date=True)
        
    forecast_date = forecast_date['WORK_DATE'].tolist()
        
    # Model ........
    model_info = pd.DataFrame()
    forecast_results = pd.DataFrame()
    
    for i in range(0, len(stock_symbol)):
        
        cur_symbol = stock_symbol[i]
        # print(i)
    
        # Model .........
        # Update, doesn't need reshape with multiple features.
        x = model_data['PRICE_PRE'].to_numpy()
        x = x.reshape(-1,1)
        
        y = model_data['CLOSE'].to_numpy()
        
        reg = LinearRegression().fit(x, y)
        
        reg.score(x, y)
        reg.coef_
        reg.intercept_
    
    
        # Forecast ......
        temp_forecast = forecast_data[
            forecast_data['STOCK_SYMBOL']==cur_symbol]
        
        temp_forecast = temp_forecast['CLOSE'].to_numpy()
        
        # Bug,只有一個feature時才需要reshape
        temp_forecast = temp_forecast.reshape(-1,1)              
        
        temp_results = reg.predict(temp_forecast)    
        
        
        # print(stock_symbol[i])
        # print(forecast_date)
        # print(temp_results)
        
        # ...
        temp_df = pd.DataFrame(data={'WORK_DATE':forecast_date,
                                     'CLOSE':temp_results})
        temp_df['STOCK_SYMBOL'] = cur_symbol
        
        forecast_results = forecast_results.append(temp_df)
        
    
    # Rearrage
    # (1) real_price for backtest function record buy_price
    cols = ['STOCK_SYMBOL', 'WORK_DATE', 'CLOSE']        
    forecast_results = forecast_results[cols]

    return_dict = {'MODEL_INFO':model_info,
                   # 'REAL_PRICE':real_price,
                   'RESULTS':forecast_results}

    return return_dict




def model_template(data_end, data_begin=None, data_period=150,
              forecast_end=None, forecast_period=30,
              stock_symbol=None,
              remove_none=True):
    
    from sklearn.linear_model import LinearRegression
    
    periods = ar.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    
    
    loc_data = load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)    
    
    # Model .........



    return ''




data_begin = 20200301

# data_begin=None
data_end=None
data_period=30
forecast_end=None
forecast_period=30

stock_symbol=['0050', '0056']














