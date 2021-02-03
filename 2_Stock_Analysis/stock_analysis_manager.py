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


# 2301 光寶科
# 2474可成
# 1714和桐
# 2385群光



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
    path = '/home/aronhack/stock_forecast/2_Stock_Analysis'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 r'/home/aronhack/stock_forecast/Function']


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




def sam_load_data(begin_date, end_date=None, period=None, 
              stock_symbol=None):
    '''
    讀取資料及重新整理
    '''
    
    
    if stock_symbol == None:
        target = ar.stk_get_list(stock_type='tw', 
                              stock_info=False, 
                              update=False,
                              local=local)    
        # Dev
        target = target.iloc[0:10, :]
        stock_symbol = target['STOCK_SYMBOL'].tolist()
    
    
    data_raw = ar.stk_get_data(begin_date=begin_date, end_date=end_date, 
                   stock_type='tw', stock_symbol=stock_symbol, 
                   local=local)    
    
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
    


def model_1(data=None, data_end=None, data_begin=None, 
            data_period=150, forecast_end=None, forecast_period=15,
            stock_symbol=['0050', '0056'],
              remove_none=True):
    '''
    Linear regression
    '''
    from sklearn.linear_model import LinearRegression
    
    periods = cbyz.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    
    
    loc_data = sam_load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)    
    
    
    # Bug, forecast_period的長度和data_period太接近時會出錯
    loc_data['PRICE_PRE'] = loc_data \
        .groupby('STOCK_SYMBOL')['CLOSE'] \
                            .shift(forecast_period)
    
    model_data = loc_data[~loc_data['PRICE_PRE'].isna()]


    # Predict data ......
    forecast_data_pre = cbyz.df_add_rank(df=loc_data,
                                       group_key='STOCK_SYMBOL',
                                       value='WORK_DATE',
                                       reverse=True)
        
    forecast_data = forecast_data_pre[(forecast_data_pre['RANK']>=0) \
                                  & (forecast_data_pre['RANK']<forecast_period)]
    
    # Date
    forecast_date = cbyz.time_get_seq(begin_date=periods['FORECAST_BEGIN'],
                                      periods=forecast_period,
                                      unit='d', simplify_date=True)
        
    forecast_date = forecast_date['WORK_DATE'].tolist()
        
    # Model ........
    model_info = pd.DataFrame()
    forecast_results = pd.DataFrame()
    
    for i in range(0, len(stock_symbol)):
        
        cur_symbol = stock_symbol[i]
        # print(i)
    
        # Model .........
        # Update, doesn't need reshape with multiple features.
        
        
        x = cbyz.ml_conv_to_nparray(model_data['PRICE_PRE'])
        
        y = model_data['CLOSE'].to_numpy()
        
        reg = LinearRegression().fit(x, y)
        
        reg.score(x, y)
        reg.coef_
        reg.intercept_
    
    
        # Forecast ......
        temp_forecast = forecast_data[
            forecast_data['STOCK_SYMBOL']==cur_symbol]
        
        
        temp_forecast = cbyz.ml_conv_to_nparray(temp_forecast['CLOSE'])       
        temp_results = reg.predict(temp_forecast)    
        
        
        # print(stock_symbol[i])
        # print(forecast_date)
        # print(temp_results)
        
        # ...
        temp_df = pd.DataFrame(data={'WORK_DATE':forecast_date,
                                     'CLOSE':temp_results})
        temp_df['STOCK_SYMBOL'] = cur_symbol
        
        forecast_results = forecast_results.append(temp_df)
        
    
    # Reorganize ------
    cols = ['STOCK_SYMBOL', 'WORK_DATE', 'CLOSE']        
    forecast_results = forecast_results[cols]

    return_dict = {'MODEL_INFO':model_info,
                   'RESULTS':forecast_results}

    return return_dict


# ................
    

def model_2(data=None, data_end=None, data_begin=None, 
            data_period=150, forecast_end=None, forecast_period=15,
            stock_symbol=['0050', '0056'],
              remove_none=True):
    '''
    XGBoost
    https://www.datacamp.com/community/tutorials/xgboost-in-python
    '''
    
    import xgboost as xgb    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    

    periods = cbyz.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    

    loc_data = sam_load_data(begin_date=periods['DATA_BEGIN'],
                             end_date=periods['DATA_END'],
                             stock_symbol=stock_symbol)    


    # Bug, forecast_period的長度和data_period太接近時會出錯
    # Update, build PRICE_PRE as function
    loc_data['PRICE_PRE'] = loc_data \
        .groupby('STOCK_SYMBOL')['CLOSE'] \
                            .shift(forecast_period)
                            
    loc_data['VOLUME_PRE'] = loc_data \
        .groupby('STOCK_SYMBOL')['VOLUME'] \
                            .shift(forecast_period)                            
    
    model_data = loc_data[~loc_data['PRICE_PRE'].isna()]    
    
    
    # ......
    results = pd.DataFrame()
    features = pd.DataFrame()
    mse = pd.DataFrame()
    
    
    for i in range(len(stock_symbol)):
        
        
        temp_data = model_data[
            model_data['STOCK_SYMBOL']==stock_symbol[i]] \
            .reset_index(drop=True)


        # Update, set x as variables
        X = temp_data[['PRICE_PRE', 'VOLUME_PRE']]
        y = temp_data['CLOSE']      
        
    
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        regressor = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
        )
        

        regressor.fit(X_train, y_train)
        
        # Feature Importance ......
        importance_dict = {'FEATURES':list(X.columns),
                           'IMPORTANCE':regressor.feature_importances_}
        
        new_features = pd.DataFrame(importance_dict)    
        new_features['STOCK_SYMBOL'] = stock_symbol[i]
        
        features = features.append(new_features)


        # Prediction ......
        y_pred = regressor.predict(X_test)
        
        new_results = pd.DataFrame({'CLOSE':y_pred})
        new_results['STOCK_SYMBOL'] = stock_symbol[i]
        results = results.append(new_results)
        
        
        # MSE(或RMSE？) ......
        new_mse = pd.DataFrame({'STOCK_SYMBOL':[stock_symbol[i]],
                                'MSE':[mean_squared_error(y_test, y_pred)]})
        
        mse = mse.append(new_mse)
        


    results = results[['STOCK_SYMBOL', 'CLOSE']] \
                .reset_index(drop=True)
                
    features = features[['STOCK_SYMBOL', 'FEATURES', 'IMPORTANCE']] \
                .reset_index(drop=True)
                
    mse = mse.reset_index(drop=True)

    return_dict = {'RESULTS':results,
                   'FEATURES':features,
                   'MSE':mse} 

    return return_dict



# ................


def get_model_list(status=[0,1]):
    '''
    List all analysis here
    '''    

    # (1) List manually
    # (2) List by model historic performance
    # (3) List by function pattern
    
    # function_list = [model_dev1, model_dev2, model_dev3]
    function_list = [model_1, model_2]
    
    
    return function_list


# ...............


# def analyze_center(data):
#     '''
#     List all analysis here
#     '''    
    
#     analyze_results = get_top_price(data)
    

#     # Results format
#     # (1) Only stock passed test will show in the results
#     # STOCK_SYMBOL
#     # MODEL_ID, or MODEL_ID
    
#     return analyze_results




# %% Master ------

def master(begin_date=20190401, today=None, hold_stocks=None, limit=90):
    '''
    主工作區
    roi:     percent
    limit:   days
    '''
    
    global stock_data
    stock_data = sam_load_data(begin_date=begin_date)
    
    
    stock_symbol = ['2301', '2474', '1714', '2385']
    # 2301 光寶科
    # 2474可成
    # 1714和桐
    # 2385群光
    
    # global analyze_results
    # analyze_results = analyze_center(data=stock_data)
    
    
    # v0
    # buy_signal = get_buy_signal(data=stock_data,
    #                             hold_stocks=hold_stocks)
    
    # sell_signal = get_sell_signal(data=analyze_results,
    #                               hold_stocks=hold_stocks)
    
    # master_results = {'RESULTS':analyze_results,
    #                   'BUY_SIGNAL':buy_signal,
    #                   'SELL_SIGNAL':sell_signal}
    
    
    global model1_results
    model1_results = model_1(data=None, data_end=None, data_begin=begin_date, 
                             data_period=150, forecast_end=None, 
                             forecast_period=15,
                             stock_symbol=stock_symbol,
                             remove_none=True)
    
    
    global model2_results
    model2_results = model_2(data=None, data_end=None, data_begin=begin_date, 
                             data_period=150, forecast_end=None, 
                             forecast_period=15,
                             stock_symbol=stock_symbol,
                             remove_none=True)    
    
    return ''



def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()



# %% Dev ---------


def model_template(data_end, data_begin=None, data_period=150,
              forecast_end=None, forecast_period=30,
              stock_symbol=None,
              remove_none=True):
    
    from sklearn.linear_model import LinearRegression
    
    periods = cbyz.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 forecast_end=forecast_end, 
                                 forecast_period=forecast_period)
    
    
    loc_data = sam_load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)    
    
    # Model .........


    # Reorganize ------
    model_info = pd.DataFrame()
    forecast_results = pd.DataFrame()
    
    cols = ['STOCK_SYMBOL', 'WORK_DATE', 'CLOSE']        
    forecast_results = forecast_results[cols]

    return_dict = {'MODEL_INFO':model_info,
                   'RESULTS':forecast_results}

    return return_dict




# data_begin = 20200301

# # data_begin=None
# data_end=None
# data_period=30
# forecast_end=None
# forecast_period=30

# stock_symbol=['0050', '0056']



# periods = cbyz.date_get_period(data_begin=20191201, 
#                              data_end=None, 
#                              data_period=20191231,
#                              forecast_end=None, 
#                              forecast_period=10)
    

# data_begin=20191201
# data_end=20191231
# data_period=None
# forecast_end=None
# forecast_period=10
# stock_symbol=['0050', '0056']


begin_date=20191201
end_date=20191231


def select_stock():
    2301
    
    return ''


# 每日交易量很難看 不太準 因為有時候會爆大量 有時候很冷門。有時候正常交易 有時候是被當沖炒股
# 流通性我基本上看股本
# 台股來說 0-10億小股本股票 流通差。10-100億中型股。破100億算大型股


# Dashboard，只跑三年資料

# Process, select target > analyze > backtest

# Filter with 股本 and 成交量




# 00:28 Lien 連祥宇 我剛剛想了一下 目前有個小問題。公式可能需要設一下時間差。台股一天漲幅最大10啪 如果漲停 按你的公式 他會回跌2趴的時候出場。可是如果只有漲2趴 回跌0.4趴的時候你的公式就會出場賣出 可是一般同日買賣 0.4啪算是很平常的小波動。又如當日2啪跌到1啪 是很正常的波動範圍 可是2啪跌到1啪已經是回檔5成了
# 00:29 Lien 連祥宇 所以我在想是否兩最高價之間需要設立時間差？ 如每日計算一次之類的（我個人覺得每日還算太頻繁）否著你會過度交易 一天進出買賣100次之類的 手續費會直接讓你賠大錢
# 00:31 Lien 連祥宇 先睡了 做夢時b波有助於思考




