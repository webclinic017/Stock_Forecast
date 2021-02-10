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
from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_squared_error


local = False
local = True

stock_type = 'tw'

# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/2_Stock_Analysis'
else:
    path = '/home/aronhack/stock_predict/2_Stock_Analysis'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 r'/home/aronhack/stock_predict/Function']


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




def sam_load_data(data_begin, data_end=None, period=None, 
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
    
    
    data_raw = ar.stk_get_data(data_begin=data_begin, data_end=data_end, 
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
    




def get_model_data(data_begin=None, data_end=None,
                   data_period=150, predict_end=None, predict_period=15,
                   stock_symbol=[]):
    
    
    # 為了避免執行df_add_shift_data，data_begin變成NA而被刪除，先將data_begin往前推
    # N天，且為了避免遇到假日，再往前推20天。
    data_begin_new = cbyz.date_cal(data_begin, amount=-predict_period-20,
                               unit='d')


    test_data = ar.stk_test_data_period(data_begin=data_begin_new, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 predict_end=predict_end, 
                                 predict_period=predict_period,
                                 stock_type='tw',
                                 local=local)
    
    
    periods = cbyz.date_get_period(data_begin=test_data['DATA_BEGIN'], 
                                   data_end=test_data['DATE_END'], 
                                   data_period=None,
                                   predict_end=test_data['PREDICT_BEGIN'], 
                                   predict_period=test_data['PREDICT_END'])
    
    # Date ......
    predict_date = cbyz.time_get_seq(begin_date=periods['PREDICT_BEGIN'],
                                      periods=predict_period,
                                      unit='d', simplify_date=True)
        
    predict_date = predict_date[['WORK_DATE']]
    
    
    # Predict DataFrame ......
    stock_symbol_df = pd.DataFrame({'STOCK_SYMBOL':stock_symbol})
    predict_df_pre = cbyz.df_cross_join(stock_symbol_df, predict_date)
    predict_df_pre['PREDICT'] = True
    
    
    # Load Data ......
    loc_data = sam_load_data(data_begin=periods['DATA_BEGIN'],
                             data_end=periods['DATA_END'],
                             stock_symbol=stock_symbol)    
    
    
    loc_data = loc_data.sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
        .reset_index(drop=True)
    
    # Add predict Data ......
    loc_predict_df = loc_data.merge(predict_df_pre, how='outer',
                                    on=['STOCK_SYMBOL', 'WORK_DATE'])
    
    loc_predict_df = loc_predict_df \
                        .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                        .reset_index(drop=True)
    
    
    loc_data_shift = cbyz.df_add_shift_data(df=loc_predict_df, 
                                            cols=['CLOSE', 'VOLUME'], 
                                            shift=predict_period,
                                            group_key=['STOCK_SYMBOL'],
                                            suffix='_PRE', remove_na=False)
    
            
    # Bug, predict_period的長度和data_period太接近時會出錯
    # Update, add as function   
    model_x = loc_data_shift['SHIFT_COLS']
    model_y = ['CLOSE']

    
    # Model Data
    loc_model_data = loc_data_shift['DF'].dropna(subset=model_x)
    loc_model_data = loc_model_data[loc_model_data['WORK_DATE']>=data_begin] \
                        .reset_index(drop=True)


    # Predict data ......
    predict_data_pre = cbyz.df_add_rank(df=loc_model_data,
                                       group_key='STOCK_SYMBOL',
                                       value='WORK_DATE',
                                       reverse=True)
        
    predict_data = predict_data_pre[(predict_data_pre['RANK']>=0) \
                                  & (predict_data_pre['RANK']<predict_period)]
    

    export_dict = {'MODEL_DATA':loc_model_data,
                   'PRECIDT_DATA':predict_data,
                   'PRECIDT_DATE':predict_date,
                   'MODEL_X':model_x,
                   'MODEL_Y':model_y}
    
    return export_dict






def model_1(model_data, predict_data, predict_date, model_x, model_y,
            remove_none=True):
    '''
    Linear regression
    '''
    
    from sklearn.linear_model import LinearRegression
    
        
    # Model ........
    model_info = pd.DataFrame()
    results = pd.DataFrame()
    
    for i in range(0, len(stock_symbol)):
        
        cur_symbol = stock_symbol[i]
        
        temp_data = model_data[
            model_data['STOCK_SYMBOL']==cur_symbol] \
            .reset_index(drop=True)


        temp_predict_data = temp_data[temp_data['PREDICT']==True]
        predict_x = temp_predict_data[model_x]
        
        
        temp_data = temp_data[temp_data['PREDICT'].isna()]

        x = temp_data[model_x]
        y = temp_data[model_y]      
        
    
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        
        reg = LinearRegression().fit(x, y)
        reg.score(x, y)
        reg.coef_
        reg.intercept_
    
    
        # predict ......      
        preds_test = reg.predict(x_test)  
        new_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
        
        
        preds = reg.predict(predict_x)            
        

        # ...
        new_results = predict_date.copy()
        new_results['CLOSE'] = preds
        new_results['STOCK_SYMBOL'] = cur_symbol
        results = results.append(new_results)        
        

    
    # Reorganize ------
    results = results[['WORK_DATE', 'STOCK_SYMBOL', 'CLOSE']] \
                .reset_index(drop=True)

    return_dict = {'MODEL_INFO':model_info,
                   'RESULTS':results}

    return return_dict


# ................
    

def model_2(model_data, predict_data, predict_date, model_x, model_y, 
            remove_none=True):
    '''
    XGBoost
    https://www.datacamp.com/community/tutorials/xgboost-in-python
    '''
    
    import xgboost as xgb    
   

    # ......
    results = pd.DataFrame()
    features = pd.DataFrame()
    rmse = pd.DataFrame()
    
    
    for i in range(len(stock_symbol)):
        
        cur_symbol = stock_symbol[i]
        
        temp_data = model_data[
            model_data['STOCK_SYMBOL']==cur_symbol] \
            .reset_index(drop=True)


        temp_predict_data = temp_data[temp_data['PREDICT']==True]
        predict_x = temp_predict_data[model_x]
        
        
        temp_data = temp_data[temp_data['PREDICT'].isna()]

        x = temp_data[model_x]
        y = temp_data[model_y]      
        
    
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        
        regressor = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
        )
        

        regressor.fit(x_train, y_train)
        
        # Feature Importance ......
        importance_dict = {'FEATURES':list(x.columns),
                           'IMPORTANCE':regressor.feature_importances_}
        
        new_features = pd.DataFrame(importance_dict)    
        new_features['STOCK_SYMBOL'] = cur_symbol
        
        features = features.append(new_features)


        # Test Group ......
        preds_test = regressor.predict(x_test)
        new_rmse = np.sqrt(mean_squared_error(y_test, preds_test))


        # Prediction ......
        preds = regressor.predict(predict_x)
        
        new_results = predict_date.copy()
        new_results['CLOSE'] = preds
        new_results['STOCK_SYMBOL'] = cur_symbol
        results = results.append(new_results)
        
        
        # RMSE ......
        new_rmse_df = pd.DataFrame({'STOCK_SYMBOL':[cur_symbol],
                                'RMSE':[new_rmse]})
        
        rmse = rmse.append(new_rmse_df)
        


    results = results[['WORK_DATE', 'STOCK_SYMBOL', 'CLOSE']] \
                .reset_index(drop=True)
                
    features = features[['STOCK_SYMBOL', 'FEATURES', 'IMPORTANCE']] \
                .reset_index(drop=True)
                
    rmse = rmse.reset_index(drop=True)

    return_dict = {'RESULTS':results,
                   'FEATURES':features,
                   'RMSE':rmse} 

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

def master(data_begin=None, data_end=None, 
           data_period=150, predict_end=None, 
           predict_period=15,
           stock_symbol=None, today=None, hold_stocks=None, limit=90):
    '''
    主工作區
    roi:     percent
    limit:   days
    '''
    
    global stock_data
    stock_data = sam_load_data(data_begin=data_begin)
    
    
    stock_symbol = ['2301', '2474', '1714', '2385']
    # 2301 光寶科
    # 2474可成
    # 1714和桐
    # 2385群光
    
    
    data_raw = get_model_data(data_begin=data_begin,
                              data_end=data_end,
                              data_period=data_period, 
                              predict_end=predict_end, 
                              predict_period=predict_period,
                              stock_symbol=stock_symbol)
    
    # Update, set to global variables?
    model_data = data_raw['MODEL_DATA']
    predict_data = data_raw['PRECIDT_DATA']
    predict_date = data_raw['PRECIDT_DATE']
    model_x = data_raw['MODEL_X']
    model_y = data_raw['MODEL_Y']    
    
    
    
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
    model1_results = model_1(model_data=model_data, predict_data=predict_data,
                             predict_date=predict_date,
                             model_x=model_x, model_y=model_y,
                             remove_none=True)
    
    
    global model2_results
    model2_results = model_2(model_data=model_data, predict_data=predict_data,
                             predict_date=predict_date,
                             model_x=model_x, model_y=model_y,
                             remove_none=True)    
    
    
    return ''



data_begin=20171001
data_end=20191231
data_period=None
predict_end=None
predict_period=30
stock_symbol=['2301', '2474', '1714', '2385']






def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()



# %% Dev ---------


def model_template(data_end, data_begin=None, data_period=150,
              predict_end=None, predict_period=30,
              stock_symbol=None,
              remove_none=True):
    
    from sklearn.linear_model import LinearRegression
    
    periods = cbyz.date_get_period(data_begin=data_begin, 
                                 data_end=data_end, 
                                 data_period=data_period,
                                 predict_end=predict_end, 
                                 predict_period=predict_period)
    
    
    loc_data = sam_load_data(begin_date=periods['DATA_BEGIN'],
                         end_date=periods['DATA_END'],
                         stock_symbol=stock_symbol)    
    
    # Model .........


    # Reorganize ------
    model_info = pd.DataFrame()
    predict_results = pd.DataFrame()
    
    cols = ['STOCK_SYMBOL', 'WORK_DATE', 'CLOSE']        
    predict_results = predict_results[cols]

    return_dict = {'MODEL_INFO':model_info,
                   'RESULTS':predict_results}

    return return_dict




# data_begin = 20200301

# # data_begin=None
# data_end=None
# data_period=30
# predict_end=None
# predict_period=30

# stock_symbol=['0050', '0056']



# periods = cbyz.date_get_period(data_begin=20191201, 
#                              data_end=None, 
#                              data_period=20191231,
#                              predict_end=None, 
#                              predict_period=10)
    






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




