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




# 每日交易量很難看 不太準 因為有時候會爆大量 有時候很冷門。有時候正常交易 有時候是被當沖炒股
# 流通性我基本上看股本
# 台股來說 0-10億小股本股票 流通差。10-100億中型股。破100億算大型股


# Dashboard，只跑三年資料
# Process, select target > analyze > backtest
# Filter with 股本 and 成交量



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
import arsenal_stock as stk



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




def sam_load_data(data_begin, data_end=None, stock_type='tw', period=None, 
                  stock_symbol=None, lite=True, full=False):
    '''
    讀取資料及重新整理
    '''
    

    
    if stock_symbol == None:
        target = stk.get_list(stock_type='tw', 
                              stock_info=False, 
                              update=False,
                              local=local)    
        # Dev
        target = target.iloc[0:10, :]
        stock_symbol = target['STOCK_SYMBOL'].tolist()
    
    
    if full == True:
        full_data = stk.get_data(data_begin=None, 
                            data_end=None, 
                            stock_type=stock_type, 
                            stock_symbol=stock_symbol, 
                            local=local)    
    else:
        full_data = pd.DataFrame()
        
    

    if lite == True:
        lite_data = stk.get_data(data_begin=data_begin, 
                               data_end=data_end, 
                               stock_type=stock_type, 
                               stock_symbol=stock_symbol, 
                               local=local)
    else:
        lite_data = pd.DataFrame()    
    
    

    export_dict = {'DATA':lite_data,
                   'FULL_DATA':full_data}    
    
    
    return export_dict




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
                   stock_symbol=[], lag=7):
    
    
    today = cbyz.get_time_serial()
    
    if today < data_end:
        msg = "sam.get_model_data - " + str(data_end) + ' exceed today.'
        print(msg)
        return msg
    
    
    # ......
    stock_symbol = cbyz.conv_to_list(stock_symbol)
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, 'str')
    
    test_data = stk.stk_test_data_period(data_begin=data_begin, 
                                         data_end=data_end, 
                                         data_period=data_period,
                                         predict_end=predict_end, 
                                         predict_period=predict_period,
                                         lag=lag,
                                         stock_type='tw', 
                                         stock_symbol=stock_symbol[0],
                                         local=local)
    
    
    data_shift_begin = test_data['DATA_SHIFT_BEGIN']
    data_end = test_data['DATA_END']
    predict_begin = test_data['PREDICT_BEGIN']
    predict_end = test_data['PREDICT_END']    
    
    
    
    # # 為了避免執行df_add_shift_data，data_begin變成NA而被刪除，先將data_begin往前推
    # # N天，且為了避免遇到假日，再往前推20天。
    # temp_data_begin = cbyz.date_cal(data_begin,
    #                                 amount=-predict_period-20,
    #                                 unit='d')

    shift_head_begin, shift_head_end, \
            data_begin, data_end, predict_begin, predict_end = \
                cbyz.date_get_period(data_begin=data_shift_begin, 
                                   data_end=data_end, 
                                   data_period=None,
                                   predict_end=predict_begin, 
                                   predict_period=predict_end)
    
    # Date ......
    predict_date = cbyz.time_get_seq(begin_date=predict_begin,
                                      periods=predict_period,
                                      unit='d', simplify_date=True)
        
    predict_date = predict_date[['WORK_DATE']]
    predict_date_list = predict_date['WORK_DATE'].tolist()
    
    
    # Predict DataFrame ......
    stock_symbol_df = pd.DataFrame({'STOCK_SYMBOL':stock_symbol})
    predict_df_pre = cbyz.df_cross_join(stock_symbol_df, predict_date)
    
    
    # Load Data ----------
    
    # Historical Data ...
    loc_data_raw = sam_load_data(data_begin=shift_head_begin,
                                 data_end=data_end,
                                 stock_symbol=stock_symbol)    
    
    
    loc_data = loc_data_raw['DATA']
    
    loc_data = loc_data.sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
        .reset_index(drop=True)
    
     
    # Calendar ...
    calendar = ar.get_calendar(simplify=True)
    calendar = calendar.drop('WEEK_ID', axis=1)
    
    
    # Add predict Data ......
    loc_predict_df = loc_data.merge(predict_df_pre, how='outer',
                                    on=['STOCK_SYMBOL', 'WORK_DATE'])
    
    loc_predict_df = loc_predict_df \
                        .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                        .reset_index(drop=True)
    
    loc_predict_df = loc_predict_df.merge(calendar, how='left', 
                                          on='WORK_DATE')
    
    
    # Shift ......
    shift_cols = ['CLOSE', 'HIGH', 'LOW', 'VOLUME']
    
    loc_data_shift, lag_cols = cbyz.df_add_shift_data(df=loc_predict_df, 
                                            cols=shift_cols, 
                                            shift=predict_period,
                                            group_key=['STOCK_SYMBOL'],
                                            suffix='_LAG', remove_na=False)
    
    
    var_cols = ['MONTH', 'WEEKDAY', 'WEEK_NUM']
    
            
    # Bug, predict_period的長度和data_period太接近時會出錯
    # Update, add as function   
    model_x = lag_cols + var_cols
    model_y = ['CLOSE']

    
    # Model Data
    loc_model_data = loc_data_shift.dropna(subset=model_x)
    loc_model_data = loc_model_data[loc_model_data['WORK_DATE']>data_begin] \
                        .reset_index(drop=True)


    # Normalize ......
    norm_cols = cbyz.li_join_flatten(model_x, model_y) 
    loc_model_data_norm = cbyz.df_normalize(df=loc_model_data,
                                            cols=norm_cols,
                                            groupby=['STOCK_SYMBOL'])



    export_dict = {'MODEL_DATA':loc_model_data_norm['RESULTS'],
                   # 'PRECIDT_DATA':predict_data,
                   'PRECIDT_DATE':predict_date_list,
                   'MODEL_X':model_x,
                   'MODEL_Y':model_y,
                   'DATA_BEGIN':data_begin,
                   'DATA_END':data_end,
                   'NORM_ORIG':loc_model_data_norm['ORIGINAL'],
                   'NORM_GROUP':loc_model_data_norm['GROUPBY']}
    
    return export_dict






def model_1(model_data, predict_date, model_x, model_y,
            stock_symbol, remove_none=True):
    '''
    Linear regression
    '''
    
    from sklearn.linear_model import LinearRegression
    
        
    # Model ........
    model_info = pd.DataFrame()
    results = pd.DataFrame()
    rmse = pd.DataFrame()
    
    for i in range(0, len(stock_symbol)):
        
        cur_symbol = stock_symbol[i]
        
        
        # Dataset
        X_train, X_test, y_train, y_test, X_predict = \
            split_data(model_data=model_data, symbol=cur_symbol, 
                       predict_begin=predict_date[0], model_x=model_x,
                       model_y=model_y)        
        
        try:
            reg = LinearRegression().fit(X_train, y_train)
            # reg.score(x, y)
            # reg.coef_
            # reg.intercept_
        except:
            continue


        # Test Group ......
        preds_test = reg.predict(X_test)
        new_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    
    
        # predict ......      
        preds_test = reg.predict(X_predict)  
        preds = reg.predict(X_predict).tolist()
        preds = cbyz.li_join_flatten(preds)
        
        
        # Combine ......
        new_results = pd.DataFrame({'WORK_DATE':predict_date,
                                    'CLOSE':preds})
        new_results['STOCK_SYMBOL'] = cur_symbol
        results = results.append(new_results)           

        # RMSE ......
        
        new_rmse_df = pd.DataFrame({'STOCK_SYMBOL':[cur_symbol],
                                    'RMSE':[new_rmse]})
        
        rmse = rmse.append(new_rmse_df)
    
    
    # Reorganize ------
    if len(results) > 0:
        results = results[['WORK_DATE', 'STOCK_SYMBOL', 'CLOSE']] \
                    .reset_index(drop=True)
    
    return_dict = {'RESULTS':results,
                   'FEATURES':pd.DataFrame(),
                   'RMSE':rmse} 

    return return_dict


# ................
    

def split_data(model_data, symbol, predict_begin, model_x, model_y):
    
    # Model Data ......
    cur_model_data = model_data[
        (model_data['STOCK_SYMBOL']==symbol) \
            & (model_data['WORK_DATE']<predict_begin)] \
        .reset_index(drop=True)
    
    
    # Predict Data ......
    cur_predict_data = model_data[
        (model_data['STOCK_SYMBOL']==symbol) \
            & (model_data['WORK_DATE']>=predict_begin)]
        
    X_predict = cur_predict_data[model_x]
    
    
    # Traning And Test
    X = cur_model_data[model_x]
    y = cur_model_data[model_y]      
    X_train, X_test, y_train, y_test = train_test_split(X, y)    
    
    return X_train, X_test, y_train, y_test, X_predict




def model_2(model_data, predict_date, model_x, model_y, 
            stock_symbol, remove_none=True):
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
        
        # Dataset
        X_train, X_test, y_train, y_test, X_predict = \
            split_data(model_data=model_data, symbol=cur_symbol, 
                       predict_begin=predict_date[0], model_x=model_x,
                       model_y=model_y)
        
        # # Model Data ......
        # cur_model_data = model_data[
        #     (model_data['STOCK_SYMBOL']==cur_symbol) \
        #         & (model_data['WORK_DATE']<predict_date[0])] \
        #     .reset_index(drop=True)


        # # Predict Data ......
        # cur_predict_data = model_data[
        #     (model_data['STOCK_SYMBOL']==cur_symbol) \
        #         & (model_data['WORK_DATE'].isin(predict_date))
        #     ]
            
        # predict_x = cur_predict_data[model_x]

        
        # # Traning And Test
        # X = cur_model_data[model_x]
        # y = cur_model_data[model_y]      
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        
        # Bug, 確認為什麼會有問題
        try:
            regressor = xgb.XGBRegressor(
                n_estimators=100,
                reg_lambda=1,
                gamma=0,
                max_depth=3
            )
            
        except:
            continue


        regressor.fit(X_train, y_train)
        
        
        # Feature Importance ......
        importance_dict = {'FEATURES':list(X_train.columns),
                           'IMPORTANCE':regressor.feature_importances_}
        
        new_features = pd.DataFrame(importance_dict)    
        new_features['STOCK_SYMBOL'] = cur_symbol
        
        features = features.append(new_features)


        # Test Group ......
        preds_test = regressor.predict(X_test)
        new_rmse = np.sqrt(mean_squared_error(y_test, preds_test))


        # Prediction ......
        preds = regressor.predict(X_predict)

        # Combine ......
        new_results = pd.DataFrame({'WORK_DATE':predict_date,
                                    'CLOSE':preds})
        new_results['STOCK_SYMBOL'] = cur_symbol
        results = results.append(new_results)

        
        # RMSE ......
        new_rmse_df = pd.DataFrame({'STOCK_SYMBOL':[cur_symbol],
                                'RMSE':[new_rmse]})
        
        rmse = rmse.append(new_rmse_df)
        


    if len(results) > 0:
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




def predict(predict_begin, predict_end, predict_period, stock_symbol,
            lag=14):
    
    
    # predict_begin = 20210402
    # predict_end = 20210415
    # predict_period = 14
    # lag = 14
    
    # stock_symbol = [2520, 2605, 6116, 6191, 3481, 2409]
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, to_type='str')
    


    # # .......
    # loc_time_seq = cbyz.time_get_seq(begin_date=begin_date,
    #                                  periods=backtest_times,
    #                                  unit='d', skip=interval,
    #                                  simplify_date=True)
    
    # loc_time_seq = loc_time_seq['WORK_DATE'].tolist()

    
    # # Work area ----------

    
    
    # # Date .........
    # for i in range(0, len(loc_time_seq)):
    

    shift_begin, shift_end, \
            data_begin, data_end, predict_begin, predict_end = \
                cbyz.date_get_period(data_begin=None,
                                       data_end=None, 
                                       data_period=360,
                                       predict_begin=predict_begin, 
                                       predict_end=None, 
                                       predict_period=predict_period)        
    

    # data_shift_begin = date_period['DATA_BEGIN']
    # data_end = date_period['DATA_END']
    # predict_begin = date_period['PREDICT_BEGIN']
    # predict_end = date_period['PREDICT_END']



    # (1) data_begin and data_end may be changed here.        
    # Update，增加df_normalize
    
    data_raw = get_model_data(data_begin=shift_begin,
                                  data_end=data_end,
                                  data_period=None, 
                                  predict_end=predict_end, 
                                  predict_period=predict_period,
                                  stock_symbol=stock_symbol,
                                  lag=lag)
    
    data_begin = data_raw['DATA_BEGIN']
    data_end = data_raw['DATA_END']        

    
    
    # Update, set to global variables?
    model_data = data_raw['MODEL_DATA']
    predict_date = data_raw['PRECIDT_DATE']
    model_x = data_raw['MODEL_X']
    model_y = data_raw['MODEL_Y']            




    # Model ......
    model_list = get_model_list()
    results = pd.DataFrame()
    rmse = pd.DataFrame()    
    # error_msg = []    
    
    for i in range(0, len(model_list)):

        cur_model = model_list[i]
        
        # Model results .......
        # (1) Update, should deal with multiple signal issues.
        #     Currently, only consider the first signal.
        # (2) predict_data doesn't use.
        
        # global model_results_raw
        model_results_raw = cur_model(stock_symbol=stock_symbol,
                                      model_data=model_data, 
                                      predict_date=predict_date,
                                      model_x=model_x, 
                                      model_y=model_y,
                                      remove_none=True)      
        
        
        if len(model_results_raw['RESULTS']) == 0:
            continue
        
        
        model_name = cur_model.__name__
        
        
        # Buy Signal
        temp_results = model_results_raw['RESULTS']
        # temp_results = temp_results.merge(buy_price, 
        #                                     how='left',
        #                                     on='STOCK_SYMBOL')
        
        
        # temp_results['ROI'] = (temp_results['CLOSE'] \
        #                         - temp_results['BUY_PRICE']) \
        #                         / temp_results['BUY_PRICE']


        # temp_results = temp_results[temp_results['ROI'] >= roi_base]
        
        
        # temp_results['MODEL'] = model_name
        # temp_results['DATA_BEGIN'] = data_begin
        # temp_results['DATA_END'] = data_end
        # temp_results['PREDICT_BEGIN'] = predict_begin
        # temp_results['PREDICT_END'] = predict_end
        # temp_results['BACKTEST_ID'] = i
        
        # buy_signal = buy_signal.append(temp_results)        
        
        
        # RMSE ......
        new_rmse = model_results_raw['RMSE']
        new_rmse['MODEL'] = model_name
        new_rmse['BACKTEST_ID'] = i
        rmse = rmse.append(new_rmse)
        
        results = results.append(temp_results)
        
    
    # if len(buy_signal) == 0:
    #     print('bt_buy_signal return 0 row.')
    #     return pd.DataFrame()
    
    
    # buy_signal = buy_signal.rename(columns={'CLOSE':'FORECAST_CLOSE'})
    rmse = rmse.reset_index(drop=True)
    
    
    # Add dynamic ROI -------
    # Temporaily Remove
    # predict_roi_pre = cbyz.df_add_rank(buy_signal, 
    #                            value='WORK_DATE',
    #                            group_key=['STOCK_SYMBOL', 'BACKTEST_ID'])
            

    # predict_roi = pd.DataFrame()
    
    
    # for i in range(0, len(loc_time_seq)):
    
    #     df_lv1 = predict_roi_pre[predict_roi_pre['BACKTEST_ID']==i]
    #     unique_symbol = df_lv1['STOCK_SYMBOL'].unique()
        
        
    #     for j in range(0, len(unique_symbol)):
            
    #         df_lv2 = df_lv1[df_lv1['STOCK_SYMBOL']==unique_symbol[j]] \
    #                     .reset_index(drop=True)
            
    #         for k in range(0, len(df_lv2)):
                
    #             if k == 0:
    #                 df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'BUY_PRICE']
    #                 continue
                    
    #             if df_lv2.loc[k, 'FORECAST_CLOSE'] >= df_lv2.loc[k-1, 'MAX_PRICE']:
    #                 df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k, 'FORECAST_CLOSE']
    #             else:
    #                 df_lv2.loc[k, 'MAX_PRICE'] = df_lv2.loc[k-1, 'MAX_PRICE']
                            
    #         predict_roi = predict_roi.append(df_lv2)    
    
    
    
    results = cbyz.df_normalize_restore(df=results, 
                                        original=data_raw['NORM_ORIG'],
                                        groupby=data_raw['NORM_GROUP'])
    
    export_dict = {'RESULTS':results, 
                   'RMSE':rmse}
    
    
    return results, rmse





# %% Master ------
# data_begin=None, data_end=None, 
def master(predict_begin=None, predict_end=None, 
           predict_period=15, data_period=150, 
           stock_symbol=None, today=None, hold_stocks=None, limit=90):
    '''
    主工作區
    roi:     percent
    limit:   days
    '''
    
    
    shift_begin, shift_end, \
            data_begin, data_end, predict_begin, predict_end = \
                cbyz.date_get_period(data_begin=None,
                                       data_end=None, 
                                       data_period=data_period,
                                       predict_begin=predict_begin,
                                       predict_end=predict_end, 
                                       predict_period=predict_period)        
            
    
    # global stock_data
    # stock_symbol = ['2301', '2474', '1714', '2385']
    # stock_symbol = [2520, 2605, 6116, 6191, 3481, 2409]
    
    stock_data = sam_load_data(data_begin=shift_begin,
                               data_end=data_end,
                               stock_symbol=stock_symbol)
    
    
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
    predict_date = data_raw['PRECIDT_DATE']
    model_x = data_raw['MODEL_X']
    model_y = data_raw['MODEL_Y']    
    
    
    # predict_begin = 20210402
    # predict_end = 20210415
    # predict_period = 14
    
    
    results = predict(predict_begin=predict_begin, predict_end=predict_end,
                       predict_period=predict_period, 
                       stock_symbol=stock_symbol)
    
    
    return results



# data_begin=20171001
# data_end=20191231
# data_period=None
# predict_end=None
# predict_period=30
# stock_symbol=['2301', '2474', '1714', '2385']






def check():
    '''
    資料驗證
    '''    
    return ''




if __name__ == '__main__':
    master()










def select_stock():
    2301
    
    return ''







