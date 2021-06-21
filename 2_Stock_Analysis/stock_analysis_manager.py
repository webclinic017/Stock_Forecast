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
# 3. 把Lag改成MA


# Optimization
# 1. 如何在回測的時候，不要分多次讀取歷史資料，而可以一次讀取完？



# 2301 光寶科
# 2474可成
# 1714和桐
# 2385群光




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
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',]


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





def sam_load_data(data_begin, data_end=None, stock_type='tw', period=None, 
                  stock_symbol=None, lite=True, full=False):
    '''
    讀取資料及重新整理
    '''
    
    
    if stock_symbol == None:
        target = stk.get_list(stock_type='tw', stock_info=False, 
                              update=False, local=local)
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



def get_model_data(lag=7):
    
    
    print('Worklist，把lag用ma的算法代替')
    print('Bug，這裡的日期有點混亂，會造成lag中有na，導致normalize出錯')
    
    global shift_begin, shift_end, data_begin, data_end
    global predict_begin, predict_end, predict_period
    
    today = cbyz.get_time_serial(to_int=True)
    
    if today < data_end:
        msg = "sam.get_model_data - " + str(data_end) + ' exceed today.'
        print(msg)
        return msg
    
    
    # ......
    global stock_symbol
    stock_symbol = cbyz.conv_to_list(stock_symbol)
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, 'str')

    
    # 為了避免執行df_add_shift_data，data_begin變成NA而被刪除，先將data_begin往前推
    # N天，且為了避免遇到假日，再往前推20天。
    temp_data_begin = cbyz.date_cal(data_begin,
                                    amount=-predict_period-30,
                                    unit='d')


    shift_head_begin, shift_head_end, \
            data_begin, data_end, predict_begin, predict_end = \
                cbyz.date_get_period(data_begin=temp_data_begin, 
                                    data_end=data_end, 
                                    data_period=None,
                                    predict_begin=predict_begin,
                                    predict_end=predict_end, 
                                    predict_period=None)

    
    # Date ......
    predict_date = cbyz.date_get_seq(begin_date=predict_begin,
                                     seq_length=predict_period,
                                     interval=1,
                                     unit='d', simplify_date=True,
                                     ascending=True)
        
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
    

    # Add K line
    print("Update, add 'K_LINE_COLOR', 'K_LINE_TYPE', then execute one-hot-encoding")
    print('Bug, PRICE_CHANGE_RATIO會有NA')
    
    loc_data = stk.add_k_line(loc_data)
    # k_line_vars = ['PRICE_CHANGE_RATIO', 'TOP_SHADOW', 'BOTTOM_SHADOW',
    #                'BAR', 'PRICE_CHANGE_LEVEL']

    # k_line_vars = ['TOP_SHADOW', 'BOTTOM_SHADOW']    
    
     
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
    shift_cols = \
        cbyz.df_get_cols_except(df=loc_predict_df,
                                except_cols=['WORK_DATE', 'STOCK_SYMBOL',
                                              'YEAR', 'MONTH', 'WEEKDAY', 
                                              'WEEK_NUM'])
         
    # shift_cols = ['OPEN', 'HIGH', 'CLOSE', 'LOW', 
    #               'VOLUME', 'PRICE_CHANGE']
        
        
    loc_data_shift, lag_cols = cbyz.df_add_shift(df=loc_predict_df, 
                                                  cols=shift_cols, 
                                                  shift=predict_period,
                                                  group_key=['STOCK_SYMBOL'],
                                                  suffix='_LAG', 
                                                  remove_na=False)
    
    
    var_cols = ['MONTH', 'WEEKDAY', 'WEEK_NUM']
    model_x = lag_cols + var_cols
    model_y = ['CLOSE']

    
    # Model Data
    print('Bug, 有許多data row都有NA')
    loc_model_data = loc_data_shift.dropna(subset=model_x)
    loc_model_data = loc_data_shift[loc_data_shift['WORK_DATE']>=data_begin] \
                        .reset_index(drop=True)


    # Normalize ......
    norm_cols = cbyz.li_join_flatten(model_x, model_y) 
    
    print("bug, df_normalize, 是不是na造成以下錯誤")
    print('檢查modex_x中是否有na')
    # '<=' not supported between instances of 'float' and 'str'    
    
    loc_model_data_norm = cbyz.df_normalize(df=loc_model_data,
                                            cols=norm_cols,
                                            groupby=['STOCK_SYMBOL'])

    
    
    chk = loc_model_data[norm_cols]
    chk = cbyz.df_chk_col_na(df=chk)
    
    
    
    export_dict = {'MODEL_DATA':loc_model_data_norm[0],
                   # 'PRECIDT_DATA':predict_data,
                   'PRECIDT_DATE':predict_date_list,
                   'MODEL_X':model_x,
                   'MODEL_Y':model_y,
                   'DATA_BEGIN':data_begin,
                   'DATA_END':data_end,
                   'NORM_ORIG':loc_model_data_norm[1],
                   'NORM_GROUP':loc_model_data_norm[2]}
    
    return export_dict


# ...............



def split_data(symbol=None):
    
    global model_data, model_x, model_y, model_addt_vars, predict_date
    
    # Model Data ......
    if symbol == None:
        cur_model_data = model_data[model_data['WORK_DATE']<predict_date[0]] \
        .reset_index(drop=True)
        
        # Predict Data ......
        cur_predict_data = model_data[model_data['WORK_DATE']>=predict_date[0]]
        
        
    else:
        cur_model_data = model_data[(model_data['STOCK_SYMBOL']==symbol) \
                & (model_data['WORK_DATE']<predict_date[0])] \
            .reset_index(drop=True)
    
        # Predict Data ......
        cur_predict_data = model_data[
            (model_data['STOCK_SYMBOL']==symbol) \
                & (model_data['WORK_DATE']>=predict_date[0])]
        

    global X_train, X_test, y_train, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite

            
    # Predict            
    X_predict = cur_predict_data[model_x + model_addt_vars]
 
    
    # Traning And Test
    X = cur_model_data[model_x + model_addt_vars]
    y = cur_model_data[model_y + model_addt_vars]      
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    
    X_train_lite = X_train.drop(model_addt_vars, axis=1)   
    X_test_lite = X_test.drop(model_addt_vars, axis=1)   
    X_predict_lite = X_predict.drop(model_addt_vars, axis=1)   
    
    
    y_train_lite = y_train['CLOSE']
    y_test_lite = y_test['CLOSE']    
    
    
    # return X_train, X_test, y_train, y_test, X_predict



# def split_data_20210621(symbol):
    
#     global model_data, model_x, model_y, predict_date
    
#     # Model Data ......
#     cur_model_data = model_data[(model_data['STOCK_SYMBOL']==symbol) \
#             & (model_data['WORK_DATE']<predict_date[0])] \
#         .reset_index(drop=True)
    
    
#     # Predict Data ......
#     cur_predict_data = model_data[
#         (model_data['STOCK_SYMBOL']==symbol) \
#             & (model_data['WORK_DATE']>=predict_date[0])]
        
#     X_predict = cur_predict_data[model_x]
    
    
#     # Traning And Test
#     X = cur_model_data[model_x]
#     y = cur_model_data[model_y]      
#     X_train, X_test, y_train, y_test = train_test_split(X, y)    
    
#     return X_train, X_test, y_train, y_test, X_predict


# ..............


def model_3(remove_none=True):
    '''
    Flatten Linear regression
    '''
    
    from sklearn.linear_model import LinearRegression
    global stock_symbol, model_x, model_y, model_addt_vars
    global model_data, predict_date

    
    # Model ........
    model_info = pd.DataFrame()
    results = pd.DataFrame()
    rmse = pd.DataFrame()
    rmse_li = []
    
    
    # Dataset
    split_data(symbol=None)        
        
    
    global X_train, X_test, y_train, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite
    
    

        
    reg = LinearRegression().fit(X_train_lite,
                                 y_train_lite)
    # reg.score(x, y)
    # reg.coef_
    # reg.intercept_
    

    # Test Group ......
    preds_test = reg.predict(X_test_lite)
    
    # RMSE ......
    rmse = np.sqrt(mean_squared_error(y_test_lite, preds_test))


    # predict ......      
    # preds_test = reg.predict(X_predict_lite)  
    preds = reg.predict(X_predict_lite).tolist()
    preds = cbyz.li_join_flatten(preds)
    
    
    # Combine ......
    results = X_predict[model_addt_vars].reset_index(drop=True)
    results['CLOSE'] = preds
    
    rmse = pd.DataFrame(data=[rmse], columns=['RMSE'])
    
    # Reorganize ------
    return_dict = {'RESULTS':results,
                   'FEATURES':pd.DataFrame(),
                   'RMSE':rmse} 

    return return_dict



# ................
    




def model_4(remove_none=True):
    '''
    XGBoost flatten
    https://www.datacamp.com/community/tutorials/xgboost-in-python
    '''
    
    import xgboost as xgb    
    global stock_symbol, model_x, model_y, model_data, predict_date


    global X_train, X_test, y_train, y_test, X_predict
    global X_train_lite, X_test_lite, y_train_lite, y_test_lite, X_predict_lite
    

    # ......
    results = pd.DataFrame()
    features = pd.DataFrame()
    rmse_li = []
    
        
    # Dataset
    split_data(symbol=None)
    
        
    regressor = xgb.XGBRegressor(
        n_estimators=100,
        reg_lambda=1,
        gamma=0,
        max_depth=3
    )
    
    

    regressor.fit(X_train_lite, y_train_lite)
        
        
    # Feature Importance ......
    importance_dict = {'FEATURES':list(X_train_lite.columns),
                       'IMPORTANCE':regressor.feature_importances_}
    
    features = pd.DataFrame(importance_dict)    
    

    # Test Group ......
    preds_test = regressor.predict(X_test_lite)
    
    # RMSE ......
    rmse = np.sqrt(mean_squared_error(y_test_lite, preds_test))


    # Prediction ......
    preds = regressor.predict(X_predict_lite)



    # Results ......
    results = X_predict[model_addt_vars].reset_index(drop=True)
    results['CLOSE'] = preds


    rmse = pd.DataFrame(data=[rmse], columns=['RMSE'])        


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
    
    function_list = [model_3, model_4]
    # function_list = [model_1, model_2, model_3, model_4]
    # function_list = [model_1]

    return function_list


# ...............


def predict():
    
    global shift_begin, shift_end, data_begin, data_end
    global predict_begin, predict_end    
    global model_data, predict_date, model_x, model_y, norm_orig, norm_group
    
   
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
        
        # global model_results_raw
        model_results_raw = cur_model(remove_none=True)      
        
        
        if len(model_results_raw['RESULTS']) == 0:
            continue
        
        
        model_name = cur_model.__name__
        
        
        # Buy Signal
        temp_results = model_results_raw['RESULTS']
        temp_results['MODEL'] = model_name
        
        # RMSE ......
        new_rmse = model_results_raw['RMSE']
        new_rmse['MODEL'] = model_name
        rmse = rmse.append(new_rmse)
        
        results = results.append(temp_results)
        
    
    # buy_signal = buy_signal.rename(columns={'CLOSE':'FORECAST_CLOSE'})
    rmse = rmse.reset_index(drop=True)
    
    
    results = cbyz.df_normalize_restore(df=results, 
                                        original=norm_orig,
                                        groupby=norm_group)
    
    export_dict = {'RESULTS':results, 
                   'RMSE':rmse}
    
    
    return results, rmse




# %% Master ------

def master(_predict_begin, _predict_end=None, 
           _predict_period=15, data_period=150, 
           _stock_symbol=None, _stock_type='tw'):
    '''
    主工作區
    '''
    
    
    # data_period = 180
    # _predict_begin = 20210601
    # _predict_end = None
    # _predict_period = 15
    # _stock_type = 'tw'
    # _stock_symbol = ['2301', '2474', '1714', '2385']

    global shift_begin, shift_end, data_begin, data_end
    global predict_begin, predict_end, predict_period
    predict_period = _predict_period
    
    
    
    Bug, shift_begin - shift_end的區間應該要大於_predict_period
    add shift amount
    
    shift_begin, shift_end, \
            data_begin, data_end, predict_begin, predict_end = \
                cbyz.date_get_period(data_begin=None,
                                       data_end=None, 
                                       data_period=data_period,
                                       predict_begin=_predict_begin,
                                       predict_end=_predict_end, 
                                       predict_period=_predict_period)        
            
    
    global stock_symbol, stock_type
    
    stock_type = _stock_type
    stock_symbol = _stock_symbol
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, to_type='str')


    # ......
    data_raw = get_model_data(lag=7)
    # print(data_raw)
    
    
    global model_data, predict_date, model_x, model_y, model_addt_vars
    global norm_orig, norm_group
    model_data = data_raw['MODEL_DATA']
    predict_date = data_raw['PRECIDT_DATE']
    model_x = data_raw['MODEL_X']
    model_y = data_raw['MODEL_Y']    
    norm_orig = data_raw['NORM_ORIG']
    norm_group = data_raw['NORM_GROUP']
    model_addt_vars = ['STOCK_SYMBOL', 'WORK_DATE']
    
    
    global predict_results
    predict_results = predict()
    
    
    return predict_results




if __name__ == '__main__':
    
    # master()

    report = master(_predict_begin=20210601, _predict_end=None, 
           _predict_period=15, data_period=360, 
           _stock_symbol=['2301', '2474', '1714', '2385'])




# %% V1 Model ......


def model_1(remove_none=True):
    '''
    Linear regression
    '''
    
    from sklearn.linear_model import LinearRegression
    global stock_symbol, model_x, model_y, model_data, predict_date

    
    # Model ........
    model_info = pd.DataFrame()
    results = pd.DataFrame()
    rmse = pd.DataFrame()
    rmse_li = []
    
    for i in range(0, len(stock_symbol)):
        
        cur_symbol = stock_symbol[i]
        
        # Dataset
        X_train, X_test, y_train, y_test, X_predict = \
            split_data(symbol=cur_symbol)        
        
        try:
            reg = LinearRegression().fit(X_train, y_train)
            # reg.score(x, y)
            # reg.coef_
            # reg.intercept_
        except:
            continue


        # Test Group ......
        preds_test = reg.predict(X_test)
        
        # RMSE ......
        new_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
        rmse_li.append([cur_symbol, new_rmse])
    
    
        # predict ......      
        preds_test = reg.predict(X_predict)  
        preds = reg.predict(X_predict).tolist()
        preds = cbyz.li_join_flatten(preds)
        
        
        # Combine ......
        new_results = pd.DataFrame({'WORK_DATE':predict_date,
                                    'CLOSE':preds})
        new_results['STOCK_SYMBOL'] = cur_symbol
        results = results.append(new_results)           


    rmse = pd.DataFrame(data=rmse_li, columns=['STOCK_SYMBOL', 'RMSE'])
    
    # Reorganize ------
    if len(results) > 0:
        results = results[['WORK_DATE', 'STOCK_SYMBOL', 'CLOSE']] \
                    .reset_index(drop=True)
    
    return_dict = {'RESULTS':results,
                   'FEATURES':pd.DataFrame(),
                   'RMSE':rmse} 

    return return_dict




def model_2(remove_none=True):
    '''
    XGBoost
    https://www.datacamp.com/community/tutorials/xgboost-in-python
    '''
    
    import xgboost as xgb    
    global stock_symbol, model_x, model_y, model_data, predict_date


    # ......
    results = pd.DataFrame()
    features = pd.DataFrame()
    rmse_li = []
    
    for i in range(len(stock_symbol)):
        
        cur_symbol = stock_symbol[i]
        
        # Dataset
        X_train, X_test, y_train, y_test, X_predict = \
            split_data(symbol=cur_symbol)
        
        
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
        
        # Bug, y_test['CLOSE']裡面有NA
        new_rmse = np.sqrt(mean_squared_error(y_test['CLOSE'].tolist(),
                                              preds_test))

        # RMSE ......
        new_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
        rmse_li.append([cur_symbol, new_rmse])


        # Prediction ......
        preds = regressor.predict(X_predict)

        # Combine ......
        new_results = pd.DataFrame({'WORK_DATE':predict_date,
                                    'CLOSE':preds})
        new_results['STOCK_SYMBOL'] = cur_symbol
        results = results.append(new_results)


    rmse = pd.DataFrame(data=rmse_li, columns=['STOCK_SYMBOL', 'RMSE'])        


    if len(results) > 0:
        results = results[['WORK_DATE', 'STOCK_SYMBOL', 'CLOSE']] \
                    .reset_index(drop=True)
                    
        features = features[['STOCK_SYMBOL', 'FEATURES', 'IMPORTANCE']] \
                    .reset_index(drop=True)
                    
        rmse = rmse.reset_index(drop=True)


    return_dict = {'RESULTS':results,
                   'FEATURES':features,
                    'RMSE':rmse
                   } 

    return return_dict





