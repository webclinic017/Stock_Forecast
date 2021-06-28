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
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/2_Stock_Analysis'
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
                  stock_symbol=None, full_data=False):
    '''
    讀取資料及重新整理
    '''
    
    # Shift one day forward to get complete PRICE_CHANGE_RATIO
    loc_begin = cbyz.date_cal(data_begin, -1, 'd')
    
    
    
    if full_data:
        data = stk.get_data(data_begin=loc_begin, 
                                 data_end=data_end, 
                                 stock_type=stock_type, 
                                 stock_symbol=None, 
                                 price_change=True,
                                 local=local)
    else:
        data = stk.get_data(data_begin=loc_begin, 
                                 data_end=data_end, 
                                 stock_type=stock_type, 
                                 stock_symbol=stock_symbol, 
                                 price_change=True,                                 
                                 local=local)
        
        
    # Exclude the symbols shorter than begin_date ......
    date_min = data['WORK_DATE'].min()
    data['MIN_DATE'] = data \
                        .groupby(['STOCK_SYMBOL'])['WORK_DATE'] \
                        .transform('min')

    data = data[data['MIN_DATE']==date_min] \
            .drop('MIN_DATE', axis=1)


    # Merge Other Data ......        
    if full_data:        
        
        # Stock Info ...
        stock_info = stk.tw_get_stock_info(export_file=True, load_file=True, 
                                           file_name=None, path=path_temp)
        
        stock_info = stock_info[['STOCK_SYMBOL', 'CAPITAL_LEVEL',
                                 'ESTABLISH_DAYS', 'LISTING_DAYS', 
                                 'INDUSTRY_ONE_HOT']]
        
        stock_info = cbyz.df_get_dummies(df=stock_info, 
                                         cols='INDUSTRY_ONE_HOT')
        
        # Merge Data ......
        data = data.merge(stock_info, how='left', on='STOCK_SYMBOL')
        data = data[~data['ESTABLISH_DAYS'].isna()].reset_index(drop=True)
    
    
    return data



# ..............



def get_model_data(ma_values=[5,20]):
    
    
    global shift_begin, shift_end, data_begin, data_end, full_data
    global predict_begin, predict_end, predict_period
    global stock_symbol
    
    
    # ......
    stock_symbol = cbyz.conv_to_list(stock_symbol)
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, 'str')


    # Date ......
    predict_date = cbyz.date_get_seq(begin_date=predict_begin,
                                     seq_length=predict_period,
                                     interval=1,
                                     unit='d', simplify_date=True,
                                     ascending=True)
        
    predict_date = predict_date[['WORK_DATE']]
    predict_date_list = predict_date['WORK_DATE'].tolist()
    
    
    # Load Historical Data ......
    loc_data = sam_load_data(data_begin=shift_begin,
                                 data_end=data_end,
                                 stock_symbol=stock_symbol, 
                                 full_data=full_data) 
    
    loc_data = loc_data.sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                .reset_index(drop=True)

    # Add K line
    loc_data = stk.add_k_line(loc_data)
    

    # Predict Symbols ......
    if full_data:
        all_symbols = loc_data['STOCK_SYMBOL'].unique().tolist()
        stock_symbol_df = pd.DataFrame({'STOCK_SYMBOL':all_symbols})
    
    else:
        stock_symbol_df = pd.DataFrame({'STOCK_SYMBOL':stock_symbol})
        
    predict_df = cbyz.df_cross_join(stock_symbol_df, predict_date)

     
    # Calendar ......
    calendar = ar.get_calendar(simplify=True)
    calendar = calendar.drop('WEEK_ID', axis=1)
    
    
    # Add predict Data ......
    loc_main = loc_data.merge(predict_df, how='outer',
                                    on=['STOCK_SYMBOL', 'WORK_DATE'])
    
    loc_main = loc_main \
                        .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                        .reset_index(drop=True)
    
    loc_main = loc_main.merge(calendar, how='left', 
                                          on='WORK_DATE')
    
    
    # One Hot Encoding ......
    data_types = loc_main.dtypes
    data_types = pd.DataFrame(data_types, columns=['TYPE']).reset_index()
    
    obj_cols = data_types[(data_types['TYPE']=='object') \
                            & (data_types['index']!='STOCK_SYMBOL')]
        
    obj_cols = obj_cols['index'].tolist()
    
    # Assign columns munually
    obj_cols = obj_cols + ['K_LINE_TYPE']
    loc_main = cbyz.df_get_dummies(loc_main, cols=obj_cols, 
                                         expand_col_name=True)    
    
    # Shift ......
    shift_cols = \
        cbyz.df_get_cols_except(df=loc_main,
                                except_cols=['WORK_DATE', 'STOCK_SYMBOL',
                                             'YEAR', 'MONTH', 'WEEKDAY', 
                                             'WEEK_NUM'])
         

    loc_main, ma_cols = stk.add_ma(df=loc_main, cols=shift_cols, 
                                   key=['STOCK_SYMBOL'], 
                          date='WORK_DATE', values=ma_values)
    

    if predict_period > min(ma_values):
        # Update, raise error here
        print('get_model_data - predict_period is longer than ma values, ' \
              + 'and it will cause na.')
        del loc_main

    
    # Add lag, or there will be na in the predict period
    loc_main, lag_cols = cbyz.df_add_shift(df=loc_main, 
                                           cols=ma_cols, 
                                           shift=predict_period,
                                           group_by=['STOCK_SYMBOL'],
                                           suffix='_LAG', 
                                           remove_na=False)
    
    var_cols = ['MONTH', 'WEEKDAY', 'WEEK_NUM']
    model_x = lag_cols + var_cols
    model_y = ['HIGH', 'LOW', 'CLOSE']

    
    # Model Data ......
    # loc_model_data = loc_data_shift.dropna(subset=model_x)
    loc_main = loc_main[loc_main['WORK_DATE']>=data_begin] \
                        .reset_index(drop=True)


    # Remove all data with na values ......
    na_df = loc_main[model_x + ['STOCK_SYMBOL']]
    na_df = na_df[na_df.isna().any(axis=1)]
    symbols_removed = na_df['STOCK_SYMBOL'].unique().tolist()
    loc_main = loc_main[~loc_main['STOCK_SYMBOL'].isin(symbols_removed)] \
                .reset_index(drop=True)


    # Check - X裡面不應該有na，但Y的預測區間會是na ......
    chk = loc_main[model_x]
    chk_na = cbyz.df_chk_col_na(df=chk, positive_only=True, return_obj=True,
                                alert=True, alert_obj='loc_main')
    
    
    # 由於有些股票剛上市，或是有特殊原因，導致資料不齊全，全部排除處理
    # if full_data:
    #     na_col = chk_na \
    #                 .sort_values(by='NA_COUNT', ascending=False) \
    #                 .reset_index(drop=True)
                    
    #     na_col = na_col.loc[0, 'COLUMN']
    #     symbols_removed = loc_main[loc_main[na_col].isna()]
    #     symbols_removed = symbols_removed['STOCK_SYMBOL'].unique().tolist()
        
    #     loc_main = loc_main[~loc_main['STOCK_SYMBOL'].isin(symbols_removed)] \
    #                 .reset_index(drop=True)

    
    
    # Normalize ......
    norm_cols = cbyz.li_join_flatten(model_x, model_y) 
    loc_main_norm = cbyz.df_normalize(df=loc_main,
                                      cols=norm_cols,
                                      groupby=['STOCK_SYMBOL'],
                                      show_progress=True)
    
    loc_model_data_raw = loc_main_norm[0]
    
    identify_cols = ['STOCK_SYMBOL', 'WORK_DATE']
    loc_model_data = loc_model_data_raw[norm_cols + identify_cols] \
                        .dropna(subset=model_x)
            
            
    if len(loc_model_data_raw) != len(loc_model_data):
        print('Err01. get_model_data - the length of loc_model_data_raw and ' \
              + 'loc_model_data are different.' )
        del loc_model_data
    

    export_dict = {'MODEL_DATA_RAW':loc_model_data_raw,
                   'MODEL_DATA':loc_model_data,
                   # 'PRECIDT_DATA':predict_data,
                   'PRECIDT_DATE':predict_date_list,
                   'MODEL_X':model_x,
                   'MODEL_Y':model_y,
                   'DATA_BEGIN':data_begin,
                   'DATA_END':data_end,
                   'NORM_ORIG':loc_main_norm[1],
                   'NORM_GROUP':loc_main_norm[2]}
    
    return export_dict



# ...............



def split_data(symbol=None):
    
    global model_data, model_x, model_y, model_addt_vars, predict_date
    global stock_symbol
    
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
        split_data(symbol=None)
            
        regressor = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
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
    
    function_list = [model_5, model_6]

    return function_list


# ...............


def predict():
    
    global shift_begin, shift_end, data_begin, data_end
    global predict_begin, predict_end    
    global model_data, predict_date, model_x, model_y, norm_orig, norm_group
    
   
    split_data(symbol=None)
    
   
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
    
    results_pivot = results \
                    .pivot_table(index=['STOCK_SYMBOL', 'WORK_DATE', 'MODEL'],
                                 columns='Y',
                                 values='VALUES') \
                    .reset_index()
    
    results = cbyz.df_normalize_restore(df=results_pivot, 
                                        original=norm_orig,
                                        groupby=norm_group)
    
    return results, rmse




# %% Master ------

def master(_predict_begin, _predict_end=None, 
           _predict_period=15, data_period=180, 
           _stock_symbol=None, _stock_type='tw', ma_values=[5,20],
           _full_data=False):
    '''
    主工作區
    '''
    
    # date_period為10年的時候會出錯
    
    # data_period = 90
    # data_period = 365 * 5
    # _predict_begin = 20210611
    # _predict_end = None
    # _predict_period = 5
    # _stock_type = 'tw'
    # ma_values = [5,20]
    # _full_data = False
    # _stock_symbol = ['2301', '2474', '1714', '2385']
    


    # target_symbols = pd.read_csv(path_export \
    #                               + '/target_symbols_20210624_212851.csv')

    # _stock_symbol = target_symbols['STOCK_SYMBOL'].tolist()


    # Process ......
    # 1. Full data to select symbols



    # Worklist .....
    # 移動平均線加權weight?
    # rmse by model and symbol?
    # 建長期投資的基本面模型
    

    global shift_begin, shift_end, data_begin, data_end, full_data
    global predict_begin, predict_end, predict_period
    
    predict_period = _predict_period
    data_shift = -(max(ma_values) * 3)
    full_data = _full_data
    
    
    shift_begin, shift_end, \
            data_begin, data_end, predict_begin, predict_end = \
                cbyz.date_get_period(data_begin=None,
                                       data_end=None, 
                                       data_period=data_period,
                                       predict_begin=_predict_begin,
                                       predict_end=_predict_end, 
                                       predict_period=predict_period,
                                       shift=data_shift)  
            
    
    global stock_symbol, stock_type
    stock_type = _stock_type
    stock_symbol = _stock_symbol
    stock_symbol = cbyz.li_conv_ele_type(stock_symbol, to_type='str')


    # ......
    data_raw = get_model_data(ma_values=ma_values)
    
    
    global model_data_raw, model_data, predict_date
    global model_x, model_y, model_addt_vars
    global norm_orig, norm_group
    
    model_data_raw = data_raw['MODEL_DATA_RAW']
    model_data = data_raw['MODEL_DATA']
    predict_date = data_raw['PRECIDT_DATE']
    model_x = data_raw['MODEL_X']
    model_y = data_raw['MODEL_Y']    
    norm_orig = data_raw['NORM_ORIG']
    norm_group = data_raw['NORM_GROUP']
    model_addt_vars = ['STOCK_SYMBOL', 'WORK_DATE']
    
    
    global predict_results
    predict_results = predict()
    predict_results
    
    
    print('sam master - predict_begin + ' + str(_predict_begin))    
    
    return predict_results




if __name__ == '__main__':
    
    # master()

    report = master(_predict_begin=20210601, _predict_end=None, 
           _predict_period=15, data_period=360, 
           _stock_symbol=['2301', '2474', '1714', '2385'])




def check():
    
    chk = cbyz.df_chk_col_na(df=model_data_raw)    
    chk = cbyz.df_chk_col_na(df=model_data)

    # Err01
    chk = loc_main[model_x]
    chk_na = cbyz.df_chk_col_na(df=chk, positive_only=True, return_obj=True,
                                alert=True, alert_obj='loc_main')
    
    chk = loc_main[loc_main['OPEN_MA_20_LAG'].isna()]
    



# %% Stock Selection ------


def find_target(data_begin, data_end):


    # Select Rules
    # 1. 先找百元以下的，才有資金可以買一整張
    # 2. 不要找疫情後才爆漲到歷史新高的


    data_end = 20210625       
    data_begin = cbyz.date_cal(data_end, -1, 'm')


    # Stock info
    stock_info = stk.tw_get_stock_info()
    
    
    # 挑選中大型股 ......
    level3_symbom = stock_info[stock_info['CAPITAL_LEVEL']>=2]
    level3_symbom = level3_symbom['STOCK_SYMBOL'].tolist()
    

    
    # Stock Data ......
    data_raw = stk.get_data(data_begin=data_begin, data_end=data_end, 
                            stock_symbol=level3_symbom, 
                            price_change=True,
                            shift=0, stock_type='tw', local=True)
    
    data = data_raw[data_raw['STOCK_SYMBOL'].isin(level3_symbom)]
    
    
    
    
    
    # Method 1 - 低價股全篩 .....
    target_symbols = data[data['CLOSE']<=80]
    target_symbols = target_symbols[['STOCK_SYMBOL']].drop_duplicates()
    
    
    
    # Method 2 .....
    # Calculation ......
    # 3天漲超過10%
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

    target_symbols.to_excel(path_export + '/target_symbols_' \
                            + time_serial + '.xlsx',
                            index=False)

    # Plot ......       
    # plot_data = results.melt(id_vars='PROFIT')

    # cbyz.plotly(df=plot_data, x='PROFIT', y='value', groupby='variable', 
    #             title="", xaxes="", yaxes="", mode=1)

    
    return results_raw, stock_info




