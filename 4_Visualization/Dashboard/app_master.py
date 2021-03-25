#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:09:54 2021

@author: Aron
"""


import os

import pandas as pd
import sys, arrow
import datetime
import dash


# 設定工作目錄 .....

local = False
local = True



if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/4_Visualization/Dashboard'
else:
    path = '/home/aronhack/stock_forecast/4_Visualization/Dashboard'
    


# Codebase
path_codebase = ['/Users/Aron/Documents/GitHub/Arsenal',
                 '/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path, 
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import arsenal as ar
import codebase_yz as cbyz



# 手動設定區 -------
global begin_date, end_date
end_date = datetime.datetime.now()
end_date = cbyz.date_simplify(end_date)

begin_date_6m = cbyz.date_cal(end_date, amount=-12, unit='m')
# begin_date_3y = cbyz.date_cal(end_date, amount=-3, unit='y')
begin_date_3y = cbyz.date_cal(end_date, amount=-13, unit='m') # Dev



stock_type = 'us'
stock_type = 'tw'


# 自動設定區 -------
pd.set_option('display.max_columns', 30)




def load_data():
    '''
    讀取資料及重新整理
    '''

    # Load Data --------------------------
    
    # Historical Data .....
    stock_data = get_stock_data(begin_date_3y, end_date, 
                                stock_type=stock_type)
    
    
    # Stock Name .....
    global stock_name
    stock_name = ar.stk_get_list(stock_type=stock_type)



    # Work Area -------------
    global main_data, main_data_lite
    
    main_data = stock_data.copy()
    main_data = main_data.sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                .reset_index(drop=True)
    
    
    main_data = main_data.merge(stock_name, how='left', 
                                on=['STOCK_SYMBOL'])
    
    # Some stock not in stock_name
    # 00776, 910482
    main_data = cbyz.df_conv_na(df=main_data, cols='STOCK_NAME', value='')
    
    
    main_data['TYPE'] = 'HISTORICAL'
    main_data['WORK_DATE'] = main_data['WORK_DATE'].apply(ar.ymd)
    
    main_data['STOCK'] = (main_data['STOCK_SYMBOL'] 
                          + ' ' 
                          + main_data['STOCK_NAME'])
    
    main_data_lite = main_data[main_data['WORK_DATE']>=ar.ymd(begin_date_6m)]
    
    
    global target_data, target_symbol
    target_symbol = []
    target_data = pd.DataFrame()
    
    
    # Dash ----------------------
    
    # Stock List
    global stock_list, stock_list_pre
    stock_list_pre = main_data[['STOCK_SYMBOL', 'STOCK']] \
                        .drop_duplicates() \
                        .reset_index(drop=True)
                    
                    
    stock_list = []
    for i in range(0, len(stock_list_pre)):
        stock_list.append({'label': stock_list_pre.loc[i, 'STOCK'],
                           'value': stock_list_pre.loc[i, 'STOCK_SYMBOL']
                           })
        
    return ''


# %% Inner Function -------


def get_stock_data(begin_date=None, end_date=None, 
                   stock_type='tw', stock_symbol=None):
    
    
    if stock_type == 'tw':
        stock_tb = 'stock_data_tw'
    elif stock_type == 'us':
        stock_tb = 'stock_data_us'
    

    # Convert stock to list
    stock_li = [stock_symbol]
    stock_li = cbyz.list_flatten(stock_li)    
    
    
    if (begin_date != None) & (end_date != None):
        
        if stock_symbol != None:
            sql_stock = cbyz.list_add_quote(stock_li, "'")
            sql_stock = ', '.join(sql_stock)
            sql_stock = ' and stock_symbol in (' + sql_stock + ')'
        else:
            sql_stock = ''
            
        sql_cond = (" where date_format(work_date, '%Y%m%d') between " + str(begin_date) + " and " + str(end_date) +
                    sql_stock)
        
    else:
        sql_stock = cbyz.list_add_quote(stock_li, "'")
        sql_stock = ', '.join(sql_stock)
        sql_cond = ' where stock_symbol in (' + sql_stock + ')'
    

    sql = ( "select date_format(work_date, '%Y%m%d') work_date, " 
    + " stock_symbol, high, close, low "
    + " from " + stock_tb + " "
    + sql_cond
    )
    
        
    results = ar.db_query(sql, local=local)
    return results



def master():
    '''
    主工作區
    '''
    load_data()
    return ''


master()





# %% Dash ----------

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server