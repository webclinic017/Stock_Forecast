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
import h5py


# 設定工作目錄 .....

local = False
local = True


dev = False
dev = True



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
import arsenal_stock as stk
import codebase_yz as cbyz


path_temp = path + '/Temp'
cbyz.os_create_folder(path=[path_temp])



# 手動設定區 -------
global begin_date, end_date
end_date = datetime.datetime.now()
end_date = cbyz.date_simplify(end_date)

begin_date_6m = cbyz.date_cal(end_date, amount=-12, unit='m')
begin_date_3y = cbyz.date_cal(end_date, amount=-3, unit='y')



stock_type = 'us'
stock_type = 'tw'


# 自動設定區 -------
pd.set_option('display.max_columns', 30)




def load_data():
    '''
    讀取資料及重新整理
    '''

    # Historical Data .....
    global main_data, main_data_lite
    
    if 'main_data_lite' not in globals():
        
        if dev:
            begin_date_3y = cbyz.date_cal(end_date, -60, 'd')
            main_data = stk.get_data(data_begin=begin_date_3y, 
                                     data_end=end_date, shift=0,
                                     stock_type=stock_type, 
                                     stock_symbol=[], 
                                     local=local)
            
        else:
            main_data = stk.get_data(data_begin=begin_date_3y, 
                                     data_end=end_date, shift=0,
                                     stock_type=stock_type, 
                                     stock_symbol=[], 
                                     local=local)           
    
        main_data = main_data \
                    .sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                    .reset_index(drop=True)
        
        
        main_data['WORK_DATE'] = main_data['WORK_DATE'].apply(ar.ymd)
        main_data_lite = main_data[main_data['WORK_DATE']>=ar.ymd(begin_date_6m)]
    

    
    
    # Stock List ......
    global stock_list_raw, stock_list
    

    if 'stock_list' not in globals():
        
        # Bug, fix this
        stock_list_raw = stk.tw_get_company_info(export_file=True, 
                                             load_file=True, 
                                             file_name=None, 
                                             path=path_temp)
    
        stock_list_raw = stock_list_raw[['STOCK_SYMBOL', 'COMPANY_NAME']]
        
        stock_list_raw['STOCK'] = stock_list_raw['STOCK_SYMBOL'].astype(str) \
                                + ' ' + stock_list_raw['COMPANY_NAME']
                        
        stock_list = []
        for i in range(0, len(stock_list_raw)):
            stock_list.append({'label': stock_list_raw.loc[i, 'STOCK'],
                                'value': stock_list_raw.loc[i, 'STOCK_SYMBOL']
                                })
        


def master():
    '''
    主工作區
    '''
    load_data()
    return ''



master()





# %% Dash ----------

# app = dash.Dash(__name__, suppress_callback_exceptions=True)


# if __name__ == '__main__':
#     server = app.server