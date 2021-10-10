#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:07:03 2021

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
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/1_Data_Collection/2_TEJ'
else:
    path = '/home/aronhack/stock_forecast/dashboard/'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk




# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

# 新增工作資料夾
global path_resource, path_function, path_temp, path_export
path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])      



# %% API ------
# 系統限制單次取得最大筆數為10,000筆，可使用 paginate=True 參數分次取得資料，
# 但總筆數單次最多為1,000,000筆。請斟酌使用篩選條件降低筆數。

import tejapi
# tejapi.ApiConfig.api_key = "22DZ20gwIuY3tfezbVnf1zjnp8cfnB"

# 斜槓方案
tejapi.ApiConfig.api_key = '22DZ20gwIuY3tfezbVnf1zjnp8cfnB'
info = tejapi.ApiConfig.info()
info['todayRows']



# %% Function ------


def master():
    '''
    主工作區
    '''
    
    # Put all query in a funciton
    
    return ''





# if __name__ == '__main__':
#     master()


    



# %% Query ------


def update(ewprcd2=True, ewtinst1c=True, ewprcd=True, ewsale=True, 
           ewifinq=True, ewnprcstd=True,
           delete=False, upload=True):
    '''
    以月或季為單位的資料，篩選的時候還是用日期下條件，所以當成是d
    '''
    
    tables = []
    
    if ewtinst1c:
        # 三大法人持股成本
        tables.append(['ewtinst1c', 'd']) 
    elif ewprcd:
        # 證券交易資料表，一個月51034筆
        tables.append(['ewprcd', 'd'])  
        
    elif ewprcd2:
        # 報酬率資訊表，兩個月約100000筆        
        tables.append(['ewprcd2', 'd']) 
        
    elif ewsale:
        # 月營收資料表，一個月約2000筆，等同於股票檔數
        tables.append(['ewsale', 'd']) 
        
    elif ewifinq:
        # 單季財務資料表，資料量等同於股票檔數
        tables.append(['ewifinq', 'd'])         

    elif ewnprcstd:
        # 證券屬性表，3125筆，資料量等同於股票檔數
        tables.append(['ewnprcstd', None]) 

    # 每一天要分開query，但假日的時候不需要
    end = cbyz.date_get_today()
    begin = cbyz.date_cal(end, -5, 'd')
    
    # Manual Settings
    # begin = 20160101
    # end = 20161231
    
    begin_str = cbyz.ymd(begin)
    begin_str = begin_str.strftime('%Y-%m-%d')

    end_str = cbyz.ymd(end)
    end_str = end_str.strftime('%Y-%m-%d')    
    
    
    
    # Delete Data ......
    if delete:
        
        for i in range(len(tables)):
            
            table = tables[i][0]
            
            # Delete incomplete data.
            sql = (" delete from " + table + " "
                   " where date_format(mdate, '%Y%m%d') >= " + str(begin))
            
            ar.db_execute(sql, commit=True)    
    
    
    # Query ......
    # 系統限制單次取得最大筆數為10,000筆，可使用 paginate=True 參數分次取得資料，
    # 但總筆數單次最多為1,000,000筆。請斟酌使用篩選條件降低筆數。
    for i in range(len(tables)):
        
        table = tables[i][0]
        time_unit = tables[i][1]
        
        if time_unit == 'd':
            data = tejapi.get('TWN/' + table.upper(),
                              mdate={'gte':begin_str, 'lte':end_str},
                              paginate=True)
            
            file_path = '/' + table + '/' + table +'_'  \
                        + begin_str + '_' + end_str + '.csv'
            
        elif time_unit == None:     
            data = tejapi.get('TWN/' + table.upper(), paginate=True)
            file_path = '/' + table + '/' + table + '.csv'            


        if len(data) == 0:
            continue

        
        if 'mdate' in data.columns:     
            # ProgrammingError: Failed processing format-parameters; 
            # Python 'timestamp' cannot be converted to a MySQL type
            data['mdate'] = data['mdate'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            ar.db_upload(data=data, table_name=table)
        except Exception as e:
            print(e)
            data.to_csv(path_export + file_path, index=False)        

        # Reset Object
        data = []





def update_20211010(ewprcd2=True, ewtinst1c=True, ewprcd=True, delete=False, 
           upload=True):
    '''
    證券交易資料表
    - 一個月51034筆
    '''
    
    tables = []
    
    if ewtinst1c:
        tables.append('ewtinst1c') # 三大法人持股成本
    elif ewprcd:
        tables.append('ewprcd') # 證券交易資料表
    elif ewprcd2:
        tables.append('ewprcd2') # 報酬率資訊表


    begin = 20211006
    end = 20211006
    
    # 每一天要分開query，但假日的時候不需要
    today = cbyz.date_get_today()
    # today = 20211003
    


    begin_str = cbyz.ymd(begin)
    begin_str = begin_str.strftime('%Y-%m-%d')

    end_str = cbyz.ymd(end)
    end_str = end_str.strftime('%Y-%m-%d')    
    
    
    
    # Delete Data ......
    if delete:
        
        for i in range(len(tables)):
            table = tables[i]
            
            # Delete incomplete data.
            sql = (" delete from " + table + " "
                   " where date_format(mdate, '%Y%m%d') >= " + str(begin))
            
            ar.db_execute(sql, commit=True)    
    
    
    # Query ......
    # 系統限制單次取得最大筆數為10,000筆，可使用 paginate=True 參數分次取得資料，
    # 但總筆數單次最多為1,000,000筆。請斟酌使用篩選條件降低筆數。
    for i in range(len(tables)):
        
        table = tables[i]
        data = tejapi.get('TWN/' + table.upper(),
                          mdate={'gte':begin_str, 'lte':end_str},
                          paginate=True)
    
        data.to_csv(path_export + '/' + table + '/' + table +'_' \
                    + begin_str + '_' + end_str + '.csv', 
                    index=False)


    # Upload ......
    # Update, auto delete the latest file
    if upload:
        
        for i in range(len(tables)):
            
            table = tables[i]
            
            # Get file list
            file_path = path_export + '/' + table
            files = cbyz.os_get_dir_list(path=file_path, level=0, extensions='csv',
                                     remove_temp=True)

            files = files['FILES']
        
        
            for j in range(len(files)):
                
                name = files.loc[j, 'FILE_NAME']
                file = pd.read_csv(file_path + '/' + name)
        
                file['mdate'] = pd.to_datetime(file.mdate)
                file['mdate'] = file['mdate'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
                ar.db_upload(data=file, table_name=table)




def check():
    '''
    Check historical data impocomlete
    '''
        
    file = pd.read_csv(path_export + '/ewiprcd/ewiprcd_data_2019-01-01_2020-12-31.csv')
    file2 = pd.read_csv(path_export + '/ewiprcd/ewiprcd_data_2021-01-01_2021-07-13.csv')
    
    final = file.append(file2)
    final.to_csv(path_export + '/ewiprcd/ewiprcd_data_2019-01-01_2021-07-13.csv', index=False)
    
    
