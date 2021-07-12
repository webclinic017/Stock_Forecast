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





def master():
    '''
    主工作區
    '''
    
    return ''




def check():
    '''
    資料驗證
    '''    
    return ''




# if __name__ == '__main__':
#     master()




import tejapi
tejapi.ApiConfig.api_key = "22DZ20gwIuY3tfezbVnf1zjnp8cfnB"

# 斜槓方案
# tejapi.ApiConfig.api_key = 'L22pqrVRPtdVR7xY2EUGHPwEbUXJV9'


info = tejapi.ApiConfig.info()
info['todayRows']


# 系統限制單次取得最大筆數為10,000筆，可使用 paginate=True 參數分次取得資料，
# 但總筆數單次最多為1,000,000筆。請斟酌使用篩選條件降低筆數。




import datetime

# 7/13 - 檢查0712的資料是不是1772筆


def query_data():
    
    # 建議每個月分開抓，如果抓到的資料是一萬筆，但額度只剩一千，剩下的一千好像會被浪費掉
    begin = 20191101
    end = 20191130
    
    # begin = 20210712
    # end = 20210712
        
    
    begin_str = cbyz.ymd(begin)
    begin_str = begin_str.strftime('%Y-%m-%d')


    end_str = cbyz.ymd(end)
    end_str = end_str.strftime('%Y-%m-%d')    

    
    # 1個月約38000筆
    data = tejapi.get('TWN/EWTINST1C', 
                      mdate={'gte':begin_str, 'lte':end_str},
                      paginate=True)
    
    data.to_csv(path_export + '/data_' + begin_str + '_' + end_str + '.csv', 
                index=False)
    
    
    

def upload():    
    
    files = cbyz.os_get_dir_list(path=path_export, level=0, extensions='csv',
                             remove_temp=True)
    
    files = files['FILES']
    
    
    for i in range(len(files)):
    
        name = files.loc[i, 'FILE_NAME']
        file = pd.read_csv(path_export + '/' + name)

        file['mdate'] = pd.to_datetime(file.mdate)
        file['mdate'] = file['mdate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
        ar.db_upload(data=file, table_name='ewtinst1c')









