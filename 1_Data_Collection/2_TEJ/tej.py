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
# 7/15 - 8/14之間使用
tejapi.ApiConfig.api_key = 'FEuiQV1vbtVb51LqN2Th1A5HVDMx2R'
info = tejapi.ApiConfig.info()
info['todayRows']



# %% Function ------


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




    

def upload():    
    
    table = 'ewtinst1c' # 三大法人持股成本
    table = 'ewtinst1c'
    
    file_path = path_export + '/' + table
    files = cbyz.os_get_dir_list(path=file_path, level=0, extensions='csv',
                             remove_temp=True)
    
    files = files['FILES']
    
    
    for i in range(len(files)):
    
        name = files.loc[i, 'FILE_NAME']
        file = pd.read_csv(file_path + '/' + name)

        file['mdate'] = pd.to_datetime(file.mdate)
        file['mdate'] = file['mdate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
        ar.db_upload(data=file, table_name=table)




# %% Query ------


def query_ewtinst1c():

    '''
    三大法人持股成本
    '''    

    # 不要太早抓資料，因為TEJ的資料可能會變

    
    # 建議每個月分開抓，如果抓到的資料是一萬筆，但額度只剩一千，剩下的一千好像會被浪費掉
    begin = 20180101
    end = 20180531
    
    
    # Now
    # 0722刪掉重上傳
    begin = 20210802
    end = 20210802
        
    
    begin_str = cbyz.ymd(begin)
    begin_str = begin_str.strftime('%Y-%m-%d')


    end_str = cbyz.ymd(end)
    end_str = end_str.strftime('%Y-%m-%d')    

    
    # 1個月約38000筆
    # 20210715 - 每日1775筆才是對的；21:35只有抓到1772筆資料
    data = tejapi.get('TWN/EWTINST1C', 
                      mdate={'gte':begin_str, 'lte':end_str},
                      paginate=True)
    
    data.to_csv(path_export + '/ewtinst1c/ewtinst1c_data_' + begin_str + '_' + end_str + '.csv', 
                index=False)
    
    
    # Delete incomplete data.
    sql = (" delete from ewtinst1c "
           " where date_format(mdate, '%Y%m%d') >= " + str(begin))
    
    ar.db_execute(sql, commit=True)
 

# ...............
    

def query_ewprcd():
    '''
    證券交易資料表
    - 一個月51034筆
    '''
    
    table_name = 'ewprcd'

    begin = 20180101
    end = 20180831
    
    
    begin_str = cbyz.ymd(begin)
    begin_str = begin_str.strftime('%Y-%m-%d')


    end_str = cbyz.ymd(end)
    end_str = end_str.strftime('%Y-%m-%d')    
    
    # 2崇4541筆
    # 系統限制單次取得最大筆數為10,000筆，可使用 paginate=True 參數分次取得資料，但總筆數單次最多為1,000,000筆。請斟酌使用篩選條件降低筆數。
    data = tejapi.get('TWN/EWPRCD',
                      mdate={'gte':begin_str, 'lte':end_str},
                      paginate=True)


    data.to_csv(path_export + '/' + table_name + '/' + table_name +'_' \
                + begin_str + '_' + end_str + '.csv', 
                index=False)





def query_ewifinq():
    '''
    財務報表，一年7779筆
    '''
    
    table_name = 'ewifinq'

    begin = 20190101
    end = 20191231
    
    
    begin_str = cbyz.ymd(begin)
    begin_str = begin_str.strftime('%Y-%m-%d')


    end_str = cbyz.ymd(end)
    end_str = end_str.strftime('%Y-%m-%d')    
    
    
    data = tejapi.get('TWN/EWIFINQ',
                      mdate={'gte':begin_str, 'lte':end_str},
                      paginate=True)


    data.to_csv(path_export + '/ewifinq/ewifinq_data_' \
                + begin_str + '_' + end_str + '.csv', 
                index=False)




def query_ewsale():
    '''
    月營收資料表，一個月2112筆資料。
    '''
    
    table_name = 'ewsale'

    begin = 20180101
    end = 20190531
    
    
    begin_str = cbyz.ymd(begin)
    begin_str = begin_str.strftime('%Y-%m-%d')


    end_str = cbyz.ymd(end)
    end_str = end_str.strftime('%Y-%m-%d')    
    
    
    data = tejapi.get('TWN/EWSALE',
                      mdate={'gte':begin_str, 'lte':end_str},
                      paginate=True)


    data.to_csv(path_export + '/ewsale/ewsale_data_' \
                + begin_str + '_' + end_str + '.csv', 
                index=False)






def query_ewiprcd():
    '''
    指數資料，不好用
    '''

    begin = 20190101
    end = 20201231
    
    begin_str = cbyz.ymd(begin)
    begin_str = begin_str.strftime('%Y-%m-%d')


    end_str = cbyz.ymd(end)
    end_str = end_str.strftime('%Y-%m-%d')    
    
    data = tejapi.get('TWN/EWIPRCD',
                      mdate={'gte':begin_str, 'lte':end_str})


    data.to_csv(path_export + '/ewiprcd_data_' + begin_str + '_' + end_str + '.csv', 
                index=False)
    
    
    



def query_index2():
    '''
    指數資料
    '''
    
    # 證券屬性表 - 3210筆 
    # data = tejapi.get('TWN/EWNPRCSTD')
    # data.to_csv(path_export + '/ewnprcstd_data.csv', index=False,
    #             encoding='utf-8-sig')


    # 指數屬性表 - 115筆 
    # data = tejapi.get('TWN/EWIPRCSTD')
    # data.to_csv(path_export + '/ewiprcstd_data.csv', index=False,
    #             encoding='utf-8-sig')







def check():
    
    file = pd.read_csv(path_export + '/ewiprcd/ewiprcd_data_2019-01-01_2020-12-31.csv')
    file2 = pd.read_csv(path_export + '/ewiprcd/ewiprcd_data_2021-01-01_2021-07-13.csv')
    
    final = file.append(file2)
    final.to_csv(path_export + '/ewiprcd/ewiprcd_data_2019-01-01_2021-07-13.csv', index=False)
    
    
