#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 20:27:39 2021

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
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/'
else:
    path = '/home/aronhack/stock_forecast/5_Notification/'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ/',
                 path + 'Function/']


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



def load_data():
    '''
    讀取資料及重新整理
    '''
    return ''



def notif_stop_loss(stock_type='tw'):
    
    stock_data_raw = stk.get_today_data(stock_type='tw')
    stock_data = stock_data_raw[['WORK_DATE', 'SECURITY_CODE', 'NAME',
                                 'OPEN_PRICE', 'CLOSE_PRICE', 
                                 'HIGH_PRICE', 'LOW_PRICE']]

    # Hold Data
    hold_data = pd.DataFrame({'SECURITY_CODE':['00646', '00762'],
                              'MEAN_COST':[30.11, 35.03]})
    

    # Merge Data
    main_data = stock_data.merge(hold_data, on=['SECURITY_CODE'])
    
    table = [main_data]
    
    ar.send_mail(to='myself20130612@gmail.com',
                 subject='ARON HACK Stock Stop Loss Notification', 
                 content='ARON HACK Stock Stop Loss Notification', 
                 df_list=table,
                 preamble='ARON HACK Stock Stop Loss Notification')
    
    return ''




def master():
    '''
    主工作區
    '''
    
    notif_stop_loss(stock_type='tw')    
    
    return ''



if __name__ == '__main__':
    master()



