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
# local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/5_Notification'
else:
    path = '/home/aronhack/stock_forecast/5_Notification'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal',
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




def get_ledger():
    
    url ='https://docs.google.com/spreadsheets/d/1xEZ7khMMNstzrGMPhTNPmkyepsyXKtdeYAAl6txdCCk/edit?usp=sharing'
    ledger = ar.google_get_sheet_data(url, worksheet='Ledger')
    ledger = cbyz.df_conv_col_type(df=ledger, cols=['VOLUME', 'PRICE'],
                                   to='float')

    cond = ledger['TYPE']=='Sell'
    ledger['PRICE'] = np.where(cond, -ledger['PRICE'], ledger['PRICE'])
    ledger['VOLUME'] = np.where(cond, -ledger['VOLUME'], ledger['VOLUME'])
   

    ledger = ledger \
                .groupby(['MARKET', 'SYMBOL']) \
                .agg({'VOLUME':'sum',
                      'PRICE':'sum',
                      'DATE':'min'}) \
                .reset_index() \
                .rename(columns={'DATE':'FIRST_PURCHASE'})
                
    ledger = ledger[ledger['VOLUME']>0].reset_index(drop=True)
    ledger['MEAN_PRICE'] = ledger['PRICE'] / ledger['VOLUME']
    return ledger



def notif_stop_loss_backup_20210510(stock_type='tw'):
    
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






def notif_stop_loss_backup_20210520(stock_type='tw'):
    
    # Hold Data
    hold_symbols = ['00646', '00762']
    hold_data = pd.DataFrame({'STOCK_SYMBOL':hold_symbols,
                              'MEAN_COST':[30.11, 35.03],
                                # 'FIRST_PURCHASE':[20190828, 20191105]})                              
                                'FIRST_PURCHASE':[20200828, 20201105]})

    main_data = stk.get_stop_loss(df=hold_data, date='FIRST_PURCHASE', 
                              price='MEAN_COSE', thld=0.2)

        
    content_li = []
    
    for i in range(len(hold_symbols)):
        
        temp_data = main_data[main_data['STOCK_SYMBOL']==hold_symbols[i]]
        temp_data = temp_data.T
        content_li.append(temp_data)
        print(i)
    
    
    # table = [main_data]
    
    ar.send_mail(to='myself20130612@gmail.com',
                 subject='ARON HACK Stock Stop Loss Notification', 
                  content='ARON HACK Stock Stop Loss Notification', 
                  df_list=content_li,
                 preamble='ARON HACK Stock Stop Loss Notification')
    
    return ''




def get_hist_price():
    
    url ='https://docs.google.com/spreadsheets/d/1xEZ7khMMNstzrGMPhTNPmkyepsyXKtdeYAAl6txdCCk/edit?usp=sharing'
    hist_data = ar.google_get_sheet_data(url, worksheet='Hist_Data')
    hist_data = cbyz.df_conv_col_type(df=hist_data, cols=['HIST_HIGH'], 
                                      to='float')
    
    hist_data = hist_data[['MARKET', 'SYMBOL', 'HIST_HIGH']]
    
    return hist_data



def update_hist_price():
    
    
    # get_ledger()
    # crypto
    # tw_stock
    
    
    # if len(crypto) > 0:
    #     update
    
    
    # fugle api
    # crypto api
    
    return ''




def notif_stop_loss():
    
    
    # Get current price
    # TW stock stop when weekend or holiday
    # 因為crypto free api doesn't provide historical data, so it should be 
    # record in the google sheet
    
    # ADD STOCK NAME
    ledger = get_ledger()

    
    # Historical Price
    hist_price = get_hist_price()
    
    # Merge
    main_data = ledger.merge(hist_price, how='left', on=['MARKET', 'SYMBOL'])
    main_data = stk.cal_stop_loss(df=main_data, mean_price='MEAN_PRICE', 
                                  hist_high='HIST_HIGH', thld=0.2)
    
    # main_data['ACTION'] = \
    #     np.where(main_data['CUR_PRICE'] <= main_data['STOP_LOSS'])
    
    
    # Reorder Columns
    main_data = main_data[['MARKET', 'SYMBOL', 'FIRST_PURCHASE', 'VOLUME', 
                           'PRICE', 'MEAN_PRICE', 'STOP_LOSS']]
    
    # Reorganize Data
    content_li = []
    for i in range(len(main_data)):
        
        temp_data = main_data[main_data.index==i]
        temp_data = temp_data.T
        temp_data.columns = ['']
        
        content_li.append(temp_data)
        print(i)
    
    
    ar.send_mail(to='myself20130612@gmail.com',
                 subject='ARON HACK Stock Stop Loss Notification', 
                  content='ARON HACK Stock Stop Loss Notification', 
                  df_list=content_li,
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




