#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 20:27:39 2021

@author: Aron
"""

# Worklist
# 1. 當交易完台股後，當天也要馬上記錄最高價




# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc


local = False
local = True


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



#
url ='https://docs.google.com/spreadsheets/d/1xEZ7khMMNstzrGMPhTNPmkyepsyXKtdeYAAl6txdCCk/edit?usp=sharing'


# %% Query -------


def get_hist_data():
    
    url ='https://docs.google.com/spreadsheets/d/1xEZ7khMMNstzrGMPhTNPmkyepsyXKtdeYAAl6txdCCk/edit?usp=sharing'
    hist_data = ar.google_get_sheet_data(url, worksheet='Hist_Data')
    hist_data = cbyz.df_conv_col_type(df=hist_data, cols=['HIST_HIGH'], 
                                      to='float')
    
    hist_data = hist_data[['MARKET', 'ID', 'SYMBOL', 'HIST_HIGH']]
    hist_data.loc[hist_data['ID']=='', 'ID'] = hist_data['SYMBOL']
    hist_data = cbyz.df_conv_col_type(df=hist_data, cols='ID', to='str')
    
    return hist_data





def get_ledger():
    
    ledger = ar.google_get_sheet_data(url, worksheet='Ledger')
    ledger = cbyz.df_conv_col_type(df=ledger, cols=['VOLUME', 'COST'],
                                   to='float')

    cond = ledger['TYPE']=='Sell'
    ledger['COST'] = np.where(cond, -ledger['COST'], ledger['COST'])
    ledger['VOLUME'] = np.where(cond, -ledger['VOLUME'], ledger['VOLUME'])
   

    ledger = ledger \
                .groupby(['MARKET', 'SYMBOL']) \
                .agg({'COST':'sum',
                      'VOLUME':'sum',
                      'DATE':'min'}) \
                .reset_index() \
                .rename(columns={'DATE':'FIRST_PURCHASE'})
                
    ledger = ledger[(ledger['VOLUME']>0) & (ledger['COST']>10)] \
                .reset_index(drop=True)
                
    ledger['MEAN_COST'] = ledger['COST'] / ledger['VOLUME']
    
    
    # Hold Days ......
    ledger = cbyz.df_ymd(df=ledger, cols='FIRST_PURCHASE')
    ledger['TODAY'] = cbyz.date_get_today(simplify=True)
    ledger = cbyz.df_date_diff(df=ledger, col1='TODAY', col2='FIRST_PURCHASE')
    
    ledger = ledger.rename(columns={'DAYS':'HOLD_DAYS'})
    ledger = ledger.drop(['FIRST_PURCHASE', 'TODAY'], axis=1)
    
    return ledger


# %% Operation ------


def update_hist_data():
    
    # Update, write a load_data(), then set ledger as a global vars,
    # preventing from call API repeatedly.
    ledger = get_ledger()
    ledger = ledger[['MARKET', 'SYMBOL']]
    
    # Historical Data
    hist_data = get_hist_data()
    
    
    # TW Stock ......
    # tw_stock = ledger[ledger['MARKET']=='TW_Stock'].reset_index(drop=True)
    # tw_stock_main
    
    
    # Crypto ......
    crypto = ledger[ledger['MARKET']=='Crypto'].reset_index(drop=True)
    crypto_symbol = crypto['SYMBOL'].unique().tolist()
    
    # Query from API
    crypto_price_raw = stk.crypto_get_price()
    
    
    global crypto_price
    crypto_price = \
        crypto_price_raw[crypto_price_raw['SYMBOL'].isin(crypto_symbol)]
    
    crypto_price = crypto_price[['ID', 'SYMBOL', 'PRICE', 'LAST_UPDATED']] \
                    .reset_index(drop=True)

    crypto_price = cbyz.df_conv_col_type(df=crypto_price, cols='ID', to='str')

                    
    
    # Merge data
    crypto_main = crypto \
        .merge(hist_data, how='left', on=['MARKET', 'SYMBOL']) \
        .merge(crypto_price, how='left', on=['SYMBOL'])
    
    crypto_main = crypto_main[crypto_main['PRICE']>crypto_main['HIST_HIGH']] \
                    .reset_index(drop=True)
    

    # Merget Data of All Markets ......
    main_data = crypto_main.copy()
    # main_data = crypto_main.append(tw_stock_main)
    
    # Upload ......
    ar.google_sheet_write(obj=main_data, url=url, worksheet='Hist_Data',
                          append=True)
    
    # Log ......
    log_time = cbyz.get_time_serial(with_time=True)
    
    ar.google_sheet_write(obj=[[log_time]], url=url, 
                          worksheet='Hist_Data_Log', append=True)    
    
    print('update_hist_data finished.')




def notif_stop_loss():
    
    
    # Update 
    update_hist_data()
    
    
    # 1.Get current price
    #   According to CryptoMarketCap, the best practice is to use their
    #   symbol ID rather than symbl name, because symbol name may be duplicated.
    #   The function crypto_get_id() was completed, but I haven't use it .
    # 2. TW stock stop when weekend or holiday
    # 3. 因為crypto free api doesn't provide historical data, so it should be 
    # record in the google sheet
    
    # ADD STOCK NAME
    ledger = get_ledger()
    
    
    # Current Price
    loc_crypto_price = crypto_price.copy()
    loc_crypto_price = loc_crypto_price.drop('LAST_UPDATED', axis=1)
    
    
    # Historical Price
    hist_price = get_hist_data()
    
    
    # Merge
    main_data = ledger \
        .merge(hist_price, how='left', on=['MARKET', 'SYMBOL']) \
        .merge(loc_crypto_price, how='left', on=['ID', 'SYMBOL'])            
        
    main_data = stk.cal_stop_loss(df=main_data, mean_price='MEAN_COST',
                                  hist_high='HIST_HIGH', thld=0.2)
    
    main_data['ACTION'] = \
        np.where(main_data['PRICE'] <= main_data['STOP_LOSS'], True, False)
    
    
    # Reorder Columns ......
    main_data = main_data[['MARKET', 'SYMBOL', 'HOLD_DAYS', 'VOLUME', 
                           'COST', 'MEAN_COST', 'PRICE', 
                           'STOP_LOSS', 'HIST_HIGH', 'ACTION']]
    
    main_data = main_data.sort_values(by=['MARKET', 'COST'], 
                                      ascending=[True, False]) \
                .reset_index(drop=True)
    
    
    # Action Rows ......
    chk = main_data[main_data['ACTION']==True] 
    
    if len(chk) == 0:
        print('No action needed.')
        return ''
    
    
    # Reorganize Data ......
    content_li = []
    for i in range(len(main_data)):
        
        temp_data = main_data[main_data.index==i]
        temp_data = temp_data.T
        temp_data.columns = ['']
        
        content_li.append(temp_data)
    
    
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
    
    notif_stop_loss()    
    
    return ''



if __name__ == '__main__':
    master()




