#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:28:10 2020

@author: Aron
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import requests
from bs4 import BeautifulSoup



# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar



# ..................


def initialize():

    # 新增工作資料夾
    global path_resource, path_function, path_temp, path_export
    path_resource = path + '/Resource'
    path_function = path + '/Function'
    path_temp = path + '/Temp'
    path_export = path + '/Export'
    
    
    cbyz.create_folder(path=[path_resource, path_function, 
                             path_temp, path_export])        
    return ''


# .............
    
def load_data_tw(export=False, update=False, upload=False):
    
      
    # 本國上市證券國際證券辨識號碼一覽表
    link = 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2'  
          
    r = requests.get(link)
    # r.status_code # Debug


    soup = BeautifulSoup(r.text, 'html.parser')
    
    
    # Parse update date
    update_date = soup.select('.h1 center')[0]
    update_date = update_date.text
    update_date = ar.str_remove_non_ascii(update_date)
    update_date = ar.str_remove_special(update_date)
    update_date = update_date.replace(' ', '')
    
    
    # Parse table ......
    table = soup.select('table.h4')[0]
    rows = table.find_all('tr')
    
    
    stock_list_raw = []
    
    for i in range(0, len(rows)):
    
        cells = rows[i].find_all('td')
        single_stock = []
        
        for j in range(0, len(cells)):
            
            single_stock.append(cells[j].text)
            
        stock_list_raw.append(single_stock)
        
    stock_list_pre = pd.DataFrame(stock_list_raw)
    
    
    # 有價證券代號及名稱 STOCK_SYMBOL
    # 國際證券辨識號碼(ISIN Code) ISIN_CODE
    # 上市日 LISTING_DATE
    # 市場別 MARKET
    # 產業別 INDUSTRY
    # CFICode CFI_CODE
    # 備註 NOTE
    
    cols = ['STOCK_SYMBOL_RAW', 'ISIN_CODE', 'LISTING_DATE',
            'MARKET', 'INDUSTRY', 'CFI_CODE', 'NOTE']
    
    stock_list_pre.columns = cols
    
    stock_list_pre = stock_list_pre.iloc[1:, :]
    stock_list_pre = stock_list_pre[~stock_list_pre['CFI_CODE'].isna()]
    stock_list_pre['ID'] = stock_list_pre.index
    
    
    # Split stock name ......
    stock_symbol = stock_list_pre['STOCK_SYMBOL_RAW'] \
        .str.split('　', expand=True)
        
    stock_symbol.columns = ['STOCK_SYMBOL', 'STOCK_NAME']
    stock_symbol['ID'] = stock_symbol.index
    

    
    # Organize ......
    stock_info = stock_list_pre.merge(stock_symbol, on='ID')
    stock_info['LISTING_DATE'].apply(cbyz.ymd)
    stock_info = stock_info[['STOCK_SYMBOL', 'STOCK_NAME',
                             'ISIN_CODE', 'LISTING_DATE', 
                             'MARKET', 'INDUSTRY', 'CFI_CODE']]
    
    
    if (export == True) & (path_export != None):
        stock_info.to_csv(path_export + '/stock_info.csv',
                          index=False,
                          encoding='utf-8-sig') 
        
    return stock_info


# ...............


def load_data_us():
    
    return ''

# ...............


def master(stock_type='tw', export=False, local=True, update=False):
    
     # Path .....
    global path
    if local == True:
        path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/stock_info'
    else:
        path = '/home/aronhack/stock_forecast/dashboard'   
    
    
    initialize()
    
    
    if stock_type == 'tw':
        stock_info_final = load_data_tw(export=export, update=update)
    elif stock_type == 'us':
        load_data_us(export=export)
    
    return stock_info_final


if __name__ == '__main__':
    master()

