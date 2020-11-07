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
    link = 'https://www.twse.com.tw/zh/page/ETF/list.html'  
    
    try:
        r = requests.get(link)
    except:
        print('Requests error')
        return ''
    
    
    # Parse table ......
    try:
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.select('#main table')[0]
    except:
        print('Parse error')
        return ''    

    rows = table.find_all('tr')
    
    
    parse_raw = []
    
    for i in range(0, len(rows)):
    
        cells = rows[i].find_all('td')
        single_stock = []
        
        for j in range(0, len(cells)):
            
            single_stock.append(cells[j].text)
            
        parse_raw.append(single_stock)
        
    parse_data = pd.DataFrame(parse_raw)
    
    
    # 上市日期 證券代號 證券簡稱	 發行人 標的指數
    cols = ['LISTING_DATE', 'STOCK_SYMBOL', 'STOCK_NAME',
            'ISSUER', 'DESCRIPTION']
    
    parse_data.columns = cols
    
    parse_data = parse_data[~parse_data['STOCK_SYMBOL'].isna()] \
                    .reset_index(drop=True)
    
    
    if (export == True) & (path_export != None):
        parse_data.to_csv(path_export + '/parse_data.csv',
                          index=False,
                          encoding='utf-8-sig') 
        
    if (export == True) & (path_export != None):
        ar.db_upload(parse_data, 'ETF_LIST_TW')
        
        
    return parse_data


# ...............


def load_data_us():
    
    return ''

# ...............


def master(stock_type='tw', export=False, 
           local=True, update=False):
    
     # Path .....
    global path
    if local == True:
        path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/etf_list'
    else:
        path = '/home/aronhack/stock_forecast/dashboard'   
    
    
    initialize()
    
    
    if stock_type == 'tw':
        stock_info_final = load_data_tw(export=export, 
                                        update=update)
    elif stock_type == 'us':
        load_data_us(export=export)
    
    return stock_info_final


if __name__ == '__main__':
    master()

