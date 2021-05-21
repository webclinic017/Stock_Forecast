#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 00:42:16 2021

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
    path = '/home/aronhack/stock_forecast/dashboard/'


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




if __name__ == '__main__':
    master()







#This example uses Python 2.7 and the python-request library.


# coinmarketcap

def crypto_connect():

    from requests import Request, Session    
    
    headers = {
      'Accepts': 'application/json',
      'X-CMC_PRO_API_KEY': '5f1463fa-c1ce-4a70-996d-4a29062acc6e',
    }
    
    session = Session()
    session.headers.update(headers)
    
    return session



def crypto_get_price(symbol=[]):
    '''
    Get current price.

    Returns
    -------
    str
        DESCRIPTION.

    '''
    
    from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
    
    session = crypto_connect()
    
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
      'start':'1',
      'limit':'5000',
      'convert':'USD'
    }
    
    try:
      response = session.get(url, params=parameters)
      data = json.loads(response.text)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
      print(e)
      return e
  
    # Convert To DataFrame ......
    results = pd.DataFrame(data['data'])
    results = cbyz.df_col_upper(results)
    
    # Filter ......
    if len(symbol) > 0:
        results = results[results['SYMBOL'].isin(symbol)] \
                    .reset_index(drop=True)
    
    # Get Price From Dict ......
    for i in range(len(results)):
        price = results.loc[i, 'QUOTE']
        price = price['USD']['price']
        results.loc[i, 'PRICE'] = price

    return results

  


def crypto_get_id(symbol=[]):
    
    session = crypto_connect()

    # Crypto ID
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/map'
    response = session.get(url)
    data = json.loads(response.text)
    
    results = pd.DataFrame(data['data'])
    results = cbyz.df_col_upper(df=results)
    
    if len(symbol) > 0:
        results = results[results['SYMBOL'].isin(symbol)] \
                    .reset_index(drop=True)

    return results







