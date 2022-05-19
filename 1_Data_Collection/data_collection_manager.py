#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:22:57 2020

@author: Aron
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import yfinance as yf


host = 2
host = 0


# Path .....
if host == 0:
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/1_Data_Collection'
elif host == 1:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'
elif host == 2:
    path = '/home/jupyter//Production/1_Data_Collection'
elif host == 3:
    path = '/home/jupyter//Develop/1_Data_Collection'

# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20220519',
                 r'/home/jupyter/Arsenal/20220519',
                 path + '/1_Shareholding_Spread',
                 path + '/2_TEJ',
                 path + '/Function']

for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk

import shareholding_spread
# import tej_v1_10 as tej
import tej_v1_0302 as tej



# 自動設定區 -------
path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                            path_temp, path_export])     

pd.set_option('display.max_columns', 30)
 


# Load Data ----------------


def tw_get_capital_flows():

    '''
    資金流向

    '''    

    from bs4 import BeautifulSoup
    import requests
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from bs4 import BeautifulSoup
    from selenium.webdriver.support.ui import Select
    
    
    # CMoney - 這個網站的產業命名和別人不一樣，而且只有小數點一位
    # link = 'https://www.cmoney.tw/finance/f00010.aspx'
          
    # 理財網
    # link = 'https://www.moneydj.com/Z/ZB/ZBA/ZBA.djhtm'
    
    # 玉山證券
    # link = 'https://www.esunsec.com.tw/tw-market/z/zb/zba/zba.djhtm'
    # iframe
    link = 'https://sjmain.esunsec.com.tw/z/zb/zba/zba.djhtm'
    
    
    
    webdriver_path = '/Users/Aron/Documents/GitHub/Arsenal/geckodriver'
    driver = webdriver.Firefox(executable_path=webdriver_path)
    driver.get(link)
    
    
    
    element = driver.find_elements_by_css_selector("#SysJustIFRAMEDIV")[0]
    html = element.get_attribute('innerHTML')
    soup = BeautifulSoup(html, 'html.parser')   
    
    table = soup.find_all('table')
    data = pd.read_html(str(table))
    
    data = data[['分類', '成交比重']]
    data.columns = ['INDUSTRY', 'CAPITAL_TREND']
    data['WORK_DATE'] = cbyz.date_get_today()
    data = data[['WORK_DATE', 'INDUSTRY', 'CAPITAL_TREND']]
    
    
    data = data[data['INDUSTRY']!='加權指數']

    ar.db_upload(data=data, 
                 table_name='tw_capital_flows',
                 local=local)    
    
    # 分類 INDUSTRY
    # 指數 INDUSTRY_INDEX
    # 漲跌  
    # 漲跌（幅）  
    # 成交量(億)  
    # 成交比重/流向率 CAPITAL_TREND
    
    
    driver.close()





def yahoo_download_data(stock_list=[], chunk_begin=None, chunk_end=None):
    '''
    讀取資料及重新整理
    '''
   
    global stock_type
    

    # Get stock list
    if len(stock_list) > 0:
        stock_list = pd.DataFrame({'STOCK_SYMBOL':stock_list})
        stock_list['STOCK_SYMBOL'] = stock_list['STOCK_SYMBOL'].astype(str)
    else:
        # stock_list = stk.get_list(stock_type=stock_type)
        # stock_list = stk.tw_get_company_info()
        stock_list = stk.twse_get_data()
        stock_list = stock_list[['STOCK_SYMBOL', 'NAME']].drop_duplicates()
    
    
    # Split as chunk ......
    if chunk_begin != None and chunk_end != None:
        stock_list = stock_list.loc[chunk_begin:chunk_end, :]
        
    elif chunk_end != None and chunk_end >= len(stock_list):
        stock_list = stock_list.loc[chunk_begin:len(stock_list), :]
    
    stock_list = stock_list[['STOCK_SYMBOL']].reset_index(drop=True)
    stock_list = cbyz.df_conv_col_type(df=stock_list, cols='STOCK_SYMBOL',
                                       to='str')
    
    
    # Get historical data ......
    hist_data_raw = pd.DataFrame()
    
    
    # for i in range(190, 200):
    for i in range(0, len(stock_list)):
    
        # Get stock symbol
        if stock_type == 'tw':
            stock_id = stock_list.loc[i, 'STOCK_SYMBOL'] + '.TW'
            
        elif stock_type == 'us':
            stock_id = stock_list.loc[i, 'STOCK_SYMBOL']
    
    
        # Call API
        try:
            data = yf.Ticker(stock_id)
            df = data.history(period="max")
        except:
            continue
        
        
        if len(df) > 0:
            df['STOCK_SYMBOL'] = stock_list.loc[i, 'STOCK_SYMBOL']
            hist_data_raw = hist_data_raw.append(df, sort=False)


        time.sleep(0.8)
        print(str(i) + '/' + str(len(stock_list)))
    
    
    # Organize data ......
    global hist_data
    hist_data = hist_data_raw.copy()
    hist_data = hist_data.reset_index()
    hist_data['Date'] = hist_data['Date'].astype('str')
    
    # There are some na values.
    hist_data = hist_data[~hist_data['Open'].isna()]
    
    
    # Rename
    hist_data = hist_data \
                .rename(columns={'Date':'WORK_DATE',
                                 'Open':'OPEN',
                                 'High':'HIGH',
                                 'Low':'LOW',
                                 'Close':'CLOSE',
                                 'Volume':'VOLUME'})
    
    hist_data = hist_data[['WORK_DATE', 'STOCK_SYMBOL', 'OPEN', 
                           'HIGH', 'LOW', 'CLOSE', 'VOLUME']]

    
    # Upload ......
    # if upload:
    #     if stock_type == 'tw':
    #         ar.db_upload(data=hist_data, 
    #                      table_name='stock_data_tw',
    #                      local=local)
            
    #     elif stock_type == 'us':
    #         ar.db_upload(data=hist_data, 
    #                      table_name='stock_data_us',
    #                      local=local)

    # else:
    #     hist_data = hist_data.to_csv(path_temp + '/hist_data_' \
    #                                  + str(chunk_begin) + '_' \
    #                                  + str(chunk_end) + '.csv',
    #                                  index=False)










def get_us_stock_data():

        
    # US Stock --------
    
    results = pd.DataFrame()
    
    for i in us_stock:
        
        data = yf.Ticker(i)
        df = data.history(period="max")
        df['STOCK_SYMBOL'] = i
    
        results = results.append(df)
            
    
    results.reset_index(level=0, inplace=True)
    
    
    # Rename
    results.rename(columns={'Date':'WORK_DATE',
                                'Open':'OPEN',
                                'High':'HIGH',
                                'Low':'LOW',
                                'Close':'CLOSE',
                                'Volume':'VOLUME'
                                    }, 
                           inplace=True)
    
    # Filter columns
    cols = ['WORK_DATE', 'STOCK_SYMBOL', 'OPEN', 
            'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    results = results[cols]
    
    
    # Failed processing format-parameters; Python 'timestamp' cannot be converted to a MySQL type
    
    # results['WORK_DATE'] = results['WORK_DATE'].apply(cbyz.ymd)
    results['WORK_DATE'] = results['WORK_DATE'].astype('str')
    
    ar.db_upload(results,
                 'stock_data_us')

    
    return ''



# %% Archive ------


def download_market_data(overwrite=False, upload=True):
    '''
    主工作區
    '''
    
    global stock_type
    stock_type = 'tw'
    
    
    overwrite=False
    upload=True    
    
    
    # 檢查兩筆，如果數字都一樣的話就不更新
    if stock_type == 'tw':
        repre_symbols = ['2330', '3008', '0050']
        # repre_symbols = ['1101', '1102']        
    

    
    if overwrite:
        # Delete existing data
        sql = "truncate table stock_data_tw"
        ar.db_execute(sql, local=True, fetch=False)
    
        data = yahoo_download_data(stock_list=[], chunk_begin=None, chunk_end=None, 
                            upload=True)
    else:
        
        # Query Data
        data = stk.twse_get_data()
        
        # Check data duplicated or not
        chk_data = data[data['STOCK_SYMBOL'].isin(repre_symbols)]
        chk_data = chk_data[['WORK_DATE', 'STOCK_SYMBOL', 
                               'OPEN', 'CLOSE', 'HIGH', 'LOW']]
        
        # Hist Data
        today = data['WORK_DATE'].max()
        chk_date = cbyz.date_cal(today, -14, 'd')
        hist_data = stk.get_data(data_begin=chk_date, 
                                 data_end=today,
                                 stock_symbol=repre_symbols,
                                 stock_type=stock_type,
                                 local=local)
        
        hist_data['DATE_MAX'] = hist_data['WORK_DATE'].max()
        hist_data = hist_data[hist_data['DATE_MAX']==hist_data['WORK_DATE']] \
                    .reset_index(drop=True)
                    
        hist_data = hist_data[['WORK_DATE', 'STOCK_SYMBOL', 
                               'OPEN', 'CLOSE', 'HIGH', 'LOW']]
        
        chk_diff = ar.df_chk_diff(chk_data, hist_data, on='STOCK_SYMBOL',
                                  chk_na=False)
        
        if not chk_diff:
            return ''
        
    # Check
    # data['WORK_DATE'] = 20210813
    # data[data['STOCK_SYMBOL']=='2399']
    
        
   # Update，如tmse的transaction
    data = data[['WORK_DATE', 'STOCK_SYMBOL', 'OPEN', 
                 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
    
    
    cbyz.df_chk_col_na(df=data)
    

    # Upload ------
    if upload:
        if stock_type == 'tw':
            ar.db_upload(data=data, 
                         table_name='stock_data_tw')
            
        elif stock_type == 'us':
            ar.db_upload(data=data, 
                         table_name='stock_data_us')
    
    return ''



# %% Execute ------


def automation():
    
    shareholding_spread.automation()
    tej.automation()
    stk.od_tw_update_fx_rate()



if __name__ == '__main__':
    
    automation()






