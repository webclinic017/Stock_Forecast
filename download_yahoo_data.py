#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:09:08 2020

@author: Aron
"""

import os, sys
import re
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date
# from flask import Flask, request
import time
import h5py
import yfinance as yf # https://pypi.org/project/yfinance/




# 設定工作目錄 .....
path = '/Users/Aron/Documents/GitHub/Data/Stock-Forecast'

# Codebase
path_codebase = ['/Users/Aron/Documents/GitHub/Arsenal',
                 '/Users/Aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import arsenal as ar
import codebase_yz as cbyz


# Load Data ----------------

# # Local DB .....
# import mysql.connector

# db = mysql.connector.connect(
#  host="localhost",
#  user="aron",
#  password="57diyyLCHH4q1kwD",
#  port="8889",
#  database="twstock"
# )


# PythonAnywhere .....
# import MySQLdb

# db=MySQLdb.connect(
#     host='aronhack.mysql.pythonanywhere-services.com',
#     user='aronhack',
#     passwd='pythonmysql2020',
#     db='aronhack$aronhack_dashboard',
#     charset = 'utf8')

# cursor = db.cursor()

# cursor.execute("select * "
#                "from stock_name;")

# stock_list = pd.DataFrame(cursor.fetchall())
# stock_list.columns = ['STOCK_SYMBOL', 'NAME']







# 自動設定區 -------
pd.set_option('display.max_columns', 30)


def init(path):
    
    global path_resource, path_function, path_temp, path_export
    
    # 新增工作資料夾
    path_resource = path + '/Resource'
    path_function = path + '/Function'
    path_temp = path + '/Temp'
    path_export = path + '/Export'
    
    
    cbyz.create_folder(path=[path_resource, path_function, 
                             path_temp, path_export])



def load_data():
    '''
    讀取資料及重新整理
    '''
    # Get stock list
    stock_list = ar.get_twstock_list()


    # Get historical data
    hist_data_raw = pd.DataFrame()
    
    # 約40分鐘
    
    # 0, 300
    # 300, 600
    # 600, 900
    # 900, 1200
    
    begin_time = datetime.now()   
    # Method 1 .....
    
    # for i in range(0, 2):
    for i in range(0, len(stock_list)):
    # for i in range(190, 200):
    
        # Get data    
        stock_id = stock_list.loc[i, 'STOCK_SYMBOL'] + '.TW'
    
        data = yf.Ticker(stock_id)
        df = data.history(period="max")
        
        df['STOCK_SYMBOL'] = stock_list.loc[i, 'STOCK_SYMBOL']
        
        if len(df) > 0:
            hist_data_raw = hist_data_raw.append(df, sort=False)
        # historical_data_raw = pd.concat([, df])
        time.sleep(0.8)
        
        print(i)
    
    
    end_time = datetime.now() 
    print(end_time - begin_time)

    
    hist_data = hist_data_raw.copy()
    hist_data.reset_index(level=0, inplace=True)
    
    
    # Rename
    hist_data.rename(columns={'Date':'WORK_DATE',
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
    hist_data = hist_data[cols]
    
    
    # Export
    global path_export
    time_seq = cbyz.get_time_serial(with_time=True)
    hist_data.to_hdf(path_export + '/yahoo_finance_data_'+time_seq+'.h5', 
                     key='s')
    
    hist_data.to_csv(path_export + '/yahoo_finance_data_'+time_seq+'.csv', 
                     index=False)
    
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






# -------------------------








part1 = historical_data.iloc[0:1000000, ]
part2 = historical_data.iloc[1000000 * 1 : 1000000 * 2, ]
part3 = historical_data.iloc[950000 * 2 : 950000 * 3, ]
part4 = historical_data.iloc[1000000 * 2 : , ]


# part1.to_hdf(path + '/Resource/yahoo_finance_data_20200626_p2.h5', key='s')
# part2.to_hdf(path + '/Resource/yahoo_finance_data_20200626_p3.h5', key='s')
# part3.to_hdf(path + '/Resource/yahoo_finance_data_20200625_part3.h5', key='s')
# part4.to_hdf(path + '/Resource/yahoo_finance_data_20200626_p4.h5', key='s')



part1['WORK_DATE'] = part1['WORK_DATE'].dt.date
data_list = part1.values.tolist()

# Insert
query = "INSERT INTO stock_data (WORK_DATE, STOCK_SYMBOL, OPEN, HIGH, LOW, CLOSE, VOLUME) VALUES(%s,%s,%s,%s,%s,%s,%s);"
cursor.executemany(query, data_list)
# db.commit()




# %% Method 2 ------
data = yf.download("0050.TW", start="2020-06-25", end="2020-07-12")
stock_id = "0050.TW"

# -------------------------


# msft = yf.Ticker("MSFT")
msft = yf.Ticker("0051.TW")


# get stock info
msft.info

# get historical market data
hist = msft.history(period="max")

# show actions (dividends, splits)
msft.actions

# show dividends
msft.dividends

# show splits
msft.splits

# show financials
msft.financials
msft.quarterly_financials

# show major holders
stock.major_holders

# show institutional holders
stock.institutional_holders

# show balance heet
msft.balance_sheet
msft.quarterly_balance_sheet

# show cashflow
msft.cashflow
msft.quarterly_cashflow

# show earnings
msft.earnings
msft.quarterly_earnings

# show sustainability
msft.sustainability

# show analysts recommendations
msft.recommendations

# show next event (earnings, etc)
msft.calendar

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
msft.isin

# show options expirations
msft.options

# get option chain for specific expiration
opt = msft.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts