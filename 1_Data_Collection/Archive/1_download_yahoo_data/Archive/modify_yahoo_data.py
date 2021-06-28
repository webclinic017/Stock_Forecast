#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:44:01 2020

@author: Aron
"""


import requests
import pandas as pd
import time
from datetime import date
import datetime


path = "/Users/Aron/Documents/GitHub/Data/Stock-Forecast"
#path = "/home/aronhack/agri_wholesale"

path_resource = path + '/Resource'
path_export = path + '/Export'



df = pd.read_csv(path + '/Resource/0050.TW_20070101_20200531.csv')
df.columns

df['SECURITY_CODE'] = '0050'
df['NAME'] = ''
df['TRADE_VALUE'] = ''
df['PRICE_CHANGE'] = ''

# Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')

# Reorder columns .....
cols = ['Date', 'SECURITY_CODE', 'NAME', 'Volume', 'TRADE_VALUE',
        'Open', 'High', 'Low', 'Close', 'PRICE_CHANGE', 'Volume' ]
df = df[cols]

# Rename ....
col_name = ['WORK_DATE', 'SECURITY_CODE', 'NAME', 'TRADE_VOLUME', 'TRADE_VALUE',
        'OPENINIG_PRICE', 'HIGHEST_PRICE', 'LOWEST_PRICE', 'CLOSING_PRICE',
        'PRICE_CHANGE', 'TRANSACTION' ]

df.columns = col_name


# Round
df = df.round({'OPENINIG_PRICE': 1, 'HIGHEST_PRICE': 1,
               'LOWEST_PRICE':1, 'CLOSING_PRICE':1, 'TRANSACTION':0})

# Export .....
df.to_csv(path_export + '/upload_local_db.csv', index=False, header = False)

