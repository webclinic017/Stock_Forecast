# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:42:25 2020

@author: Aron
"""


import requests
import pandas as pd
import time
from datetime import date
import datetime


path = "//psf/Home/Documents/GitHub/Data/Stock-Forecast"
path_export = path + '/Export'

# https://data.gov.tw/dataset/11549
link = 'https://quality.data.gov.tw/dq_download_json.php?nid=11549&md5_url=bb878d47ffbe7b83bfc1b41d0b24946e'

r = requests.get(link)
data = pd.DataFrame(r.json())

data.columns


# Yahoo Columns
# Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')



data.columns = ['SECURITY_CODE', 'NAME', 'TRADE_VOLUME', 'TRADE_VALUE', 
                'OPENING_PRICE', 'HIGHEST_PRICE' ,'LOWEST_PRICE', 
                'CLOSING_PRICE', 'PRICE_CHANGE', 'TRANSACTION']

data['WORK_DATE'] = date.today()

cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]



data.to_csv(path_export + '/stock_20200612.csv', index=False, header = True)






# 證券代號、證券名稱、成交股數、成交金額、開盤價、最高價、最低價、收盤價、漲跌價差、成交筆數
# 'SECURITY_CODE', 'NAME', 'TRADE_VOLUME', 'TRADE_VALUE', 'OPENING_PRICE',
# 'HIGHEST_PRICEm' ,'LOWEST_PRICE', 'CLOSING_PRICE', 'CHANGE', 'TRANSACTION'

