#!/usr/bin/env python
# coding: utf-8

import os
import re
import requests
import numpy as np
import pandas as pd
from datetime import date
# from flask import Flask, request
import MySQLdb



db = MySQLdb.connect(
    host = 'aronhack.mysql.pythonanywhere-services.com',
    user = 'aronhack',
    passwd = 'pythonmysql2020',
    db = 'aronhack$aronhack_dashboard',
    charset = 'utf8')

cursor = db.cursor()



link = 'https://quality.data.gov.tw/dq_download_json.php?nid=11549&md5_url=bb878d47ffbe7b83bfc1b41d0b24946e'
r = requests.get(link)
data = pd.DataFrame(r.json())
data.columns


# For task debug
data


# Bug, 有些欄位不需要
# ['證券代號', '證券名稱', '成交股數', '成交金額', '開盤價',
#  '最高價', '最低價', '收盤價', '漲跌價差', '成交筆數']
data.columns = ['SECURITY_CODE', 'NAME', 'TRADE_VOLUME', 'TRADE_VALUE', 
                'OPEN_PRICE', 'HIGH_PRICE' ,'LOW_PRICE', 
                'CLOSE_PRICE', 'PRICE_CHANGE', 'TRANSACTION']


data['WORK_DATE'] = date.today()

cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]




# data.to_csv(path + '/stock.csv', index=False, header = True)
# data.to_csv(path + '/stock2.csv', index=False, header = False)

# cursor.execute ("CREATE TABLE stock_data (WORK_DATE varchar(10), SECURITY_CODE varchar(6), NAME varchar(14), TRADE_VOLUME varchar(11), TRADE_VALUE varchar(13), OPENING_PRICE varchar(8), HIGHEST_PRICE varchar(8), LOWEST_PRICE varchar(8), CLOSING_PRICE varchar(8), PRICE_CHANGE varchar(8), TRANSACTION varchar(6))")


data_list = data.values.tolist()



# Insert
# Before modify columns
query = "INSERT INTO stock_data (WORK_DATE, SECURITY_CODE, NAME, TRADE_VOLUME, TRADE_VALUE, OPENING_PRICE, HIGHEST_PRICE, LOWEST_PRICE, CLOSING_PRICE, PRICE_CHANGE, TRANSACTION) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"

# query = "INSERT INTO stock_data (WORK_DATE, SECURITY_CODE, NAME, TRADE_VOLUME, TRADE_VALUE, OPENING_PRICE, HIGHEST_PRICE, LOWEST_PRICE, CLOSING_PRICE, PRICE_CHANGE, TRANSACTION) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"
cursor.executemany(query, data_list)
db.commit()



# cursor.execute ("select * from stock_data;")

# check = cursor.fetchall()
# check
