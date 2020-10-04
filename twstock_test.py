# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:31:25 2020

@author: Aron
"""


import twstock
stock = twstock.Stock('2330')
data.sid  # 回傳股票代號

stock.price
stock.high  
stock.date


data = stock.fetch(2015, 7)  # 獲取 2015 年 7 月之股票資料
stock.fetch(2010, 5)  # 獲取 2010 年 5 月之股票資料
stock.fetch_31()      # 獲取近 31 日開盤之股票資料
data = stock.fetch_from(2015, 4)  # 獲取 2000 年 10 月至今日之股票資料
