#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:05:43 2020

@author: Aron
"""

from fbprophet import Prophet
import pandas as pd
# import matplotlib.pyplot as plt
# import xgboost as xgb
# from sklearn.metrics import accuracy_score
import datetime


# path = "/Users/Aron/Documents/GitHub/Data/Stock-Forecast"
path = "//psf/Home/Documents/GitHub/Data/Stock-Forecast"
# path = "/home/aronhack/stock_forecast"
path_resource = path + '/Resource'
path_export = path + '/Export'



# Tutorial -------------

# http://gonzalopla.com/predicting-stock-exchange-prices-with-machine-learning/
# https://towardsdatascience.com/forecasting-stock-prices-using-xgboost-a-detailed-walk-through-7817c1ff536a
# https://medium.com/@hsahu/stock-prediction-with-xgboost-a-technical-indicators-approach-5f7e5940e9e3


# Load Data ------------

# Market List
stock_data = pd.read_csv(path_resource + "/0050.TW_20070101_20200531.csv")
stock_data = stock_data.drop('Adj Close', 1)

stock_data = stock_data[['Date', 'Close']]
stock_data.columns = ['ds', 'y']

latest_date = stock_data.iloc[-1, ].ds

# Data Overview
stock_data.columns
stock_data.head()


# -------------

# Add stock data with SECURITY_ID
# stock_data.columns
# stock_data = stock_data.round({'Open':1, 'High': 1, 'Low': 1,
#                                'Close':1, 'Adj Close':1, 'Volume':0})

# stock_data.to_csv(path_export + '/stock_data.csv', index=False, header=True)



# Prophet ---------
# (1) Update limit forecast digits

m = Prophet()
m.fit(stock_data)

future = m.make_future_dataframe(periods=14)
future.tail()


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast = forecast[forecast['ds'] > latest_date]

forecast.columns

# Organize Columns ------------
forecast.columns = ['DATE', 'PRICE', 'LOW_PRICE', 'HIGH_PRICE']
forecast['SECURITY_CODE'] = '0050'

cols = forecast.columns.tolist()
cols = cols[-1:] + cols[:-1]
forecast = forecast[cols]

# Limit digits
forecast = forecast.round({'PRICE': 1, 'LOW_PRICE': 1, 'HIGH_PRICE':1})


forecast.to_csv(path_export + '/upload_to_db.csv', index=False, header = True)


# Upload To Database ...........
import mysql.connector

db = mysql.connector.connect(
 host="localhost",
 user="aron",
 password="57diyyLCHH4q1kwD",
 port="8889",
 database="powerbi"
)



