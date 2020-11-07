#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:37:24 2020

@author: Aron
"""


# Stock Prediction with XGBoost: A Technical Indicators’ approach
# https://medium.com/@hsahu/stock-prediction-with-xgboost-a-technical-indicators-approach-5f7e5940e9e3


import pandas as pd
import sklearn



import pandas as pd
import numpy as np
import sys, time, os, gc
import arrow


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'

# Codebase
path_codebase = ['/Users/Aron/Documents/GitHub/Arsenal',
                 '/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path

    
import codebase_yz as cbyz
import arsenal as ar

data = ar.get_stock_data(begin_date=20190101, 
                               end_date=20201031, 
                            stock_symbol='0050')
    
data = data.rename(columns={'CLOSE':'Close'})



#  Moving Averages (SMA & EWMA)

# Simple Moving Average 
def SMA(data, days): 
    sma = pd.Series(pd.rolling_mean(data['Close'], days), name = 'SMA_' + str(days))
    data = data.join(sma) 
    return data


# Exponentially-weighted Moving Average 
def EWMA(data, days):
    ema = pd.Series(pd.ewma(data['Close'], span = days, min_periods = days - 1), 
    name = 'EWMA_' + str(days))
    data = data.join(ema) 
    return data


days_list = [10,50,100,200]
for days in days_list:
    data = SMA(data, days)
    data = EWMA(data,days)
    
    
#Bollinger Bands (UpperBB & LowerBB)
def bbands(data, window=days):
    MA = data.Close.rolling(window=days).mean()
    SD = data.Close.rolling(window=days).std()
    data['UpperBB'] = MA + (2 * SD) 
    data['LowerBB'] = MA - (2 * SD)
    return data
days = 50
data = bbands(data, days)


# Force Index (ForceIndex)
def ForceIndex(data, days): 
    FI = pd.Series(data['Close'].diff(days) * data['Volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data
days = 1
data = ForceIndex(data,days)

# Commodity Channel Index (CCI)
def CCI(data, days): 
    TP = (data['High'] + data['Low'] + data['Close']) / 3 
    CCI = pd.Series((TP - pd.rolling_mean(TP, days)) / (0.015 * pd.rolling_std(TP, days)), 
    name = 'CCI')
    data = data.join(CCI)
    return data
days = 20
data = CCI(data, days)

# Ease Of Movement (EVM)
def EVM(data, days): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br 
    EVM_MA = pd.Series(pd.rolling_mean(EVM, days), name = 'EVM') 
    data = data.join(EVM_MA) 
    return data 
days = 14
data = EVM(data, days)

# Rate of Change (ROC)
def ROC(data,days):
    N = data['Close'].diff(days)
    D = data['Close'].shift(days)
    roc = pd.Series(N/D,name='ROW')
    data = data.join(roc)
    return data 
days = 5
data = ROC(data,days)




# Data Preprocessing


def rescale(data):
    data = data.dropna().astype('float')
    data = sklearn.preprocessing.scale(data)
    data = pd.DataFrame(data, columns=data.columns)
    return data

def class_balance(train):
    count_class_0, count_class_1 = train['target'].value_counts()
    train_class_0 = train[train['target'] == 0]
    train_class_1 = train[train['target'] == 1]

    if count_class_0>count_class_1:
        train_class_0_under = train_class_0.sample(count_class_1)
        train_sampled = pd.concat([train_class_0_under, train_class_1], axis=0)
    else:
        train_class_1_under = train_class_1.sample(count_class_0)
        train_sampled = pd.concat([train_class_0, train_class_1_under], axis=0)
    
    print(train_sampled['target'].value_counts())
    train_sampled['target'].value_counts().plot(kind='bar', title='Count (target)')
    plt.show()
    return train_sampled



# Convert to Classification problem
    
def prepare_X_y(data):
    X = data.values
    ind = list(data.columns).index('Open')
    y = []
    for i in range(X.shape[0]-1):
        if (X[i+1,ind]-X[i,ind])>0:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    X = X[:-1]
    return X,y




# Training
    
def split_train_test(X,y):
    split_ratio=0.9
    train_size = int(round(split_ratio * X.shape[0]))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    return model
  
def predict(model):
    y_pred = model.predict(X_test)
    y_pred = [round(value) for value in y_pred]
    return y_pred