#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:35:30 2020

@author: Aron
"""

import os
import re
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date
# from flask import Flask, request
import time
import h5py
import yfinance as yf # https://pypi.org/project/yfinance/

# 將project_home指定為你的專案路徑
project_home = u'/Users/Aron/Documents/GitHub/Arsenal/'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path


path = '/Users/Aron/Documents/GitHub/Data/Stock-Forecast'

# Load Data ----------------

# data = pd.read_hdf(path + '/Export/0050_20200723.h5', 's')
data = pd.read_hdf(path + '/Export/2230_20200723.h5', 's')



df = data.copy()
df.columns.upper()
df2 = col_upper(df)

test = df.columns.str.upper()




for i in range(1, len(data)):
    
    data.loc[i, 'LAST_CLOSE'] = data.loc[i-1, 'Close']
    print(i)

data['PRICE_DIFF'] = data['Close'] - data['LAST_CLOSE']
data['PRICE_DIFF_RATIO'] = data['PRICE_DIFF'] / data['LAST_CLOSE']
data['LIMIT_UP'] = data['PRICE_DIFF_RATIO'] > 0.095
data['LIMIT_DOWN'] = data['PRICE_DIFF_RATIO'] < -0.095





