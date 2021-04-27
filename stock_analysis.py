#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:35:30 2020

@author: Aron
"""


# Outline
# (1) Data Collection
# (2) Price Manager
# > Price Forcast 
# > Export Buy Signal
# (3) Backtest Manager
# > Fee Manager
# > Profit Manager
# > Budget

# Monitor System ( n days in the future)
# Notification System


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import os, sys


local = False
local = True


data_begin = 20180101
data_end = 20200831


# Path .....

if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path + '/0_Finance_Controller',
                 path + '/1_Data_Collection',
                 path + '/2_Stock_Analysis',
                 path + '/3_Backtest',
                 path + '/4.Visualization']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar

import finance_controller_master as fcm
import data_collection_manager as dcm
import stock_analysis_manager as sam
import backtest_manager as btm
# import visualization_manager as vsm



# Finance Controller -------
fcm_data = fcm.master()
budget = fcm_data['BUDGET']
hold_stocks = fcm_data['HOLD_STOCKS']



# Stock Analysis -------
signal = sam.master(today=20200701,
                    hold_stocks=hold_stocks)


# Backtest -------
# from = 20180101
# periods = 12
# unit = 'w'
backtest = btm.master(being_date=20190101,
                      signal=signal,
                      budget=budget)



# Priority
# Model review




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





