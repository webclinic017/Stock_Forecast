#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 22:34:35 2021

@author: Aron
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/'
else:
    path = '/home/aronhack/stock_forecast/dashboard/'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ/',
                 path + 'Function/']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk


# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

# 新增工作資料夾
global path_resource, path_function, path_temp, path_export
path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])      



def load_data():
    '''
    讀取資料及重新整理
    '''
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




if __name__ == '__main__':
    master()


http://gbeced.github.io/pyalgotrade/docs/v0.20/html/tutorial.html



stk_data = stk.get_data(data_begin=20190101, data_end=20210301, 
             stock_type='tw', stock_symbol=['0050'], local=True)



from pyalgotrade import strategy
from pyalgotrade.barfeed import quandlfeed


class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super(MyStrategy, self).__init__(feed)
        self.__instrument = instrument

    def onBars(self, bars):
        bar = bars[self.__instrument]
        self.info(bar.getClose())

# Load the bar feed from the CSV file
feed = quandlfeed.Feed()
# feed.addBarsFromCSV("orcl", "WIKI-ORCL-2000-quandl.csv")


feed.addBarsFromCSV("orcl", path + '/3_Backtest/EOD-DIS.csv')




# Evaluate the strategy with the feed's bars.
myStrategy = MyStrategy(feed, "orcl")
myStrategy.run()



