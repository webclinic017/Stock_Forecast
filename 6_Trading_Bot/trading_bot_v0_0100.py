#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

History

20210703 - Replaced MA with WMA in SAM

"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
import random

host = 3
host = 2
host = 4
host = 0
market = 'tw'


# Path .....
if host == 0:
    # Home
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/3_Backtest'
    path_sam = '/Users/aron/Documents/GitHub/Stock_Forecast/2_Stock_Analysis'

elif host == 2:
    # PythonAnyWhere
    path = '/home/jupyter/Production/3_Backtest'
    path_sam = '/home/jupyter/Production/2_Stock_Analysis'    
    
elif host == 3:
    # GCP
    path = '/home/jupyter/Develop/3_Backtest'
    path_sam = '/home/jupyter/Develop/2_Stock_Analysis'      

elif host == 4:    
    # RT
    path = r'D:\Data_Mining\GitHub共用\Stock_Forecast\3_Backtest'
    path_sam = r'D:\Data_Mining\GitHub共用\Stock_Forecast\2_Stock_Analysis'    


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'D:\Data_Mining\GitHub共用\Arsenal',
                 r'D:\Data_Mining\Projects\Codebase_YZ',
                 r'/Users/aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20220219',
                 r'/home/jupyter/Arsenal/20220219',
                 path + '/Function',
                 path_sam]


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk
import codebase_ml as cbml
# import stock_analysis_manager_v2_10_dev as sam
# import stock_analysis_manager_v2_11_dev as sam
# import stock_analysis_manager_v2_112_dev as sam
# import stock_analysis_manager_v2_400_dev as sam
import stock_analysis_manager_v2_502_dev as sam



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     





def master_single():
    
    new_data = query_quote()
    
    extract price and volume 
    append(new_data)
    



def master():
    
    
    # v1. Only Sell, Not Buy
    # v2. Automatically sell and buy
    import multiprocessing
    stop_loss = 0.7
    
    # Schedule to restart
    connnect_sql()
    
    ledger = stk.read_ledger()
    
    target = [2601]
    ignore_symbol = []
    
    
    while 2 > 1:
        
        # - Can I request multiple symbols from fugle
        for s in symbols:
            master_single()

        stop_loss()


        if mode == 1:
            Stimulate()
            
        elif mode == 2:
            Enhance Learning()
            
        
        if sell_signal:
            - sell with strategy[i]
            - notification
        
        
        # Export
        if excetue_time % 60:
            pd.to_csv()


        sleep(2)



















