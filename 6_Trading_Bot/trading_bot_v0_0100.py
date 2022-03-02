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
# host = 0
market = 'tw'


# Path .....
if host == 0:
    # Home
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/6_Trading_Bot'

elif host == 2:
    # PythonAnyWhere
    path = '/home/jupyter/Production/6_Trading_Bot'
    
elif host == 3:
    # GCP
    path = '/home/jupyter/Develop/6_Trading_Bot'

elif host == 4:    
    # RT
    path = r'D:\Data_Mining\GitHub共用\Stock_Forecast\6_Trading_Bot'


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'D:\Data_Mining\GitHub共用\Arsenal',
                 r'D:\Data_Mining\Projects\Codebase_YZ',
                 r'/Users/aron/Documents/GitHub/Codebase_YZ',
                 r'/home/jupyter/Codebase_YZ/20220223',
                 r'/home/jupyter/Arsenal/20220223',
                 path,
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk
# import codebase_ml as cbml
# import stock_analysis_manager_v2_400_dev as sam
# import stock_analysis_manager_v2_502_dev as sam


# import multiprocessing as mp
# import trading_bot_function as tbf
import fugle_v1_0100_dev as fugle

ar.host = host



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

path_resource = path + '/Resource'
path_function = path + '/Function'
path_temp = path + '/Temp'
path_export = path + '/Export'


cbyz.os_create_folder(path=[path_resource, path_function, 
                         path_temp, path_export])     



def master_single():
    
    # new_data = query_quote()
    
    # extract price and volume 
    # append(new_data)
    
    pass
    


def master():
    
    
    
    
    # Version Plan
    # v1. Only Sell, Not Buy
    # v2. Automatically sell and buy    
    
    
    # Worklist
    # - Multiprocessing
    
    
    
    # Sell Signal
    # - 停損不停利策略
    # - 事先預判有多少法人持股，如果數量很少，當他們賣的時候就要跟著賣了
    
    # Fugle API ......
    # HTTP request 每分鐘	60
    # WebSocket 連線數 5    
    
    
    # Worklist ......
    # - GCP always on schedule
    # - GCP ploty
    # - GCP collect Fugle data on free VM
    
    
    

    # import multiprocessing
    # stop_loss = 0.7
    
    # # Schedule to restart
    # connnect_sql()
    
    # ledger = stk.read_ledger()
    
    # target = [2601]
    # ignore_symbol = []
    
    
    while 2 > 1:
        
        
        # Reference ......
        # # - Can I request multiple symbols from fugle
        # for s in symbols:
        #     master_single()
            
        # 1. Multiprocessing and Lock
        # https://stackoverflow.com/questions/70516936/python-multiprocessing-api-query            
        # https://dmort-ca.medium.com/part-2-multiprocessing-api-requests-with-python-19e593bd7904
            
        # 2. Multiprocessing example giving AttributeError
        # https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror

        # stop_loss()


        # if mode == 1:
        #     Stimulate()
            
        # elif mode == 2:
        #     # Enhance Learning()
        #     pass
            
        
        # if sell_signal:
        #     # - sell with strategy[i]
        #     # - notification
        #     pass
        
        
        # # Export
        # if excetue_time % 60:
        #     # pd.to_csv()
        #     pass


        sys.sleep(2)






# %% Archive ------


def multiplrocessing_archive():
    
    # - https://cslocumwx.github.io/blog/2015/02/23/python-multiprocessing/
    # Since Python will only run processes on available cores, setting 
    # max_number_processes to 20 on a 10 core machine will still mean that
    # Python may only use 8 worker processes.
    
    # - https://stackoverflow.com/questions/70516936/python-multiprocessing-api-query
    
    
    # - Multiprocessing for heavy API requests with Python and the PokéAPI
    #   https://hackernoon.com/multiprocessing-for-heavy-api-requests-with-python-and-the-pokeapi-3u4h3ypn
    
    # cpu_count = mp.cpu_count()
    # cpu_count = max(cpu_count -1, 1)
    # cpu_count = 2 # Dev 
    # max_number_processes = cpu_count
    
    
    # Test 1 ......
    # with mp.Pool(5) as p:
    #     print(p.map(tbf.test1_fun, [1, 2, 3]))    


    # # Test 2 ......
    # with mp.Pool(5) as p:
    #     print(p.map(tbf.test2_fun, [1, 2, 3]))
     
    
    # multiprocessing_lock = multiprocessing.Lock()
    

    # pool = multiprocessing.Pool(cpu_count)
    # total_tasks = 12
    # tasks = range(total_tasks)
    # results = pool.map_async(tbf.api_query_process, tasks)
    # pool.close()
    # pool.join()    
    
    
    # def locked_api_query_process(cloud_type, api_name, cloud_account, resource_type):
    #     with multiprocessing_lock:
    #         api_query_process(cloud_type, api_name, cloud_account, resource_type)
    
    # with multiprocessing.Pool(processes=cpu_count) as pool:
    #     jobs = []
    #     for _ in range(12):
    #         jobs.append(pool.apply_async(locked_api_query_process(*args)))
    #     for job in jobs:
    #         job.wait()
    
    
    # def locked_api_query_process():
    #     with multiprocessing_lock:
    #         api_query_process()
    
    # with multiprocessing.Pool(processes=cpu_count) as pool:
    #     jobs = []
    #     for _ in range(12):
    #         jobs.append(pool.apply_async(locked_api_query_process()))
            
    #     for job in jobs:
    #         job.wait()

    pass


# %% Execution ------
    
    
if __name__ == "__main__":    
    
    
    pass
    

