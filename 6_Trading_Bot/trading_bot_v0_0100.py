#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

History

20210703 - Replaced MA with WMA in SAM

"""


# Link
# https://developer.fugle.tw/
# https://developer.fugle.tw/docs/trading/tutorial/trade



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
# import fugle_v1_0200_dev as arfg

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
    


def master_level_1():
    
    # Version Plan ......
    # - Level 1只能賣不能買，設一些簡單的條件停損或是在下跌的時候獲利了結
    # - Level 2是每天給幾萬塊的扣打，只能從選我給的幾個標的中買進
    # - Level 3才讓模型自選標的        
    
    
    # v0.0100
    # - First version
    
    # v0.0200
    # - 考慮query_quote時剛好刷新max high的記錄
    
    
    # Worklist
    # - New symbol strategy，大部份新股在上市櫃後是否都會上漲，如果再加停損不停利策略，
    #   這樣勝率是不是會很高
    # - Set global complete marke_data in stk
    # - Review buy_signa, did the price of symbol increase in the past days
    # - 您好，從您的截圖看不出來您修改了哪些部分。如果您是想要 log websocket 的 error message，可以修改部分程式碼， 使用 on_error 這個 hook 來記錄。
    #   https://github.com/fugle-dev/fugle-realtime-python/blob/master/fugle_realtime/websocket_client/ws.py#L6    
    
    
    # Sell Signal
    # - 停損不停利策略
    # - 事先預判有多少法人持股，如果數量很少，當他們賣的時候就要跟著賣了
        

    inv_raw = sdk.get_inventories()    
    # [{'ap_code': '',
    #   'cost_qty': '1000',
    #   'cost_sum': '-29341',
    #   'make_a_per': '-0.92',
    #   'make_a_sum': '-269',
    #   'price_avg': '29.30',
    #   'price_evn': '29.48',
    #   'price_mkt': '29.20',
    #   'price_now': '29.20',
    #   'price_qty_sum': '29300',
    #   'qty_b': '1000',
    #   'qty_bm': '1000',
    #   'qty_c': '0',
    #   'qty_l': '0',
    #   'qty_s': '0',
    #   'qty_sm': '0',
    #   'rec_va_sum': '29072',
    #   'stk_na': '華邦電',
    #   'stk_no': '2344',
    #   's_type': 'H',
    #   'trade': '0',
    #   'value_mkt': '29200',
    #   'value_now': '29200',
    #   'stk_dats': [{'buy_sell': 'B',
    #     'cost_r': '0',
    #     'fee': '41',
    #     'make_a': '-269',
    #     'make_a_per': '-0.92',
    #     'ord_no': '60077016896670',
    #     'pay_n': '-29341',
    #     'price': '29.30',
    #     'price_evn': '29.48',
    #     'qty': '1000',
    #     'qty_c': '0',
    #     'qty_h': '0',
    #     'qty_r': '0',
    #     't_date': '20220518',
    #     'tax': '0',
    #     'tax_g': '0',
    #     'trade': '0',
    #     't_time': '131055213',
    #     'value_mkt': '29200',
    #     'value_now': '29200'}]}]
    
    
    # 變成function？
    inv = []
    for i in range(len(inv_raw)):
        new_inv = [inv_raw['stk_no'], inv_raw['price_evn'], inv_raw['qty_b']]
        inv.append(inv)
        
    inv = pd.DataFrame(inv, columns=['SYMBOL', 'COST', 'VOLUME'])
    hold = inv['SYMBOL'].tolist()    
    
    
    
    t_date': '20220518'，這裡應該可以直接抓到交易日期，找出first trading date
    


    # Get Mean Cost ......
    # - Unessential after apply API
    
    # # Develop
    # ledger, hold_volume, hold_data = \
    #     stk.get_ledger(begin_date=20220101, end_date=20220120, market=market)
    
    # # Production
    # ledger, hold_volume, hold_data = \
    #     stk.get_ledger(begin_date=None, end_date=None, market=market)

    # # Hold Symbol
    # hold = hold_volume['SYMBOL'].tolist()
    
    # # Mean Cost
    # hold_data = hold_data \
    #             .groupby(['SYMBOL']) \
    #             .agg({'WORK_DATE':'min',
    #                   'VOLUME':'sum',
    #                   'COST':'sum'}) \
    #             .reset_index()
    
    # hold_data['COST_MEAN'] = hold_data['COST'] / hold_data['VOLUME']
    
    
    # First BUy
    first_buy = hold_data[['SYMBOL', 'WORK_DATE']] \
                .sort_values(by=['SYMBOL', 'WORK_DATE'], ascending=True) \
                .drop_duplicates(subset=['SYMBOL']) \
                .reset_index(drop=True)
                
    first_buy['KEEP'] = 1
    
    
    # Market Data ......
    min_date = hold_data['WORK_DATE'].min()
    today = cbyz.date_get_today()
    
    market_data = stk.get_data(data_begin=min_date, data_end=today, 
                               market=market, restore=True, unit='d', 
                               symbol=symbol, ratio_limit=True, 
                               price_change=False, price_limit=True,
                               trade_value=False)
    
    # Stop Loss
    stop_loss = market_data \
        .merge(first_buy, how='left', on=['SYMBOL', 'WORK_DATE'])
        
    stop_loss = cbyz.df_fillna(df=stop_loss, cols='KEEP', 
                               sort_keys=['SYMBOL', 'WORK_DATE'], 
                               group_by=['SYMBOL'], method='ffill')
    
    stop_loss = stop_loss.dropna(subset=['KEEP'], axis=0)
    stop_loss = stop_loss \
                .groupby(['SYMBOL']) \
                .agg({'HIGH':'max'}) \
                .reset_index()
                
    stop_loss = stop_loss \
        .merge(hold_data, how='left', on='SYMBOL')
     
    # 本來設定0.8，但0.8很容易超過
    stop_loss['STOP_LOSS'] = stop_loss['COST_MEAN'] \
        + (stop_loss['HIGH'] - stop_loss['COST_MEAN']) * 0.7
    

    # Fugle API ......
    # HTTP request 每分鐘	60
    # WebSocket 連線數 5    
    
    
    # Worklist ......
    # - GCP always on schedule
    # - GCP ploty
    # - GCP collect Fugle data on free VM



    
    # Schedule to restart
    # connnect_sql()
    
    # ledger = stk.read_ledger()
    
    # target = [2601]
    # ignore_symbol = []
    
    
    while 2 > 1:
        
        # Bug, query quote一定要從arfg啟動嗎？還是可以從其他fuction呼叫，後者的話還需要
        # 儲存嗎
        arfg.query_quote(hold)
        
        # Reference ......
        # # - Can I request multiple symbols from fugle
        # for s in symbols:
        #     master_single()
            

        # stop_loss()
        
        

        if sell:
            hold.remove(sell_symbol)


        # if mode == 1:
        #     Simulate()
            
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


        sys.sleep(1)








# %% Archive ------


def multiplrocessing_archive():

    # 1. Multiprocessing and Lock
    # https://stackoverflow.com/questions/70516936/python-multiprocessing-api-query            
    # https://dmort-ca.medium.com/part-2-multiprocessing-api-requests-with-python-19e593bd7904
        
    # 2. Multiprocessing example giving AttributeError
    # https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror

    
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


# %% Test API ------
    



# %% Execution ------
    
    
if __name__ == "__main__":    
    
    
    pass
    

