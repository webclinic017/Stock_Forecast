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


from configparser import ConfigParser
from fugle_trade.sdk import SDK
from fugle_trade.order import OrderObject
from fugle_trade.constant import (APCode, Trade, PriceFlag, BSFlag, Action)    



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
                 r'/home/jupyter/Codebase_YZ/20220519',
                 r'/home/jupyter/Arsenal/20220519',
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
import fugle_v1_0200_dev as arfg

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
    


def connect_account():


    # New Vars
    global inv_raw, inv, symbol, sdk


    # Login
    ar.get_black_box()
    
    # Prevent creating multiple connection
    if 'sdk' not in globals():
        
        config = ConfigParser()
        # config.read(ar.path_black_box + '/config.simulation.ini')
        config.read(ar.path_black_box + '/config.ini')
        
        sdk = SDK(config)
        sdk.login()
        time.sleep(5)

    
    inv_raw = sdk.get_inventories()    
    time.sleep(3)
    
    # stk_no 股票代碼
    # cost_qty 成本股數
    # price_evn 損益平衡價
    
    # stk_dats 庫存明細
    
    
    # 變成function？
    inv = []
    for i in range(len(inv_raw)):
        
        cur_inv = inv_raw[i]

        # Extract first trading date
        stk_dats = cur_inv['stk_dats']
        first_buy = 29990000
        for j in range(len(stk_dats)):
            
            cur_dats = stk_dats[j]
            t_date = int(cur_dats['t_date'])
            first_buy = t_date if t_date < first_buy else first_buy
        
        new_inv = [cur_inv['stk_no'], cur_inv['price_evn'], 
                   cur_inv['cost_qty'], first_buy]
        
        inv.append(new_inv)        
        
        
    inv = pd.DataFrame(inv, columns=['symbol', 'cost', 'volume', 'first_buy'])
    inv = cbyz.df_conv_col_type(df=inv, cols=['cost'], to='float')
    
    symbol_df = inv[['symbol']]
    symbol = inv['symbol'].tolist()    
        


def gen_stop_loss():


    connect_account()


    # Market Data ......
    min_date = inv['first_buy'].min()
    today = cbyz.date_get_today()
    
    market_data = stk.get_data(data_begin=min_date, data_end=today, 
                               market=market, adj=True, unit='d', 
                               symbol=symbol, ratio_limit=True,
                               price_change=False, price_limit=False,
                               trade_value=True)
    
    market_data = cbyz.df_col_lower(market_data)
    market_data = market_data[['symbol', 'work_date', 'high']]
    
    
    
    # Stop Loss
    stop_loss = market_data.merge(inv, how='left', on=['symbol'])
    stop_loss = stop_loss[stop_loss['work_date']>=stop_loss['first_buy']]

    stop_loss = stop_loss \
                .groupby(['symbol']) \
                .agg({'high':'max'}) \
                .reset_index()
                
    stop_loss = stop_loss.merge(inv, how='left', on='symbol')
     
    stop_loss['stop_loss_pos'] = stop_loss['cost'] \
        + (stop_loss['high'] - stop_loss['cost']) * stop_loss_pos

    stop_loss['stop_loss_neg'] = stop_loss['cost'] * stop_loss_neg

    return stop_loss



def get_sell_order_result():
    
    global sdk
    raw = sdk.get_order_results()
    
    result = []
    for i in range(len(raw)):
        
        cur_order = raw[i]
        
        is_sell = cur_order['buy_sell']=='S'
        qty = cur_order['org_qty'] > cur_order['cel_qty']
        
        if is_sell and qty > 0:
            result.append([cur_order['stock_no'], qty])
            
    result = pd.DataFrame(result, columns=['symbol', 'qty'])
    return result


def master_level_1():
    
    
    global stop_loss_pos, stop_loss_neg
    stop_loss_pos = 0.7
    stop_loss_neg = 0.95
    
    
    global symbol, sdk
    global stop_loss
    
    
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
    ohlc = stk.get_ohlc()
    ohlc = cbyz.li_lower(ohlc)

    
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
    
    switch = True
    print('Bug - 即時行情沒有合併進market_data中')
    
    while 2 > 1:
        
        # Bug, query quote一定要從arfg啟動嗎？還是可以從其他fuction呼叫，後者的話還需要
        # 儲存嗎
        
        if switch:
            # Update, 因為目前Trading Bot只賣不買，所以這一段可以這樣寫
            stop_loss = gen_stop_loss()
            time.sleep(5)
            switch = False
        
        order = get_sell_order_result()
        order_symbol = order['symbol'].tolist()
        
        
        for i in range(len(symbol)):
            cur_symbol = symbol[i]
            
            # 避免重複下單
            if cur_symbol in order_symbol:
                continue
        
            arfg.query_quote(cur_symbol)
            data = arfg.data_raw
            
            # 最新一筆成交記錄
            price = data['data']['quote']['trade']['price']
            
            
            temp_stop_loss = stop_loss[stop_loss['symbol']==cur_symbol] \
                            .reset_index(drop=True)
                            
            temp_pos = temp_stop_loss.loc[0, 'stop_loss_pos']
            temp_neg = temp_stop_loss.loc[0, 'stop_loss_neg']
            temp_cost = temp_stop_loss.loc[0, 'cost']
            
            cond_pos = price <= temp_pos \
                and (price - temp_cost) / temp_cost >= 0.02
            
            print(cur_symbol, '-', price, temp_cost, temp_pos, temp_neg)
            
            if (cond_pos) or (price <= temp_neg):
                
                temp_volume = int(temp_stop_loss.loc[0, 'volume'])
                temp_volume = int(temp_volume / 1000) 
                
                try:
                    order = OrderObject(
                        buy_sell = Action.Sell,
                        price_flag = PriceFlag.Limit,
                        price = price - 0.5,
                        stock_no = cur_symbol,
                        quantity = temp_volume,
                    )
                    sdk.place_order(order)      
                    
                    switch = True
                    pass
                    
                except Exception as e:
                    print(e)
                
        # Reference ......
        # # - Can I request multiple symbols from fugle
        # for s in symbols:
        #     master_single()
            

        # stop_loss()
        


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

        time.sleep(5)
        
        serial = cbyz.get_time_serial(with_time=True, to_int=True)
        print(serial, '------')
        
        
        cur_time = serial % 1e6
        if cur_time > 133500:
            break
        
        if len(order) > 0:
            print(order)





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
    
def test_api():
    
    from configparser import ConfigParser
    from fugle_trade.sdk import SDK
    from fugle_trade.order import OrderObject
    from fugle_trade.constant import (APCode, Trade, PriceFlag, BSFlag, Action)    

    
    config = ConfigParser()
    # config.read('./config.simulation.ini')
    # config.read(r'/home/jupyter/esun/simulation/config.simulation.ini')
    # config.read(r'/home/jupyter/esun/config.ini')
    
    
    # Local
    ar.get_black_box()
    config.read(ar.path_black_box + '/config.simulation.ini')
    config.read(ar.path_black_box + '/config.ini')
    
    sdk = SDK(config)
    sdk.login()


    # Custom
    # order = OrderObject(
    #     buy_sell = Action.Buy,
    #     price_flag = PriceFlag.Limit,
    #     price = 26.8,
    #     stock_no = "2344",
    #     quantity = 1,
    # )
    # sdk.place_order(order)
    # print("Your order has been placed successfully.")

    
def check():
    
    # 註冊當 websocket 發生錯誤時的 callback
    @sdk.on('error')
    def on_error(err):
        print(err)
    
    # 註冊接收委託回報的 callback
    @sdk.on('order')
    def on_order(data):
        print(data)
    
    # 註冊接收成交回報的 callback
    @sdk.on('dealt')
    def on_dealt(data):
        print(data)
    
    # 定義完註冊的部分之後，需要透過 websocket 建立連線，才能收到後續的回報
    sdk.connect_websocket()    
        



# %% Execution ------
    
    
if __name__ == "__main__":    
    
    
    master_level_1()
    

