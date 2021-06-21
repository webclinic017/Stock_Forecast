#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:23:08 2020

@author: Aron
"""


token = "68481182df39d05fad4b8234bbd17a15"


# %% HTTPS ------

import requests
import pandas as pd
import json
url = "https://api.fugle.tw/realtime/v0.2/intraday/quote?symbolId=2603&apiToken=" + token

response = requests.get(url)
content = response.content

df = json.loads(content.decode("utf-8").replace("'",'"'))
# df = pd.DataFrame.from_dict(df['data'])

dict(df)



# %% Websocket ------
import websocket

def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")


if __name__ == "__main__":
    websocket.enableTrace(False)
    # ws = websocket.WebSocketApp("wss://api.fugle.tw/realtime/v0/intraday/quote?symbolId=2884&apiToken=demo",
    #                           on_message = on_message,
    #                           on_error = on_error,
    #                           on_close = on_close)
    
    
    ws = websocket.WebSocketApp("wss://api.fugle.tw/realtime/v0.2/intraday/quote?symbolId=2884&apiToken="+token,
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    
    ws.run_forever()





# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc
from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_squared_error


local = False
local = True

stock_type = 'tw'

# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/2_Stock_Analysis'
else:
    path = '/home/aronhack/stock_predict/2_Stock_Analysis'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ',]


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk



# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

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










