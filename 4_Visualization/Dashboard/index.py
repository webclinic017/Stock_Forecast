#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:04:08 2020

@author: Aron
"""

"""
Version Note -----

Complete ...
1. Add dropdown
2.2 Add dash cache
2.3 Add three year cache
2.4 Add mobile layout
2.5 Change chart structure
2.5.1 Restore dcc
2.5.3 Fixe dropdown issue
2.5.4 Optimize layout

In progress
Add stock_info_tw to database


Desktop name dropdown - width issue

Add multiple chart


Bug ...


Optimization ...
1.Add loading icon
因為dash一開始的物件是空的，導致dashboard會先呈現扁平狀態。
Solution: add sample chart

"""


import os
import pandas as pd
import numpy as np
import sys


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from flask_caching import Cache

import urllib.parse as urlparse
from urllib.parse import parse_qs


# Worklist
# 1. Plot color, 國外的常用顏色和台灣相反，綠是漲，紅是跌



# 設定工作目錄 .....
local = False
# local = True


if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis/4_Visualization/Dashboard'
else:
    path = '/home/aronhack/stock_forecast/4_Visualization/Dashboard'
    


# Codebase
path_codebase = ['/Users/Aron/Documents/GitHub/Arsenal',
                 '/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path, 
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import arsenal as ar
import arsenal_stock as stk
import codebase_yz as cbyz


# from app_master import app
import app_master as ms
import desktop_app
import mobile_app


# 手動設定區 -------

# stock_type = 'us'
# stock_type = 'tw'



# 自動設定區 -------
pd.set_option('display.max_columns', 30)



# %% Layout ----

app = dash.Dash()


if local == False:
    cache = Cache(app.server, config={
        # try 'filesystem' if you don't want to setup redis
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
    })
    app.config.suppress_callback_exceptions = True
    

colors = {
    'background': '#ffffff',
    # 'background': '#f5f5f5',
    'text': '#303030'
}


app_main_style = {
    'backgroundColor': colors['background'],
    # 'padding': '0 30px',
    'min-height': '650px',
    # 'display':'flex',
    }


title_style = {
    'textAlign': 'left',
    'color': '#303030',
    'padding-top': '20px',
    'disply': 'inline-block',
    'width': '50%'
    }

stk_selector_style = {
    'width': '100%', 
    'padding-top': '40px', 
    'display': 'block',
    }

stk_selector_style_desktop = {
    'width': '70%', 
    # 'padding-top': '40px', 
    'display': 'block',
    'height': 'auto',
    }

stk_selector_style_mobile = {
    'width': '100%', 
    # 'padding-top': '40px', 
    'display': 'block',
    'height': 'auto',
    }


data_period_style = {
    # 'width': '100px', 
    'display': 'flex',
    'justify-content': 'left',
    'align-items': 'center',
    'padding-top': '10px',
    }



# %% Application ----


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='url_debug'),
    
    # Settings
    dcc.Input(id="device", type='hidden', value=0),    
    dcc.Input(id="tick0", type='hidden', value=ms.first_date_lite),    
    dcc.Input(id="dtick", type='hidden', value=20),
    
    
    html.Div([
        dcc.Dropdown(
            id='stk_selector',
            options=ms.stock_list,
            multi=True,
            value=[]
        ),
    ], style=stk_selector_style),
    
    html.Div([
        html.P('半年資料'),
        daq.ToggleSwitch(
            id='data_period',
            value=False,
            style={'padding':'0 10px'}
        ),
        html.P('三年資料'),
    ], style=data_period_style),      

    
    # html.Div(id='app_main', 
    #          style=app_main_style),
    
    dcc.Graph(id="graph"),
    
    ]
)



# %% Callback ----

# Output(component_id='stk_selector', component_property='style'),
# Input(component_id='url', component_property='search'),

@app.callback(
    Output('stk_selector', 'style'),
    Output('device', 'value'),    
    Input('url', 'search'),
    Input('device', 'value'),
    Input('stk_selector', 'style')
)


def get_url(url, device, style):


    if url == '':
        device = 'desktop'
        loc_style = stk_selector_style_desktop
        return loc_style

    
    parsed = urlparse.urlparse(url)
    width = parse_qs(parsed.query)['w'][0]
    
    
    if int(width) < 992:
        device = 1
        loc_style = stk_selector_style_mobile
        print('mobile width')
        # return mobile_app.layout
    else:
        device = 0
        loc_style = stk_selector_style_desktop
        print('desktop width')
        # return desktop_app.layout
        

    # return ''
    return loc_style, device
        
    
# ................


@app.callback(
    Output('tick0', 'value'),
    Output('dtick', 'value'),
    Input('device', 'value'),
    Input('data_period', 'value')
)

def update_tick_attr(device, data_period):

    # Update, different settings for desktop and mobile    
    if device == 0:
        if data_period:
            return ms.first_date, 240
        else:
            return ms.first_date_lite, 60
    else:
        if data_period:
            return ms.first_date, 240
        else:
            return ms.first_date_lite, 100       


# ................


@app.callback(
    Output('graph', 'figure'),
    Input('device', 'value'),
    Input('stk_selector', 'value'),
    Input('data_period', 'value'),
    Input('tick0', 'value'),
    Input('dtick', 'value'),    
)

def update_output(device, dropdown_value, data_period, tick0, dtick):


    # Figure ......
    fig = go.Figure()
    
    for i in range(len(dropdown_value)):

        s = dropdown_value[i]  
        name = ms.stock_list_raw[ms.stock_list_raw['STOCK_SYMBOL']==s]
        name = name['STOCK'].tolist()[0]
        
        
        # Filter Data ......
        if data_period:
            df = ms.main_data[ms.main_data['STOCK_SYMBOL']==s] \
                .reset_index(drop=True) 
        else:
            df = ms.main_data_lite[ms.main_data_lite['STOCK_SYMBOL']==s] \
                .reset_index(drop=True)    
                
        trace = go.Candlestick(
            x=df['WORK_DATE'],
            open=df['OPEN'],
            high=df['HIGH'],
            low=df['LOW'],
            close=df['CLOSE'],
            name=name
        )
        
        fig.add_trace(trace)



    # Layout ------
    
    layout = {'plot_bgcolor': colors['background'],
              'paper_bgcolor': colors['background'],
              'font': {
                  'color': colors['text']
                  },
              'xaxis':{'title':'日期',
                       'fixedrange':True},
              'yaxis':{'title':'收盤價',
                       'fixedrange':True},
              }


    # Legend Layout ......
    if device == 0:
        legend_style = dict()    
    else:
        legend_style = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
        

    # 1. In plotly, there are rangebreaks to prevent showing weekends and 
    #    holidays, but the weekends and holidays may be different in Taiwan. 
    #    As a results, the alternative way to show it is to show as category
    
    fig.layout = dict(xaxis={'type':"category", 
                            'categoryorder':'category ascending',
                            'tickmode':'linear',
                            'tick0':tick0,
                            'dtick':dtick,
                            })

    # Plotly doesn't have padding?
    # 'padding':{'l':0, 'r':0, 't':20, 'b':20}

    mobile_layout = {'legend':legend_style,
                     'margin':{'l':36, 'r':36, 't':80, 'b':80}
                     }

    fig.update_layout(mobile_layout)
    fig.update_layout(xaxis_rangeslider_visible=False)



    return fig

if __name__ == '__main__':
    
    app.run_server()
    # app.run_server(debug=True)




# Test -------------
    
# stock_name
    
# other-listed
# import requests
# link2 = 'https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.json'
# r2 = requests.get(link2)
# results2 = pd.DataFrame(r2.json())
# results2.columns


# chk = results.copy()
# chk = chk[chk['STOCK_SYMBOL']=='MSFT']
    
    

# dict1 = {'VALUE1':1, 'VALUE2':2}
# dict2 = str(dict1)


# test = ar.df_cross_join(stock_name, remove_same=True)

# test = test[['STOCK_SYMBOL_x', 'STOCK_SYMBOL_y']]
# test.columns = ['STOCK_Y', 'STOCK_VAR']
# test['RULE_ID'] = 1
# test['RESULTS'] = dict2
# test['STOCK_Y'] = test['STOCK_Y'].astype(str)
# test = test[['RULE_ID', 'STOCK_Y', 'STOCK_VAR', 'RESULTS']]


# upload = test.iloc[0:10000, :]
# upload.to_csv(path + '/stock_rule.csv', index=False)


# ar.db_upload(upload, 'stock_rule_tw', True)






