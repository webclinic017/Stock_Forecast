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
import sys, arrow
import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
# import dash_bootstrap_components as dbc


#import re
#import numpy as np
# from flask import Flask, request
from flask_caching import Cache

import flask_caching


# Parse url
import urllib.parse as urlparse
from urllib.parse import parse_qs


# Worklist
# (1) Convert get_stock_data
# (2) stock_get_list, upload to database

# 設定工作目錄 .....

local = False
local = True


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


# # 手動設定區 -------
# global begin_date, end_date
# end_date = datetime.datetime.now()
# end_date = cbyz.date_simplify(end_date)

# begin_date_6m = cbyz.date_cal(end_date, amount=-6, unit='m')
# begin_date_3y = cbyz.date_cal(end_date, amount=-3, unit='y')



# stock_type = 'us'
# stock_type = 'tw'

global device

# # 自動設定區 -------
pd.set_option('display.max_columns', 30)




def check():
    '''
    資料驗證
    '''        
    
    return ''



# %% Application ----

# app = dash.Dash(__name__, suppress_callback_exceptions=True)
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


btn_max_style = {
    # 'width': '100px', 
    'display': 'flex',
    'justify-content': 'left',
    'align-items': 'center',
    'padding-top': '10px',
    }


figure_style = {
    # 'transform': 'scale(1.1)'
    }






app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='url_debug'),
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
            id='btn_max',
            value=False,
            style={'padding':'0 10px'}
        ),
        html.P('三年資料'),
    ], style=btn_max_style),      

    
    html.Div(id='app_main', 
             style=app_main_style),
    
    dcc.Graph(id="graph"),
    
    ]
)



@app.callback(
    # Output(component_id='app_main', component_property='children'),
    Output(component_id='stk_selector', component_property='style'),
    Input(component_id='url', component_property='search'),
    Input(component_id='stk_selector', component_property='style')
)


def get_url(url, style):

    
    # Bug, 有兩個id=stk_selector
    
    global device
    

    if url == '':
        device = 'desktop'
        loc_style = stk_selector_style_desktop
        return loc_style

    
    parsed = urlparse.urlparse(url)
    width = parse_qs(parsed.query)['w'][0]
    
    
    if int(width) < 992:
        device = 'mobile'
        loc_style = stk_selector_style_mobile
        # return mobile_app.layout
    else:
        device = 'desktop'
        loc_style = stk_selector_style_desktop
        # return desktop_app.layout
        

    # return ''
    return loc_style
        
    

# Output(component_id='line_chart', component_property='children'),
# Output(component_id='debug', component_property='children')
# Output(component_id='app_main', component_property='children'),
# Output(component_id='graph', component_property='figure'),

@app.callback(
    Output(component_id='app_main', component_property='children'),
    Input(component_id='stk_selector', component_property='value'),
    Input(component_id='btn_max', component_property='value')  
)


def update_output(dropdown_value, time_switch_value):

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
    if device == 'desktop':
        legend_style = dict()    
        graph_style = {}
    else:
        legend_style = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
        


    mobile_layout = {'legend':legend_style,
                      'margin':{'l':36, 'r':36, 't':80, 'b':80},
                      'padding':{'l':0, 'r':0, 't':20, 'b':20}
                      }
    
    layout.update(mobile_layout)
    
    # Graph Style
    graph_style = {'border': 'solid 1px #b0b0b0',
                   'height': 300
                   }
    graph_style.update(graph_style)


    



    # Worklist 
    # 2. add legend name

    # https://plotly.com/python/candlestick-charts/

    fig = go.Figure()
    
    for i in range(len(dropdown_value)):

        s = dropdown_value[i]  
        df = ms.main_data[ms.main_data['STOCK_SYMBOL']==s] \
            .reset_index(drop=True)        

        
        trace = go.Candlestick(
            x=df['WORK_DATE'],
            open=df['OPEN'],
            high=df['HIGH'],
            low=df['LOW'],
            close=df['CLOSE'],
            name=s
        )
        
        fig.add_trace(trace)


    fig.layout = dict(xaxis=dict(type="category", 
                                  categoryorder='category ascending'))
    

    results = dcc.Graph(
                id='main_graph',
                figure=fig,
                style=graph_style
            )
    

    return results



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






