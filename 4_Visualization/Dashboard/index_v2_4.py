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

In progress
2.4 Add mobile layout
2.5 Add multiple chart


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
import dash_bootstrap_components as dbc


#import re
#import numpy as np
# from flask import Flask, request
from flask_caching import Cache


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
import codebase_yz as cbyz


from app_master import app
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


# # 自動設定區 -------
pd.set_option('display.max_columns', 30)




def check():
    '''
    資料驗證
    '''    
    
    chk = main_data[main_data['STOCK_SYMBOL']=='00776']
    
    chk = stock_list_pre[stock_list_pre['STOCK_SYMBOL']=='00776']
    
    chk = plot_data[plot_data['STOCK_SYMBOL']=='00776']
    
    
    return ''



# %% Application ----


if local == False:
    cache = Cache(app.server, config={
        # try 'filesystem' if you don't want to setup redis
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
    })
    app.config.suppress_callback_exceptions = True
    

colors = {
    'background': '#f5f5f5',
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

name_dropdown_style = {
    'width': '50%', 
    'padding-top': '40px', 
    'display': 'block'
    }

btn_max_style = {
    # 'width': '100px', 
    'display': 'flex',
    'justify-content': 'left',
    'align-items': 'center',
    'padding-top': '10px',
    }

debug_style = {
    'display': 'none',
    }


figure_style = {
    # 'transform': 'scale(1.1)'
    }



# Original
# app.layout = html.Div([
    
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='url_debug'),
        
#     html.Div([
#         dcc.Dropdown(
#             id='name_dropdown',
#             options=ms.stock_list,
#             multi=True,
#             value=[]
#         ),
#     ],style=name_dropdown_style),
    
#     html.Div([
#         html.P('半年資料'),
#         daq.ToggleSwitch(
#             id='btn_max',
#             value=False,
#             style={'padding':'0 10px'}
#         ),
#         html.P('三年資料'),
#     ],style=btn_max_style),        
    
    
    
#     html.Div(id='debug',
#              style = debug_style),
    
    
#     html.Div(id="line_chart"),
#     # html.P('Data Source'),
#     ],  
#     id='app',
#     style=container_style
# )



app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='url_debug'),
    html.Div(id='app_main', style=app_main_style)
    
    ]
)


@app.callback(
    Output(component_id='app_main', component_property='children'),
    Input(component_id='url', component_property='search')
)


def get_url(url):
    
    if url != '':
        print('blank')

    
    parsed = urlparse.urlparse(url)
    width = parse_qs(parsed.query)['w'][0]
    
    print(width)
    
    if width != '' and int(width) < 992:
        print('mobile_app')
        return mobile_app.layout
    else:
        print('desktop_app')
        return desktop_app.layout
        
    


@app.callback(
    Output(component_id='line_chart', component_property='children'),
    Output(component_id='debug', component_property='children'),
    Input(component_id='name_dropdown', component_property='value'),
    Input(component_id='btn_max', component_property='value')  
)


def update_output(dropdown_value, time_switch_value):

    
    print(dropdown_value)
    
    
    if dropdown_value == None:
        print('dropdown_value is None')
        return ''
    
    
    # global stock_list_pre
    selected_list = ms.stock_list_pre[
        ms.stock_list_pre['STOCK_SYMBOL'].isin(dropdown_value)] \
                    .reset_index(drop=True)
    

    if time_switch_value == True:
        
        plot_data = ms.main_data
        
    else:
        # global main_data_lite
        plot_data = ms.main_data_lite
        
        
    data1 = [{'x': plot_data[(plot_data['STOCK_SYMBOL'] == \
                              selected_list['STOCK_SYMBOL'][i]) & \
                      (plot_data['TYPE'] == 'HISTORICAL')]['WORK_DATE'],
              'y': plot_data[(plot_data['STOCK_SYMBOL'] == \
                              selected_list['STOCK_SYMBOL'][i]) & \
                      (plot_data['TYPE'] == 'HISTORICAL')]['CLOSE'],
              'type': 'line',
              'name': selected_list['STOCK'][i],
              } for i in range(0, len(selected_list))]
        

    historical_plot =  dcc.Graph(
                            id='main_graph',
                            figure={
                                'data': data1,
                                'layout': {
                                    'plot_bgcolor': colors['background'],
                                    'paper_bgcolor': colors['background'],
                                    'font': {
                                        'color': colors['text']
                                    },
                                },
                            },
                            style=figure_style 
                        )


        

    # figure = [historical_plot, historical_plot]
    figure = historical_plot


    # Debug
    debug_dropdown = '_'.join(dropdown_value)
    

    # return figure
    return figure, 'ID_'+debug_dropdown+'_MAX_'+str(time_switch_value)+'_'



if __name__ == '__main__':
    app.run_server()
    # app.run_server(debug=True)


def execution():
    app.run_server()




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

