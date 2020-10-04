#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:04:08 2020

@author: Aron
"""

"""
Version Note
1. Add dropdown
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
import pandas as pd
import sys, arrow

#import os
#import re
#import numpy as np
from flask import Flask, request


# 設定工作目錄 .....
path = '/Users/Aron/Documents/GitHub/Data/Stock-Forecast'

# Codebase
path_codebase = ['/Users/Aron/Documents/GitHub/Arsenal',
                 '/Users/Aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import arsenal as ar
import codebase_yz as cbyz



# 手動設定區 -------
global begin_date, end_date
end_date = arrow.now()
begin_date = end_date.shift(months=-6)
# begin_date = end_date.shift(years=-1)


end_date = end_date.format('YYYYMMDD')
end_date = int(end_date)

begin_date = begin_date.format('YYYYMMDD')
begin_date = int(begin_date)





# 自動設定區 -------
pd.set_option('display.max_columns', 30)


def init(path):
    

    return ''



def load_data():
    '''
    讀取資料及重新整理
    '''

    # Load Data --------------------------
    
    # Historical Data .....
    global begin_date, end_date, stock_data
    stock_data = get_stock_data(begin_date, end_date)
    
    
    # Forecast Data .....
    # sql_forcast_data = "select work_date, STOCK_SYMBOL, " + \
    #                "high_price, close_price, low_price " + \
    #                "from stock_forecast;"
    
    # forecast_data = ar.db_query(sql_forcast_data)
    # forecast_data.columns = cols
    # forecast_data['TYPE'] = 'FORECAST'
    
    # forecast_data
    
    
    # Stock Name .....
    stock_name = get_stock_name()


    # Work Area -------------
    global main_data
    main_data = stock_data.copy()
    main_data = main_data.sort_values(by=['STOCK_SYMBOL', 'WORK_DATE']) \
                .reset_index(drop=True)
    
    main_data = main_data.merge(stock_name, 
                                how='left', 
                                on=['STOCK_SYMBOL'])
    
    main_data['TYPE'] = 'HISTORICAL'
    main_data['WORK_DATE'] = main_data['WORK_DATE'].apply(ar.ymd)
    
    main_data['STOCK'] = (main_data['STOCK_SYMBOL'] 
                          + ' ' 
                          + main_data['NAME'])
    
    # Dash ----------------------
    
    # Stock List
    global stock_list, stock_list_pre
    stock_list_pre = (main_data[['STOCK_SYMBOL', 'STOCK']]
                    .drop_duplicates()
                    .reset_index(drop=True)
                    )
                    
    stock_list = []
    
    for i in range(0, len(stock_list_pre)):
        stock_list.append({'label': stock_list_pre.loc[i, 'STOCK'],
                           'value': stock_list_pre.loc[i, 'STOCK_SYMBOL']
                           })

    
    return ''


def master():
    '''
    主工作區
    '''
    load_data()
    return ''



def check():
    '''
    資料驗證
    '''    
    return ''


# %% Inner Function -------

def get_stock_data(begin_date=None, end_date=None, stock=None):
    
    
    # Convert stock to list
    stock_li = [stock]
    stock_li = cbyz.list_flatten(stock_li)    
    
    
    if (begin_date != None) & (end_date != None):
        
        if stock != None:
            sql_stock = cbyz.list_add_quote(stock_li, "'")
            sql_stock = ', '.join(sql_stock)
            sql_stock = ' and stock_symbol in (' + sql_stock + ')'
        else:
            sql_stock = ''
            
        sql_cond = (" where date_format(work_date, '%Y%m%d') between " + str(begin_date) + " and " + str(end_date) +
                    sql_stock)
        
    else:
        sql_stock = cbyz.list_add_quote(stock_li, "'")
        sql_stock = ', '.join(sql_stock)
        sql_cond = ' where stock_symbol in (' + sql_stock + ')'
    
    
    # sql = "select date_format(work_date, '%Y%m%d') work_date, " + \
    #       "stock_symbol, high, close, low " + \
    #       "from stock_data " + \
    #       "where date_format(work_date, '%Y%m%d') between " + str(begin_date) + " and " + str(end_date)


    sql = ( "select date_format(work_date, '%Y%m%d') work_date, " 
    + " stock_symbol, high, close, low "
    + " from stock_data "
    + sql_cond
    )
    
        
    results = ar.db_query(sql)
    return results


def get_stock_name():
    
    sql = "select * from stock_name;"    
    results = ar.db_query(sql)
    return results



# Add lead and lag
# https://stackoverflow.com/questions/23664877/pandas-equivalent-of-oracle-lead-lag-function





# Application ----------------------
master()

app = dash.Dash()

colors = {
    'background': '#f5f5f5',
    'text': '#303030'
}

app.layout = html.Div(
    children=[
    # html.H1(
    #     children='Stock Forecast',
    #     style={
    #         'textAlign': 'center',
    #         'color': colors['text']
    #     }
    # ),
    # html.Div(
    #     children='Description',
    #     style={
    #         'textAlign': 'center',
    #         'color': colors['text']
    #     }
    # ),
    dcc.Dropdown(
        id='name_dropdown',
        options=stock_list,
        multi=True,
        value="MTL"
    ),
    
    daq.ToggleSwitch(
        id='btn_max',
        value=False
    ),
    
    html.Div(id='debug'),
    
    
    html.Div(id="line_chart"),
    # html.P('Data Source'),
    ],
    style={'backgroundColor': colors['background']}
)


@app.callback(
    
    [Output('line_chart', 'children'),
     Output('debug', 'children')
     ],
    [Input("name_dropdown", "value"),
     Input('btn_max', 'value')
     ]
    
)



# target_data = get_stock_data(stock='006205')


def update_output(dropdown_value, btn_max):

    global stock_list_pre
    selected_list = stock_list_pre[stock_list_pre['STOCK_SYMBOL']
                                    .isin(dropdown_value)] \
                    .reset_index(drop=True)
    

    if btn_max == True:
        
        # Optimize, keep existing data.
        global target_data
        target_data = get_stock_data(stock=dropdown_value)
        
        target_data['TYPE'] = 'HISTORICAL'
        target_data['WORK_DATE'] = target_data['WORK_DATE'].apply(ar.ymd)
        
        plot_data = target_data
        
    else:
        global main_data
        plot_data = main_data
        
        
    data1 = [{'x': plot_data[(plot_data['STOCK_SYMBOL'] == \
                              selected_list['STOCK_SYMBOL'][i]) & \
                      (plot_data['TYPE'] == 'HISTORICAL')]['WORK_DATE'],
              'y': plot_data[(plot_data['STOCK_SYMBOL'] == \
                              selected_list['STOCK_SYMBOL'][i]) & \
                      (plot_data['TYPE'] == 'HISTORICAL')]['CLOSE'],
              'type': 'line',
              'name': selected_list['STOCK'][i],
              } for i in range(0, len(selected_list))]
        

    # data5 = [{'x': df[df[1] == val][0],
    #               'y': df[df[1] == val][6],
    #               'type': 'line',
    #               'name': val + " - 交易量",
    #               } for val in dropdown_value]

    figure = [
        dcc.Graph(
            id='example-graph',
            figure={
                'data': data1,
                'layout': {
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                    },
#                    'title': 'DASH'
                }
            }
        ),
    ]


    # Debug
    debug_dropdown = '_'.join(dropdown_value)

    # return figure
    return figure, 'ID_'+debug_dropdown+'_MAX_'+str(btn_max)+'_'



if __name__ == '__main__':
    app.run_server()
    # app.run_server(debug=True)
