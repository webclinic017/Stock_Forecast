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


# Worklist
# (1) Convert get_stock_data
# (2) stock_get_list, upload to database

# 設定工作目錄 .....

# Local
path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'



# Server
# path = '/home/aronhack/stock_forecast/dashboard'
# path = '/home/aronhack/stock_analysis_us/dashboard'




# Codebase
path_codebase = ['/Users/Aron/Documents/GitHub/Arsenal',
                 '/Users/Aron/Documents/GitHub/Codebase_YZ',
                 path + '/Function']


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


local = False
local = True


stock_type = 'us'
stock_type = 'tw'


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
    stock_data = get_stock_data(begin_date, end_date, 
                                stock_type=stock_type)
    
    # Stock Name .....
    global stock_name
    stock_name = ar.stk_get_list(stock_type=stock_type)



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
                          + main_data['STOCK_NAME'])
    
    global target_data, target_symbol
    target_symbol = []
    target_data = pd.DataFrame()
    
    
    # Dash ----------------------
    
    # Stock List
    global stock_list, stock_list_pre
    stock_list_pre = main_data[['STOCK_SYMBOL', 'STOCK']] \
                        .drop_duplicates() \
                        .reset_index(drop=True)
                    
                    
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

def get_stock_data(begin_date=None, end_date=None, 
                   stock_type='tw', stock_symbol=None):
    
    
    if stock_type == 'tw':
        stock_tb = 'stock_data_tw'
    elif stock_type == 'us':
        stock_tb = 'stock_data_us'
    

    # Convert stock to list
    stock_li = [stock_symbol]
    stock_li = cbyz.list_flatten(stock_li)    
    
    
    if (begin_date != None) & (end_date != None):
        
        if stock_symbol != None:
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
    

    sql = ( "select date_format(work_date, '%Y%m%d') work_date, " 
    + " stock_symbol, high, close, low "
    + " from " + stock_tb + " "
    + sql_cond
    )
    
        
    results = ar.db_query(sql, local=local)
    return results


# def stock_get_name(stock_type='tw'):
    
#     if stock_type == 'tw':
#         sql = "select * from stock_name;"
#     elif stock_type == 'us':
#         sql = "select * from stock_name_us;"
        
#     results = ar.db_query(sql, local=local)
#     return results





# Application ----------------------
master()



app = dash.Dash()

colors = {
    'background': '#f5f5f5',
    'text': '#303030'
}


container_style = {
    'backgroundColor': colors['background'],
    'padding': '0 30px',
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


app.layout = html.Div([
        
    html.Div([
        dcc.Dropdown(
            id='name_dropdown',
            options=stock_list,
            multi=True,
            value=[]
        ),
    ],style=name_dropdown_style),
    
    html.Div([
        html.P('半年資料'),
        daq.ToggleSwitch(
            id='btn_max',
            value=False,
            style={'padding':'0 10px'}
        ),
        html.P('全部資料'),
    ],style=btn_max_style),        
    
    
    
    
    html.Div(id='debug',
             style = debug_style),
    
    
    html.Div(id="line_chart"),
    # html.P('Data Source'),
    ],
    
    style=container_style
)


@app.callback(
    
    [Output('line_chart', 'children'),
     Output('debug', 'children')
     ],
    [Input("name_dropdown", "value"),
     Input('btn_max', 'value')
     ]
    
)



def update_output(dropdown_value, btn_max):

    global stock_list_pre
    selected_list = stock_list_pre[stock_list_pre['STOCK_SYMBOL']
                                    .isin(dropdown_value)] \
                    .reset_index(drop=True)
    

    if btn_max == True:
        
        # Add loading icon
        
        global target_data, target_symbol
        list_inter = ar.list_intersect(dropdown_value, target_symbol)

        # Remove data
        if len(target_data)>0:
            target_data = target_data[target_data['STOCK_SYMBOL'] \
                                      .isin(list_inter['Intersect'])]
        
        # Add new data
        if len(list_inter['Diff_L']) > 0:
            new_data = get_stock_data(stock_type=stock_type,
                                      stock_symbol=list_inter['Diff_L'])
            new_data['TYPE'] = 'HISTORICAL'
            new_data['WORK_DATE'] = new_data['WORK_DATE'].apply(ar.ymd)
            target_data = target_data.append(new_data)
        
        target_symbol = dropdown_value
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
        

    historical_plot =  dcc.Graph(
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
                            },
                        )

    # data5 = [{'x': df[df[1] == val][0],
    #               'y': df[df[1] == val][6],
    #               'type': 'line',
    #               'name': val + " - 交易量",
    #               } for val in dropdown_value]
        

    figure = [historical_plot, historical_plot]


    # Debug
    debug_dropdown = '_'.join(dropdown_value)

    # return figure
    return figure, 'ID_'+debug_dropdown+'_MAX_'+str(btn_max)+'_'



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
    
    

dict1 = {'VALUE1':1, 'VALUE2':2}
dict2 = str(dict1)


test = ar.df_cross_join(stock_name, remove_same=True)

test = test[['STOCK_SYMBOL_x', 'STOCK_SYMBOL_y']]
test.columns = ['STOCK_Y', 'STOCK_VAR']
test['RULE_ID'] = 1
test['RESULTS'] = dict2
test['STOCK_Y'] = test['STOCK_Y'].astype(str)
test = test[['RULE_ID', 'STOCK_Y', 'STOCK_VAR', 'RESULTS']]


upload = test.iloc[0:10000, :]
upload.to_csv(path + '/stock_rule.csv', index=False)


ar.db_upload(upload, 'stock_rule_tw', True)

