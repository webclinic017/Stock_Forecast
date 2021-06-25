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
from dash.dependencies import Input, Output
import pandas as pd
import sys

#import os
#import re
#import numpy as np
from flask import Flask, request


# In[ ]:


# Load Data ----------------

# Local DB .....
import mysql.connector

db = mysql.connector.connect(
 host="localhost",
 user="aron",
 password="57diyyLCHH4q1kwD",
 port="8889",
 database="twstock"
)

cursor = db.cursor()


# In[13]:


# PythonAnywhere .....
# import MySQLdb
#
# db=MySQLdb.connect(
#     host='aronhack.mysql.pythonanywhere-services.com',
#     user='aronhack',
#     passwd='pythonmysql2020',
#     db='aronhack$aronhack_dashboard',
#     charset = 'utf8')
#
# cursor = db.cursor()


# In[21]:


# Load Data --------------------------

cols = ['WORK_DATE', 'SECURITY_CODE',
        'HIGH_PRICE', 'PRICE', 'LOW_PRICE']

# Historical Data .....
cursor.execute("select work_date, security_code, "
               "high, close, low "
               "from stock_data;")

historical_data = pd.DataFrame(cursor.fetchall())
historical_data.columns = cols
historical_data['TYPE'] = 'PAST'
historical_data = historical_data.sort_values(by=['WORK_DATE', 'SECURITY_CODE'])

historical_data


# In[15]:


# Forecast Data .....
cursor = db.cursor()
cursor.execute("select work_date, security_code, "
               "high_price, close_price, low_price "
               "from stock_forecast;")

forecast_data = pd.DataFrame(cursor.fetchall())
forecast_data.columns = cols
forecast_data['TYPE'] = 'FORECAST'

forecast_data


# In[17]:


# Stock Name .....
cursor = db.cursor()
cursor.execute("select * "
               "from stock_name;")

stock_name = pd.DataFrame(cursor.fetchall())
stock_name.columns = ['SECURITY_CODE', 'NAME']

stock_name


# In[22]:


# Work Area -------------

df = pd.concat([historical_data, forecast_data])
df = df.sort_values(by=['SECURITY_CODE', 'WORK_DATE'])

df = pd.merge(df, stock_name, how='left', on=['SECURITY_CODE'])
df['SECURITY_CODE'] = df['SECURITY_CODE'] + ' ' + df['NAME']

# Add lead and lag
# https://stackoverflow.com/questions/23664877/pandas-equivalent-of-oracle-lead-lag-function


# Dash ----------------------

# Unique Name
# Update, Sort by name
stock_list_pre = df['SECURITY_CODE'].unique().tolist()
stock_list = []

# Optimize
for i in stock_list_pre:
    stock_list.append({'label': i, 'value': i})




# Application ----------------------

app = dash.Dash()

colors = {
    'background': '#f5f5f5',
    'text': '#303030'
}

app.layout = html.Div(children=[
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
    html.Div(id="line_chart"),
    # html.P('Data Source'),
    ],
    style={'backgroundColor': colors['background']}
)


@app.callback(
    Output("line_chart", "children"),
    [Input("name_dropdown", "value"),]
)



def update_output(dropdown_value):

    # Optimize
    data1 = [{'x': df[(df['SECURITY_CODE'] == val) & (df['TYPE'] == 'PAST')]['WORK_DATE'],
              'y': df[(df['SECURITY_CODE'] == val) & (df['TYPE'] == 'PAST')]['PRICE'],
              'type': 'line',
              'name': val + " - Close Price",
              } for val in dropdown_value]

    data2 = [{'x': df[(df['SECURITY_CODE'] == val) & (df['TYPE'] == 'FORECAST')]['WORK_DATE'],
              'y': df[(df['SECURITY_CODE'] == val) & (df['TYPE'] == 'FORECAST')]['HIGH_PRICE'],
              'type': 'line',
              'name': val + " - High Price",
              } for val in dropdown_value]

    # data3 = [{'x': df[(df['SECURITY_CODE'] == '0050') & (df['TYPE'] == 'PAST')]['WORK_DATE'],
    #           'y': df[(df['SECURITY_CODE'] == '0050') & (df['TYPE'] == 'PAST')]['LOW_PRICE'],
    #           'type': 'line',
    #             'name': val + " - High Price",
    #           } for val in dropdown_value]

    # data4 = [{'x': df[df[1] == val][0],
    #               'y': df[df[1] == val][5],
    #               'type': 'line',
    #               'name': val + " - 平均價",
    #               } for val in dropdown_value]
    #
    # data5 = [{'x': df[df[1] == val][0],
    #               'y': df[df[1] == val][6],
    #               'type': 'line',
    #               'name': val + " - 交易量",
    #               } for val in dropdown_value]

    figure = [
        dcc.Graph(
            id='example-graph',
            figure={
                'data': data1 + data2,
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

    return figure



if __name__ == '__main__':
   app.run_server()
