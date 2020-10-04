# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:09:52 2020

@author: Aron
"""

import pandas as pd
import mysql.connector

db = mysql.connector.connect(
 host="localhost",
 user="aron",
 password="57diyyLCHH4q1kwD",
 port="8889",
 database="twstock"
)


# PythonAnywhere .....
# import MySQLdb
#
# db=MySQLdb.connect(
#     host='aronhack.mysql.pythonanywhere-services.com',
#     user='aronhack',
#     passwd='pythonmysql2020',
#     db='aronhack$aronhack_dashboard',
#     charset = 'utf8')


# Load Data ..........

cols = ['WORK_DATE', 'SECURITY_CODE', 
        'HIGH_PRICE', 'PRICE', 'LOW_PRICE']

cursor = db.cursor()
cursor.execute("select work_date, security_code, "
               "high_price, close_price, low_price "
               "from stock_data;")

historical_data = pd.DataFrame(cursor.fetchall())
historical_data.columns = cols
historical_data['TYPE'] = 'PAST'



cursor = db.cursor()
cursor.execute("select work_date, security_code, "
               "high_price, close_price, low_price "
               "from stock_forecast;")

forecast_data = pd.DataFrame(cursor.fetchall())
forecast_data.columns = cols
forecast_data['TYPE'] = 'FORECAST'



# Main ..............

main = pd.concat([historical_data, forecast_data])



