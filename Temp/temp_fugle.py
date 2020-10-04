# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

dash==1.9.1
dash-table==4.6.1
pandas==0.25.3
numpy==1.18.1
websocket-client==0.57.0
requests==2.22.0

import websocket


pip install websocket_client==0.57.0

import sys

# 將project_home指定為你的專案路徑
project_home = u'/Users/Aron/Documents/GitHub/Arsenal/'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# 可以讀取整個工具組，也可以讀取特定function
import fugle_realtime_websocket_api
from fugle_realtime import intraday


intraday.chart(apiToken="demo", output="dataframe", symbolId="2330")
intraday.quote(apiToken="demo", output="dataframe", symbolId="2884")

	
intraday.chart(apiToken="68481182df39d05fad4b8234bbd17a15", output="dataframe", symbolId="0050")
intraday.quote(apiToken="68481182df39d05fad4b8234bbd17a15", output="dataframe", symbolId="0050")
intraday.meta(apiToken="68481182df39d05fad4b8234bbd17a15", output="dataframe", symbolId="0050")
intraday.dealts(apiToken="68481182df39d05fad4b8234bbd17a15", output="dataframe", symbolId="0050")





