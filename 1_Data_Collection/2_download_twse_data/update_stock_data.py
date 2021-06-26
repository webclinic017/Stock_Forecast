#!/usr/bin/env python
# coding: utf-8

import os
import re
import requests
import numpy as np
import pandas as pd
from datetime import date
# from flask import Flask, request
import MySQLdb



# Worklist
# 如果價錢和前一天完全一樣就不更新


db = MySQLdb.connect(
    host = 'aronhack.mysql.pythonanywhere-services.com',
    user = 'aronhack',
    passwd = 'pythonmysql2020',
    db = 'aronhack$aronhack_dashboard',
    charset = 'utf8')

cursor = db.cursor()





def master():
    
    # 檢查兩筆，如果數字都一樣的話就不更新
    chk_exists
    
    
    stk.twse_get_data(upload=False)


if __name__ == '__main__':
    
    master()