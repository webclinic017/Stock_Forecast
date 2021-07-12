#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:05:39 2021

@author: Aron
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Forecast/1_Data_Collection/1_Stock_Ratio'
else:
    path = '/home/aronhack/stock_forecast/dashboard/'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ/',
                 path + 'Function/']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk



data = stk.get_data(data_begin=20210630, data_end=20210703, stock_symbol=['0050'])


# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 

# 新增工作資料夾
global path_resource, path_function, path_temp, path_export
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

    chk = file[(file['STOCK_SYMBOL']=='2347') & (file['WORK_DATE']==20210611)]
    chk = file[file['WORK_DATE']==20200710].drop_duplicates(subset=['STOCK_SYMBOL'])
    # chk = file[file[level]=='Null']    
    
    return ''




# 集保比例 ......
    
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import Select



# https://www.tdcc.com.tw/smWeb/QryStockAjax.do
link = 'https://www.tdcc.com.tw/portal/zh/smWeb/qryStock'
webdriver_path = '/Users/Aron/Documents/GitHub/Arsenal/geckodriver'
driver = webdriver.Firefox(executable_path=webdriver_path)
driver.get(link)


# Date ......
date_li = []
dropdown = driver.find_element_by_name("scaDate")
options = [x for x in dropdown.find_elements_by_tag_name("option")]

for option in options:
    date_li.append(option.get_attribute("value"))
    

date_li.sort(reverse=True)     
# date_li.sort(reverse=False) 
date_list = cbyz.li_remove_items(date_li, ['20210702', '20210625', '20210709'])



# Stock Symbol ......
symbols_raw = stk.twse_get_data()
symbols = symbols_raw['STOCK_SYMBOL'].tolist()


# Merge ......
symbols_tb = symbols_raw[['STOCK_SYMBOL']]
date_df = pd.DataFrame({'WORK_DATE':date_list})
main_tb_pre = cbyz.df_cross_join(date_df, symbols_tb)


# Bug，要考慮還沒有file的狀況
load_fail = False
level = '持股/單位數分級'

try:
    file = pd.read_csv(path_export + '/stock_ratio.csv')
    file['STOCK_SYMBOL'] = file['STOCK_SYMBOL'].str.replace('ID_', '')
    
except:
    load_fail = True

    
# file = file \
#         .drop_duplicates(subset=['STOCK_SYMBOL', 'WORK_DATE', '持股/單位數分級']) \
#         .reset_index(drop=True)







    
if load_fail == False and len(file) > 0:
    
    file = cbyz.df_conv_col_type(df=file, 
                                 cols=['STOCK_SYMBOL', 'WORK_DATE'],
                                 to='str')

    main_tb = main_tb_pre.merge(file, how='left', 
                                on=['STOCK_SYMBOL', 'WORK_DATE'])
    main_tb = main_tb[main_tb[level].isna()]
    
    summary = main_tb.groupby(['WORK_DATE']).size().reset_index(name='COUNT')

else:
    main_tb = main_tb_pre.copy()




# NoSuchElementException: Unable to locate element: //select[@name='scaDate']/option[text()='20200710']

# 這個程式的哲學是盡量減少sleep的時間，雖然這可能會導致抓不到某些element，但下一次再回洗它就可以了。




# %% Query ......
result = pd.DataFrame()
# date_list = date_list[:1]


for i in range(len(date_list)):
    
    # Date
    d = date_list[i]
    
    temp_tb = main_tb[main_tb['WORK_DATE']==d]
    if len(temp_tb) == 0:
        continue    

    for s in range(len(symbols)):
        symbol = symbols[s]
        
        # 已經存在現有檔案中
        chk = temp_tb[temp_tb['STOCK_SYMBOL']==symbol]
        if len(chk) == 0:
            continue
        
        # 1. 抓一小段資料後，網址可能會自動跳轉到首頁，首頁的載入速度又很慢
        #    如果sleep的時間太短，sleep結束後，
        #    首頁可能還沒載入完畢，程式不會出錯，但會一直卡在首頁
        # 3. sleepe為0.8時可能會出現以下錯誤
        # StaleElementReferenceException: The element reference of <input id="StockNo" name="stockNo" type="text"> is stale; either the element is no longer attached to the DOM, it is not in the current frame context, or the document has been refreshed        
        if driver.current_url != link:
            driver.get(link)
            time.sleep(1)
        
        # 當發生自動跳轉問題時，需要重新設定時間區間，所以把這段寫在內層的for loop，而不是
        # 寫在外面
        try:
            driver.find_element_by_xpath("//select[@name='scaDate']/option[text()='" \
                     + d + "']").click()
        except:
            continue

        
        # NoSuchElementException: Unable to locate element: [id="StockNo"]
        try:
            time.sleep(0.8)
            symbol_input = driver.find_element_by_id("StockNo")
            symbol_input.clear()
            symbol_input.send_keys(symbol)              
        except:
            continue
            
        
        # 抓一小段資料後，網址可能會自動跳轉
        if driver.current_url != link:
            continue


        # Submit .....
        try:
            #　sleep的時間為0.3時，可能會太快，導致抓不到submit的按鈕
            submit = driver.find_element_by_css_selector("input[type='submit']")
            submit.submit()
        except:
            continue


        try:
            time.sleep(1)
            # time.sleep(0.5)
            role_main = driver.find_element_by_css_selector("[role='main']")
            html = role_main.get_attribute('innerHTML')
            soup = BeautifulSoup(html, 'html.parser')       
        except:
            continue
        
        
        # Parse Table ......            
        table = soup.findAll('table', {"class": 'table'})

        if len(table) == 0:
            continue

        table = str(table[0])
        new_df = pd.read_html(table)[0]
        
        # 刪除table，避免抓到前一檔的資料        
        del table        
        
        if new_df.iloc[0, 0] == '查無此資料':
            new_df = new_df[new_df.index > 0]
            new_df.loc[0] = [np.nan for i in range(len(new_df.columns))]
            
            # 用NA的話，存成csv的時候會變成空格，在前面用isna()的時候會誤判
            new_df.loc[0, '持股/單位數分級'] = 'Null'

        
        # 存成csv的時候，股票編號可能會被當成數字，像是0050會變成50，因此在前面加上ID_
        new_df['STOCK_SYMBOL'] = 'ID_' + symbol
        new_df['WORK_DATE'] = d
        
        result = result.append(new_df)
        time.sleep(0.4)


        if len(result) > 0 and \
            (i == len(symbols) - 1 or s % 50 == 0):
            
            try:
                file = pd.read_csv(path_export + '/stock_ratio.csv')
                file = file.append(result)
            except:
                file = result
    
            file.to_csv(path_export + '/stock_ratio.csv', 
                          index=False, encoding='utf-8-sig')
            
            result = pd.DataFrame()
            print('update_file')
    
    
    # # 避免最後一輪迴圈的時候，可能有剩餘的資料沒有記錄到 ......
    # if len(result) > 0:
    #     file = pd.read_csv(path_export + '/stock_ratio.csv')
    #     file = file.append(result)
    
    #     file.to_csv(path_export + '/stock_ratio.csv', 
    #                   index=False, encoding='utf-8-sig')
        
    #     result = pd.DataFrame()        
    #     print('update_file')
            
    print(str(i) + '/' + str(len(date_list)))


# Update, 把這寫成function
# 避免最後有些資料還沒有儲存，卻因為continue結束迴圈......
if len(result) > 0:
    file = pd.read_csv(path_export + '/stock_ratio.csv')
    file = file.append(result)

    file.to_csv(path_export + '/stock_ratio.csv', 
                  index=False, encoding='utf-8-sig')
    
    result = pd.DataFrame()        
    print('update_file')





def tdcc_shareholdings_spread():
    
    
    file = pd.read_csv(path_export + '/stock_ratio.csv')
    
    file = file.rename(columns={'序':'INDEX',
                                '持股/單位數分級':'LEVEL',
                                '人數':'COUNT',
                                '股數/單位數':'UNIT_RATIO',
                                '占集保庫存數比例 (%)':'RATIO'})
    
    file = file[~file['LEVEL'].str.contains('合')]
    
    return











