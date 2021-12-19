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


host = 0


# Path .....
if host == 0:
    path = '/Users/aron/Documents/GitHub/Stock_Forecast/1_Data_Collection/1_Shareholding_Spread'
elif host == 2:
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



# %% Inner Function ......

def read_single_files():
    '''
    讀取手動下載的CSV

    Returns
    -------
    file_date : TYPE
        DESCRIPTION.

    '''
    
    # Get file list
    file_path = path_resource
    files = cbyz.os_get_dir_list(path=file_path, level=0, 
                                 extensions='csv',
                                 remove_temp=True)
    
    files = files['FILES']
    files = files[files['FILE_NAME'].str.contains('TDCC_OD_1-5')] \
                .reset_index(drop=True)
    
    file_date = []
    
    for i in range(len(files)):
        name = files.loc[i, 'FILE_NAME']
        file = pd.read_csv(file_path + '/' + name)
        file = file['資料日期'].unique().tolist()
        file_date = file_date + file

    # The master code process with str
    file_date = [str(i) for i in file_date]

    return file_date



# %%  Master ......
    

def master():
    
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from bs4 import BeautifulSoup
    from selenium.webdriver.support.ui import Select
    
    
    # https://www.tdcc.com.tw/smWeb/QryStockAjax.do
    link = 'https://www.tdcc.com.tw/portal/zh/smWeb/qryStock'
    
    webdriver_path = '/Users/aron/Documents/GitHub/Arsenal/geckodriver'
    driver = webdriver.Firefox(executable_path=webdriver_path)
    driver.get(link)
    
    
    # Date ......
    date_li_raw = []
    dropdown = driver.find_element_by_name("scaDate")
    options = [x for x in dropdown.find_elements_by_tag_name("option")]
    
    # driver.find_element_by_name("scaDate")
    # <ipython-input-13-cfefc7bef0f9>:1: DeprecationWarning: find_element_by_* commands are deprecated. Please use find_element() instead
    #   dropdown = driver.find_element_by_name("scaDate")    
    # dropdown = driver.find_element(tagname="scaDate")
    # options = [x for x in dropdown.find_elements("option")]
    
    
    for option in options:
        date_li_raw.append(option.get_attribute("value"))
        
    date_li_raw.sort(reverse=False) 
    
    
    # Remove finished date
    file_date = read_single_files()
    date_list = cbyz.li_remove_items(date_li_raw, file_date)
    
    
    # Stock Symbol ......
    symbols_raw = stk.twse_get_data()
    symbols = symbols_raw['SYMBOL'].tolist()
    
    
    # Merge ......
    symbols_tb = symbols_raw[['SYMBOL']]
    date_df = pd.DataFrame({'WORK_DATE':date_list})
    main_tb_pre = cbyz.df_cross_join(date_df, symbols_tb)
    
    
    # Bug，要考慮還沒有file的狀況
    load_success = False
    level = '持股/單位數分級'
    
    try:
        file = pd.read_csv(path_resource + '/stock_ratio.csv')
        file['SYMBOL'] = file['SYMBOL'].str.replace('ID_', '')
    except Exception as e:
        print(e)
    else:
        load_success = True
    
    
    # Exclude finished rows    
    if load_success and len(file) > 0:
        
        file = cbyz.df_conv_col_type(df=file, 
                                     cols=['SYMBOL', 'WORK_DATE'],
                                     to='str')
    
        main_tb = main_tb_pre \
                .merge(file, how='left', on=['SYMBOL', 'WORK_DATE'])
        
        main_tb = main_tb[main_tb[level].isna()].reset_index(drop=True)
        
        # Summary, for notice noly
        summary = main_tb.groupby(['WORK_DATE']) \
                    .size() \
                    .reset_index(name='COUNT')
    
    else:
        main_tb = main_tb_pre.copy()
    
    
    
    # NoSuchElementException: Unable to locate element: //select[@name='scaDate']/option[text()='20200710']
    # 這個程式的哲學是盡量減少sleep的時間，雖然這可能會導致抓不到某些element，但下一次再回洗它就可以了。

    
    # Query ......
    result = pd.DataFrame()
    
    for i in range(len(date_list)):
        
        # Date
        d = date_list[i]
        temp_tb = main_tb[main_tb['WORK_DATE']==d]
        
        if len(temp_tb) == 0:
            continue    
    
        for s in range(len(symbols)):
            symbol = symbols[s]
            
            # 已經存在現有檔案中
            chk = temp_tb[temp_tb['SYMBOL']==symbol]
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
    
            # Get Target HTML Element
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
            new_df['SYMBOL'] = 'ID_' + symbol
            new_df['WORK_DATE'] = d
            
            result = result.append(new_df)
            time.sleep(0.4)
    
            
            # Save File
            if len(result) > 0 and \
                (i == len(symbols) - 1 or s % 50 == 0):
                
                try:
                    file = pd.read_csv(path_resource + '/stock_ratio.csv')
                    file = file.append(result)
                except Exception as e:
                    file = result
                    print(e)
        
                file.to_csv(path_resource + '/stock_ratio.csv', 
                            index=False, encoding='utf-8-sig')
                
                result = pd.DataFrame()
                print('update_file')
        
        
        print(str(i) + '/' + str(len(date_list)))
    
    
    # Update, 把這寫成function
    # 避免最後有些資料還沒有儲存，卻因為continue結束迴圈......
    if len(result) > 0:
        file = pd.read_csv(path_resource + '/stock_ratio.csv')
        file = file.append(result)
    
        file.to_csv(path_resource + '/stock_ratio.csv', 
                    index=False, encoding='utf-8-sig')
        
        result = pd.DataFrame()        
        print('update_file')



# ........


def tdcc_upload_shareholdings_spread():
    
    
    cols = ['WORK_DATE', 'SYMBOL', 'LEVEL', 
                    'COUNT', 'VOLUME', 'RATIO']    
    
    # .......
    file0 = pd.read_csv(path_resource + '/stock_ratio.csv')
    file0.columns = ['LEVEL', 'LEVE_DESCR', 'COUNT', 'VOLUME', 
                     'RATIO', 'SYMBOL', 'WORK_DATE', ]
    
    file0 = file0[cols]
    file0['SYMBOL'] = file0['SYMBOL'].str.replace('ID_', '')
    
    
    # ......
    file1 = pd.read_csv(path_resource + '/集保戶股權分散表_0625.csv')
    file2 = pd.read_csv(path_resource + '/集保戶股權分散表_0702.csv')
    file3 = pd.read_csv(path_resource + '/集保戶股權分散表_0709.csv')    
    
    
    # .......
    file = file1.append(file2).append(file3)
    file.columns = cols
    
    file = file.append(file0)
    file = file[file['RATIO']<100]
    
    # 有些資料在「持股/單位數分級」會有一列是「差異數調整」，這種情況下的「人數」會是na
    file = file[~file['COUNT'].isna()]
    
    
    file = cbyz.df_conv_col_type(df=file, cols=['VOLUME'], to='int')
    
    
    # 
    file = file \
        .sort_values(by=['SYMBOL', 'WORK_DATE']) \
        .reset_index(drop=True)
        

    cbyz.df_chk_col_na(df=file)
    

    ar.db_upload(data=file, 
                 table_name='sharehold_spread',
                 local=local)    


    # chunk = 100000
    # times = int(len(file) / chunk) + 1
    # total = 0
    
    # for i in range(times):
        
    #     temp = file.loc[i * chunk : (i + 1) * chunk - 1]
    #     total = total + len(temp)
        
    #     ar.db_upload(data=temp, 
    #                  table_name='shareholdings_spread',
    #                  local=local)    
    
    #     print(i)


    
    return




if __name__ == '__main__':
    

    # Check, 記錄查無此資料的symbol >>  20201016 - 00894    
    master()



    

