#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:07:03 2021

@author: Aron
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc


host = 2
# host = 4
# host = 0
market = 'tw'


# Path .....
if host == 0:
    path = '/Users/Aron/Documents/GitHub/Stock_Forecast/1_Data_Collection/2_TEJ'
elif host == 1:
    path = '/home/aronhack/stock_forecast/Production/1_Data_Collection/2_TEJ'
elif host == 2:
    path = '/home/jupyter/Production/1_Data_Collection/2_TEJ'
# elif host == 3:
#     path = '/home/jupyter/Develop/1_Data_Collection/2_TEJ'    
elif host == 4:    
    # RT
    path = r'D:\Data_Mining\GitHub共用\Stock_Forecast\1_Data_Collection\2_TEJ'


# Codebase ......
path_codebase = [r'/Users/aron/Documents/GitHub/Arsenal/',
                 r'/home/aronhack/stock_predict/Function',
                 r'D:\Data_Mining\GitHub共用\Arsenal',
                 r'D:\Data_Mining\Projects\Codebase_YZ',                 
                 r'/home/jupyter/Arsenal/20220522',    
                 path + '/Function']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar
import arsenal_stock as stk

ar.host = host



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

import tejapi


# %% Inner Function ------

def chk_date(ewprcd2=True, ewtinst1c=True, 
             ewprcd=True, ewsale=True, ewifinq=True, ewnprcstd=True,
             ewgin=True, ewtinst1=True,delete=False, upload=True):
    
    tables = []
    
    if ewtinst1c:
        # 三大法人持股成本
        tables.append(['ewtinst1c', 'd']) 
        
    if ewprcd:
        # 證券交易資料表，一個月51034筆
        tables.append(['ewprcd', 'd'])  
        
    if ewprcd2:
        # 報酬率資訊表，兩個月約100000筆        
        tables.append(['ewprcd2', 'd']) 
        
    if ewsale:
        # 月營收資料表，一個月約2000筆，等同於股票檔數
        tables.append(['ewsale', 'd']) 
        
    if ewifinq:
        # 單季財務資料表，資料量等同於股票檔數
        tables.append(['ewifinq', 'd'])         

    if ewnprcstd:
        # 證券屬性表，3125筆，資料量等同於股票檔數
        tables.append(['ewnprcstd', None]) 
        
    if ewgin:
        # 融資券資料
        # - 20220301的資料為1869筆，但每天的數量可能會不一樣
        tables.append(['ewgin', 'd', None]) 

    if ewtinst1:
        # 三大法人資料
        # - 20220301的資料為2307筆
        tables.append(['ewtinst1', 'd', None])         


    # Query ......    
    # 1. 如果沒有設!='00000000'的話，min_date都會是00000000
    li = []    
    for i in range(len(tables)):
        
        table = tables[i][0]
        sql = (" select min(date_format(mdate, '%Y%m%d')) min_date, "
               " max(date_format(mdate, '%Y%m%d')) max_date "
               " from " + table + " "
               " where date_format(mdate, '%Y%m%d') != '00000000' "
              )
        
        try:
            query = ar.db_query(sql)
            min_date = query.loc[0, 'MIN_DATE']
            max_date = query.loc[0, 'MAX_DATE']
            error = ''
            
        except Exception as e:
            min_date = np.nan
            max_date = np.nan
            error = e
    
        li.append([table, min_date, max_date, error])
    
    result = pd.DataFrame(li, 
                          columns=['TABEL', 'MIN_DATE', 
                                   'MAX_DATE', 'ERROR'])
    print(result)
    return result
    


def tej_api_info():
    
    global _key_id
    
    # 系統限制單次取得最大筆數為10,000筆，可使用 paginate=True 參數分次取得資料，
    # 但總筆數單次最多為1,000,000筆。請斟酌使用篩選條件降低筆數。
    ar.get_black_box()
    key_df = pd.read_csv(ar.path_black_box + '/tej_key.csv')
    tejapi.ApiConfig.api_key = key_df.loc[_key_id, 'key']
    
    tej_info = tejapi.ApiConfig.info()
    return tej_info['todayRows'], tej_info['rowsDayLimit']



# %% Master ------



def master_single(info, begin=None, end=None, delete=False, 
                     upload=True, export=False):
    '''
    以月或季為單位的資料，篩選的時候還是用日期下條件，所以當成是d
    '''
    

    table = info['table']    
    time_unit = info['time_unit']    
    update_by_day = info['update_by_day']
    export_format = info['export_format']
    backtrack = info['backtrack']
    
    
    # Create Folder
    folder = path_export + '/' + table
    if not os.path.exists(folder):
        os.mkdir(folder)    

    
    if begin == None and end == None:
        end = cbyz.date_get_today()
        begin = cbyz.date_cal(obj=end, amount=-backtrack, unit='d')
    
    
    # Time Setting
    year_str = str(end)[0:4]
    year_head = int(year_str + '0101')
    year_end = int(year_str + '1231')    
    

    msg = 'begin與end不可跨年度，避免儲存以月為單位的檔案時發生錯誤'
    assert str(begin)[0:4] == str(end)[0:4], msg


    assert begin <= end, 'begin date is larger than end date'
    print('tej update - ' + str(begin) + ' - ' + str(end)) 
    
    
    # Calendar ......
    if update_by_day:
        calendar = cbyz.date_get_calendar(begin_date=begin, end_date=end,
                                          simplify=True)
    
        calendar = cbyz.df_col_lower(calendar)
        calendar = calendar[['work_date']]
        calendar.columns = ['begin']
        calendar['end'] = calendar['begin']
    else:
        calendar = pd.DataFrame({'begin':[begin], 'end':[end]})
    
    
    
    for i in range(len(calendar)):
  
        cur_begin = calendar.loc[i, 'begin']
        cur_end = calendar.loc[i, 'end']
        
        # Set Date ......      
        if export_format == None:
            begin_str = cbyz.ymd(cur_begin)
            end_str = cbyz.ymd(cur_end)
            
        elif export_format == 'y':
            begin_str = cbyz.ymd(year_head)
            end_str = cbyz.ymd(year_end)
            
        begin_str = begin_str.strftime('%Y-%m-%d')
        end_str = end_str.strftime('%Y-%m-%d')    
                    
        
        # Delete Data ...
        if delete and host in [0, 2]:
            
            # Delete incomplete data.
            sql = (" delete from " + table + " "
                   " where date_format(mdate, '%Y%m%d') >= " + str(cur_begin) + ""
                   " and date_format(mdate, '%Y%m%d') <= " + str(cur_end) + "")
            
            try:
                ar.db_execute(sql, commit=True)    
            except Exception as e:
                print('Delete Error')
                print(e)
        
        
        # Query ......        
        if time_unit == 'd':
            data = tejapi.get('TWN/' + table.upper(),
                              mdate={'gte':begin_str, 'lte':end_str},
                              paginate=True)
            
            file_path = '/' + table +'_' + begin_str + '_' + end_str + '.csv'
            
        elif time_unit == None:     
            data = tejapi.get('TWN/' + table.upper(), paginate=True)
            file_path = '/' + table + '.csv'            


        if len(data) == 0:
            print('tej zero rows - ', table, cur_begin, cur_end)
            continue

        # Prevent data incomplete
        used_rows, row_limit = tej_api_info()
        print('TEJ API -', used_rows, '/', row_limit)
        
        if used_rows > row_limit:
            break
    
        if 'mdate' in data.columns:     
            # ProgrammingError: Failed processing format-parameters; 
            # Python 'timestamp' cannot be converted to a MySQL type
            data['mdate'] = data['mdate'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        
        # Upload And Export ......
        bacup_export = False
        try:
            if upload:
                ar.db_upload(data=data, table_name=table)
        except Exception as e:
            print(table, ' failed to upload')
            print(e)
            bacup_export = True
            
        if export or bacup_export:
            data.to_csv(folder + file_path, index=False)        


        # Reset Object
        data = []
        del begin_str, end_str
        print('tej update - ', table, cur_begin, cur_end)



def master(begin=None, end=None, ewprcd=True, amtop1=False,
           ewsale=True, ewifinq=False, ewnprcstd=False, ewtinst1c=False,
           ewgin=False, ewtinst1=False, delete=False, upload=True, 
           export=False, key_id=0):
    '''
    以月或季為單位的資料，篩選的時候還是用日期下條件，所以當成是d
    '''
    
    
    # v1.0200 - 20220317
    # - Add ewgin and ewtinst1    
    # - Add export param
    
    # v1.0201 - 20220317
    # - Rename update as master >> Done
    # - After completing query then deleting old data to prevent reach 
    #   limit and interupt
    
    # v1.0301
    # - Update day by day for amtop1
    # - Add query limit 
    
    
    # v1.0400
    # - Remove plaintext key
    # - Add key_id parameter for tej_api_info
    
    
    assert len(str(begin)) == 8 or begin == None, 'begin date error'
    assert len(str(end)) == 8 or end == None, 'end date error'
    
    
    global _key_id
    _key_id = key_id
    tej_api_info()
    
    # tables list欄位定義
    # 1：資料標名稱
    # 2：更新頻率
    # 3：每個saved file的時間區間，只對在在資料夾的檔案有效。當此數值為y時，資料會
    #    從1/1重抓，並覆寫原始檔案。
    tables = []
    cols = ['table', 'time_unit', 'update_by_day', 
            'backtrack', 'export_format']

    if ewprcd:
        # 證券交易資料表，一個月51034筆；存在DB
        tables.append(['ewprcd', 'd', False, 14, None]) 
    
    if amtop1:
        # 主要券商進出明細-股票別，兩天19萬筆
        tables.append(['amtop1', 'd', True, 2, None])     

    if ewsale:
        # 月營收資料表，一個月約2000筆，等同於股票檔數；存在資料夾
        tables.append(['ewsale', 'd', False, 14, 'y'])     
        
    if ewifinq:
        # 單季財務資料表，資料量等同於股票檔數；存在資料夾
        tables.append(['ewifinq', 'd', False, 14, 'y'])        

    if ewnprcstd:
        # 證券屬性表，3125筆，資料量等同於股票檔數
        tables.append(['ewnprcstd', False, 14, None]) 

    if ewnprcstd:
        # 證券屬性表，3125筆，資料量等同於股票檔數
        tables.append(['ewnprcstd', False, 14, None]) 
    
    # if ewtinst1c:
    #     # 三大法人持股成本；存在DB
    #     tables.append(['ewtinst1c', 'd', None]) 
        
    # if ewprcd2:
    #     # 這個資料集不好用
    #     # 報酬率資訊表，兩個月約100000筆        
    #     tables.append(['ewprcd2', 'd']) 
        
    # if ewgin:
    #     # 融資券資料
    #     # - 20220301的資料為1869筆，但每天的數量可能會不一樣
    #     tables.append(['ewgin', 'd', None]) 

    # if ewtinst1:
    #     # 三大法人資料
    #     # - 20220301的資料為2307筆
    #     tables.append(['ewtinst1', 'd', None]) 

    tables = pd.DataFrame(tables, columns=cols)

    
    
    # Query ......
    # 系統限制單次取得最大筆數為10,000筆，可使用 paginate=True 參數分次取得資料，
    # 但總筆數單次最多為1,000,000筆。請斟酌使用篩選條件降低筆數。
    
    for i in range(len(tables)):
        
        info = tables.loc[i, :]

        master_single(info=info, begin=begin, end=end, delete=delete, 
                         upload=upload, export=export)            



# def master_20220519(begin=None, end=None, ewprcd=True, amtop1=False,
#             ewsale=True, ewifinq=False, ewnprcstd=False, ewtinst1c=True,
#             ewgin=True, ewtinst1=True, delete=False, upload=True, export=False):
#     '''
#     以月或季為單位的資料，篩選的時候還是用日期下條件，所以當成是d
#     '''
    
    
#     # v1.0200 - 20220317
#     # - Add ewgin and ewtinst1    
#     # - Add export param
    
#     # v1.0201 - 20220317
#     # - Rename update as master >> Done
#     # - After completing query then deleting old data to prevent reach 
#     #   limit and interupt
    
#     # v1.0301
#     # - Update day by day for amtop1
    
    
#     assert len(str(begin)) == 8 or begin == None, 'begin date error'
#     assert len(str(end)) == 8 or end == None, 'end date error'
    
#     msg = 'begin與end不可跨年度，避免儲存以月為單位的檔案時發生錯誤'
#     assert str(begin)[0:4] == str(end)[0:4], msg


#     if begin == None and end == None:
#         end = cbyz.date_get_today()
#         begin = cbyz.date_cal(end, -14, 'd')

#     assert begin <= end, 'begin date is larger than end date'
#     print('tej update - ' + str(begin) + ' - ' + str(end)) 
    
    
#     # Time Setting
#     year_str = str(end)[0:4]
#     year_head = int(year_str + '0101')
#     year_end = int(year_str + '1231')
    
    
#     # tables list欄位定義
#     # 1：資料標名稱
#     # 2：更新頻率
#     # 3：每個saved file的時間區間，只對在在資料夾的檔案有效。當此數值為y時，資料會
#     # 從1/1重抓，並覆寫原始檔案。
#     tables = []
    

#     if ewprcd:
#         # 證券交易資料表，一個月51034筆；存在DB
#         tables.append(['ewprcd', 'd', None]) 
    
#     if amtop1:
#         # 主要券商進出明細-股票別，兩天19萬筆
#         tables.append(['amtop1', 'd', None])     

#     if ewsale:
#         # 月營收資料表，一個月約2000筆，等同於股票檔數；存在資料夾
#         tables.append(['ewsale', 'd', 'y'])     
        
#     if ewifinq:
#         # 單季財務資料表，資料量等同於股票檔數；存在資料夾
#         tables.append(['ewifinq', 'd', 'y'])        

#     if ewnprcstd:
#         # 證券屬性表，3125筆，資料量等同於股票檔數
#         tables.append(['ewnprcstd', None]) 

#     if ewnprcstd:
#         # 證券屬性表，3125筆，資料量等同於股票檔數
#         tables.append(['ewnprcstd', None]) 
    
#     # if ewtinst1c:
#     #     # 三大法人持股成本；存在DB
#     #     tables.append(['ewtinst1c', 'd', None]) 
        
#     # if ewprcd2:
#     #     # 這個資料集不好用
#     #     # 報酬率資訊表，兩個月約100000筆        
#     #     tables.append(['ewprcd2', 'd']) 
        
#     # if ewgin:
#     #     # 融資券資料
#     #     # - 20220301的資料為1869筆，但每天的數量可能會不一樣
#     #     tables.append(['ewgin', 'd', None]) 

#     # if ewtinst1:
#     #     # 三大法人資料
#     #     # - 20220301的資料為2307筆
#     #     tables.append(['ewtinst1', 'd', None]) 

    
#     # Delete Data ......
#     if delete:
        
#         for i in range(len(tables)):
            
#             if host in [3, 4]:
#                 break
            
#             table = tables[i][0]
            
#             # Delete incomplete data.
#             sql = (" delete from " + table + " "
#                     " where date_format(mdate, '%Y%m%d') >= " + str(begin) + ""
#                     " and date_format(mdate, '%Y%m%d') <= " + str(end) + "")
            
#             try:
#                 ar.db_execute(sql, commit=True)    
#             except Exception as e:
#                 print('Delete Error')
#                 print(e)
            
    
#     # Query ......
#     # 系統限制單次取得最大筆數為10,000筆，可使用 paginate=True 參數分次取得資料，
#     # 但總筆數單次最多為1,000,000筆。請斟酌使用篩選條件降低筆數。
#     global data
    
#     for i in range(len(tables)):
        
#         table = tables[i][0]
#         time_unit = tables[i][1]
#         saved_file_period = tables[i][2]

#         # Create Folder
#         folder = path_export + '/' + table
#         if not os.path.exists(folder):
#             os.mkdir(folder)


#         # Set Date ......      
#         if saved_file_period == None:
#             begin_str = cbyz.ymd(begin)
#             end_str = cbyz.ymd(end)
            
#         elif saved_file_period == 'y':
#             begin_str = cbyz.ymd(year_head)
#             end_str = cbyz.ymd(year_end)
            
#         begin_str = begin_str.strftime('%Y-%m-%d')
#         end_str = end_str.strftime('%Y-%m-%d')    
            
        
#         if time_unit == 'd':
#             data = tejapi.get('TWN/' + table.upper(),
#                               mdate={'gte':begin_str, 'lte':end_str},
#                               paginate=True)
            
#             file_path = '/' + table +'_' + begin_str + '_' + end_str + '.csv'
            
#         elif time_unit == None:     
#             data = tejapi.get('TWN/' + table.upper(), paginate=True)
#             file_path = '/' + table + '.csv'            


#         if len(data) == 0:
#             continue

        
#         if 'mdate' in data.columns:     
#             # ProgrammingError: Failed processing format-parameters; 
#             # Python 'timestamp' cannot be converted to a MySQL type
#             data['mdate'] = data['mdate'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        
#         # Upload And Export ......
#         bacup_export = False
#         try:
#             if upload:
#                 ar.db_upload(data=data, table_name=table)
#         except Exception as e:
#             print(table, ' failed to upload')
#             print(e)
#             bacup_export = True
            
#         if export or bacup_export:
#             data.to_csv(folder + file_path, index=False)        


#         # Reset Object
#         data = []
#         del begin_str, end_str


# .............


def upload_saved_files(amtop1=False, ewprcd2=True, ewtinst1c=True, ewprcd=True, ewsale=True, 
           ewifinq=True, ewnprcstd=True,
           delete=False, upload=True):
    
    '''
    Dev, 先拼拼貼貼，還沒寫完
    
    '''
    
    tables = []


    if amtop1:
        # 主要券商進出明細-股票別，兩天19萬筆
        tables.append(['amtop1', 'd'])     

    if ewtinst1c:
        # 三大法人持股成本
        tables.append(['ewtinst1c', 'd']) 
    elif ewprcd:
        # 證券交易資料表，一個月51034筆
        tables.append(['ewprcd', 'd'])  
        
    elif ewprcd2:
        # 報酬率資訊表，兩個月約100000筆        
        tables.append(['ewprcd2', 'd']) 
        
    elif ewsale:
        # 月營收資料表，一個月約2000筆，等同於股票檔數
        tables.append(['ewsale', 'd']) 
        
    elif ewifinq:
        # 單季財務資料表，資料量等同於股票檔數
        tables.append(['ewifinq', 'd'])         

    elif ewnprcstd:
        # 證券屬性表，3125筆，資料量等同於股票檔數
        tables.append(['ewnprcstd', None]) 
    

        
    for i in range(len(tables)):
        
        table = tables[i][0]
        
        # Get file list
        file_path = path_export + '/' + table
        files = cbyz.os_get_dir_list(path=file_path, level=0, 
                                     extensions='csv',
                                     remove_temp=True)

        files = files['FILES']
    
        for j in range(len(files)):
            
            name = files.loc[j, 'FILE_NAME']
            file = pd.read_csv(file_path + '/' + name)
    
            table_info = ar.db_get_table_info(table)
            table_info = table_info[table_info['DATA_TYPE']=='datetime'] \
                        .reset_index(drop=True)
            date_cols = table_info['COLUMN_NAME'].tolist()

            for k in range(len(date_cols)):
                date_col = date_cols[k]
                file[date_col] = pd.to_datetime(file[date_col])
                file[date_col] = file[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')                

            ar.db_upload(data=file, table_name=table)



def create_table():
    
    file_path = r'/Users/Aron/Documents/GitHub/Data/Stock_Forecast/1_Data_Collection/2_TEJ/Export/ewsale/ewsale_2021-07-01_2021-09-30.csv'
    
    file = pd.read_csv(file_path)
    file = file.loc[0:100]
    file.to_csv(path_export + '/ewsale.csv', index=False)
    



def automation():
    
    # ewtinst1c
    # - 三大法人持股成本
    # - 2016-2020已完成
    # ewprcd
    # - 證券交易資料表，一個月51034筆
    # - 2016-2020已完成
    # ewifinq
    # - 單季財務資料表，資料量等同於股票檔數
    # - 2016-2020已完成
    # ewnprcstd
    # - 證券屬性表，3125筆，資料量等同於股票檔數
    # ewsale
    # - 月營收資料表，一個月約2000筆，等同於股票檔數
    # - 2016-2020已完成    
    
    # ewprcd2
    # - 報酬率資訊表，兩個月約100000筆
    # - 未完成


    info = tej_api_info()
    print('TEJ API - ', info[0], '/', info[1])

    
    # 目前的模式是全部一起刪除後再一起update，可能會刪除完後，在update的時候出錯
    # Table 'twstock.ewprcd2' doesn't exist    
    # Table 'twstock.ewifinq' doesn't exist
    # Table 'twstock.ewnprcstd' doesn't exist   
    
    # Check
    chk = chk_date()
    
    # master(begin=None, end=None, ewprcd=True, ewtinst1c=False, 
    #         ewsale=True, ewifinq=True, ewnprcstd=False,
    #         ewgin=True, ewtinst1=True, delete=True, upload=True)
    

    master(begin=None, end=None, ewprcd=True, amtop1=False,
           ewsale=True, ewifinq=True, ewnprcstd=False, ewtinst1c=False,
           ewgin=False, ewtinst1=False, delete=True, upload=True, 
           export=True, key_id=0)

    # ewsale有bug
    # Failed processing format-parameters; Python 'timestamp' cannot be converted to a MySQL type
    # 1. ewsale是不是手動上傳的？
    # 2. TEJ只開放五年的資料
        
    # master(begin=20211216, end=20211231, ewprcd=False, ewtinst1c=False, 
    #         ewsale=True, ewifinq=False, ewnprcstd=False,
    #         delete=True, upload=True)   

    # master(begin=20210101, end=20211231, ewprcd=False, ewtinst1c=False, 
    #         ewsale=False, ewifinq=True, ewnprcstd=False,
    #         delete=True, upload=True)        

    chk2 = chk_date()
    
    info = tej_api_info()
    print('TEJ API - ', info[0], '/', info[1])
    



def check():
    '''
    Check historical data impocomlete
    '''
        
    file = pd.read_csv(path_export + '/ewiprcd/ewiprcd_data_2019-01-01_2020-12-31.csv')
    file2 = pd.read_csv(path_export + '/ewiprcd/ewiprcd_data_2021-01-01_2021-07-13.csv')
    
    final = file.append(file2)
    final.to_csv(path_export + '/ewiprcd/ewiprcd_data_2019-01-01_2021-07-13.csv', index=False)
    
    
        
# %% Check ------    
    

    
    
# %% Manually Operate

def operate_sql():
    
    # Create Table
    # sql = ("CREATE TABLE ewtinst1 (coid char(7), mdate datetime, qfii_ex1 decimal(9,0), fund_ex decimal(9,0), dlr_ex decimal(9,0), ttl_ex decimal(9,0), qfii_buy decimal(9,0), qfii_sell decimal(9,0), fd_buy decimal(9,0), fd_sell decimal(9,0), dlr_buy decimal(9,0), dlr_sell decimal(9,0), ttl_stk decimal(9,0), fld008 decimal(9,0), fld011 decimal(9,0), t_pct decimal(9,2), fld022 decimal(9,2), fld023 decimal(9,2))")
    # ar.db_execute(sql)
    
    
    # sql = " truncate table ewtinst1c "
    # ar.db_execute(sql)
    
    # sql = " select count(*) from ewtinst1c "
    # ar.db_query(sql)
    
    
    # file_path = '/Users/aron/Documents/GitHub/Stock_Forecast/0_Finance_Controller/Resource/ewprcd_restore.csv'
    # file = pd.read_csv(file_path)
    # ar.db_upload(data=file, table_name='ewprcd')
    pass


def manually_upload():
    
    
    chk = chk_date()
    
    # print('ewtinst1 中有na')
    master(begin=20170101, end=20171231, ewprcd=False, ewtinst1c=False, 
            ewsale=True, ewifinq=False, ewnprcstd=False, 
            ewgin=False, ewtinst1=False, 
            delete=False, upload=True, export=True, key_id=0)
    
    
    # print('ewtinst1 中有na')
    master(begin=20220101, end=20220520, ewprcd=False, ewtinst1c=False, 
            ewsale=False, ewifinq=True, ewnprcstd=False, 
            ewgin=False, ewtinst1=False, 
            delete=False, upload=True, export=True, key_id=0)
    
    
    # Update amtop1
    master(begin=20190108, end=20190115, ewprcd=False, amtop1=True,
           ewsale=False, ewifinq=False, ewnprcstd=False, ewtinst1c=False,
           ewgin=False, ewtinst1=False, delete=True, upload=True, 
           export=True, key_id=0)
    

    chk2 = chk_date()

    # data['coid'] = 'ID_' + data['coid']
    # data.to_csv(path_export + '/coid_fix.csv', index=False)
    
    

# %% Dev ------

def dev():
    
    # print('ewtinst1 中有na')
    master(begin=20220101, end=20220520, ewprcd=False, ewtinst1c=False, 
            ewsale=False, ewifinq=True, ewnprcstd=False, 
            ewgin=False, ewtinst1=False, 
            delete=False, upload=True, export=True, key_id=0)
    


# %% Execution ......

if __name__ == '__main__':
    automation()
