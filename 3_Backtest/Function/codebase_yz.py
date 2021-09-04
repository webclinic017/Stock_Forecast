# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:54:07 2020

@author: 0112542
"""

# Worklist
# 1.ymd,將2020/10/1的str也轉成日期


import pandas as pd
import numpy as np
import sys, time, os, gc
import datetime


# Codebase
path_codebase = ['D:\Data_Mining\Projects\Codebase_YZ']

for i in range(0, len(path_codebase)):
    if path_codebase not in sys.path:
        sys.path = [path_codebase[i]] + sys.path
    
import toolbox as t
import codebase_rt as cbrt

# import debrieftool as dt
# from target_apyori_v1_4 import apriori


# %% 手動設定區 -------

# 在console中顯示全部的欄位
pd.set_option('display.max_columns', 30)
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8' # 修正SQL文字編碼



# %% OS系列 -------------

def os_create_folder(path=[]):
    '''
    新增資料夾
    path : str or list，可以一次輸入多個路徑
    '''
    if not isinstance(path, list):
        path = [path]
    
    for i in range(0, len(path)):
    
        if not os.path.exists(path[i]):
            os.mkdir(path[i])
    
    return ''


# ........


def os_get_avail_mem():
    '''
    取得可用的記憶體，單位為GB。
    '''

    import psutil
    return psutil.virtual_memory().available / (1024.0 ** 3)


# ..........
    

def os_get_dir_list(path, level=-1, extensions=None, remove_temp=True):

    '''
    Update, add pattern.
    file_format : str or list.
    level       : int, default -1. -1時會列出所有的子目錄和檔案。
    remove_temp : boolean, default True. 移除暫存檔，即名稱前面有~的檔案
    mode        : 0為完整路徑，1為檔案名稱
    '''
    
    path_level = path.count(os.path.sep)
    
    # walk包含三個部份 ......
    # [0] root
    # [1] subdirs
    # [2] files
    
    path_list =  list(os.walk(path))
    
    result_dirs = []
    result_files = pd.DataFrame()
    
    for i in range(0, len(path_list)):
        
        root = path_list[i][0]
        subdirs = path_list[i][1]
        files = path_list[i][2]
        
        cur_level = root.count(os.path.sep)

        if (level != -1) and \
            (path_level + level < cur_level):
                continue                
        
            
        # 子目錄 ......
        if len(subdirs) > 0:
            temp = li_ele_add(li=subdirs, 
                                  head=root + '/')
            
            result_dirs.append(temp)
            
        # 檔案 ......
        if len(files) > 0:
            
            # 移除不在file_format中的檔名
            if extensions != None:
            
                new_files = files.copy()
                
                for j in range(0, len(files)):
                    file_split = files[j].split('.')
                    file_split = file_split[-1]
                    
                    if file_split not in extensions:
                        new_files.remove(files[j])
                        
                files = new_files


            # 上面remove完後，files可能會是空的
            if len(files) > 0:
                
                # 保留路徑及完整檔名
                new_path = []
                new_name = []
                
                for k in range(0, len(files)):
                    
                    # 移除暫存檔 ...
                    if (remove_temp==True) and (files[k][0] == '~'):
                        continue
                    
                    
                    # if mode == 0:
                    #     row = [root + '/' + files[k]]
                    # elif mode == 1:
                    #     row = [files[k]]
                    # elif mode == 2:
                    #     row = [root + '/' + files[k], files[k]]
                        
                        
                    new_path.append(root + '/' + files[k])
                    new_name.append(files[k])
                    
                    
                new_file_df = pd.DataFrame({'PATH':new_path,
                                            'FILE_NAME':new_name})

                result_files = result_files.append(new_file_df)
        
    
    # 資料整理 ......
    result_dirs = li_flatten(result_dirs)
    
    results={'DIRS':result_dirs,
             'FILES':result_files}

    return results


# ..........


def os_get_py_version():
    '''

    Returns
    -------
    py_version : float
        DESCRIPTION.

    '''

    py_version = sys.version_info
    py_version = str(sys.version_info.major) + '.' \
                    + str(sys.version_info.minor)
    
    py_version = float(py_version)
    
    return py_version


# .....................
    

def os_load_files_from_dir(path, level=0):
    '''
    讀取資料夾下的所有檔案，並且合併成DataFrame
    

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    level : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    path_list = os_get_dir_list(path=path, level=level)
    path_list = path_list['FILES']
    
    # 讀取資料
    df = pd.DataFrame()

    for i in range(len(path_list)):
        file = pd.read_csv(path_list['PATH'][i])
        df = df.append(file)
        print(str(i) + '/' + str(len(path_list)))
        
    df = df.reset_index(drop=True)
        
    return df



# %% List系列 -------------

def repeat_li_ele(x, times):
    '''
    重覆list中的每一個elements。
    x = [1, 2, 3]
    y = [2, 3, 4]
    repeat_list_elements(x, y)
    '''    
    return [item for item, count in zip(x, times) for i in range(count)]


def li_add_quote(li, quote="'"):
    '''
    為list中的每一個物件加上括號。
    '''        
    results = [quote + s + quote for s in li]
    return results


# ...........
    

def li_conv_ele_type(li, to_type='str'):
    '''
    轉換list中每一個物件的data type
    '''
    if to_type == 'str':
        results = list(map(str, li))
        
    elif to_type == 'int':
        results = list(map(int, li))
    
    return results
    

# ...........
    

def conv_to_list(obj):
    '''
    將物件轉換為list
    可以使用li_join_flatten取代
    '''
    
    if not isinstance(obj, list) :
        results = [obj]
    else:
        results = obj

    return results


# .....


def li_intersect(*args):
    
    li = [i for i in args]
    li = list(set(li[0]).intersection(*li))
    
    return li


# ......
    

def li_remove_empty(li):

    while "" in li: 
        li.remove("") 
        
    while [] in li: 
        li.remove([])
        
    return li


# ........
   

def li_remove_items(li, remove):
    '''
    移除list中的物件
    '''
    loc_li = li.copy()
    
    for i in remove:
        if i in loc_li:
            loc_li.remove(i)
        else:
            print(str(i) + '不在list中')
    
    return loc_li


# ................


def li_flatten(li, join=False, divider=", "):
    '''
    透過遞迴的方式，將巢狀的list變成flatten list。
    join:    Boolean, default False. 是否合併為字串
    '''
    
    if (not isinstance(li, list)) or (len(li) == 0):
        return li
    
    # 移除空值，避免出錯
    li = li_remove_empty(li)

    
    if len(li) == 1:
        if type(li[0]) == list:
            
            results = li_flatten(li[0])
        else:
            results = li
            
    elif type(li[0]) == list:
        results = li_flatten(li[0]) + li_flatten(li[1:])
        
    else:
        results = [li[0]] + li_flatten(li[1:])

        
    if join == True:
        results = li_conv_ele_type(results, 'str')
        results = divider.join(results)
        
    return results


# ......
    

def li_join_flatten(*args, flatten=True, join=False, drop_duplicates=False):
    '''
    將所有argument合併為flatten list。
    '''
    
    if len(args)==0:
        print('li_join_flatten中沒有輸入任何物件')
        

    results = [i for i in args]
    
    # if len(results) == 1:
    #     return results    
    
    if flatten == True:
        results = li_flatten(results, join=join)

    
    if drop_duplicates == True:
        results = li_drop_duplicates(results)
        
    
    return results


# .....
    
def li_rank(li, asced=True, since_zero=False):
    '''
    給予list排名
    asced      : boolean, default True. True時以數字最小的排名為0，False時以最大為0。 
    since_zero : boolean, default True. True時從0開始，False時從1開始。 
    '''
    if asced == False:
        li = [i * -1 for i in li]
        
    if since_zero==True:
        results = [sorted(li).index(x) for x in li]
    else :
        results = [sorted(li).index(x) + 1 for x in li]
    
    return results


# .........
    

def li_upper(li):
    '''
    Chk, element非文字時是否會出錯
    '''
    results = [x.upper() for x in li]
    return results


# .........
    

def li_lower(li):
    '''
    Chk, element非文字時是否會出錯
    '''    
    results = [x.lower() for x in li]
    return results



# .........

def li_drop_duplicates(li):
    '''
    移除重複
    '''
    return list(dict.fromkeys(li))


# ........
    

def li_ele_add(li, head='', tail=''):
    '''    
    修改list中的物件
    (1) 還沒測試list為int的情況
    (2) 如果list中的物件類型不同，應該直接return
    '''
    if (all(isinstance(x, str) for x in li)) and \
        (isinstance(head, str)) and \
        (isinstance(tail, str)):
            
            # if head != '':
            #     results = [head+i for i in li]
            # else:
            #     results = li
                
            # if tail != '':
            #     results = [i+tail for i in li]
                
            results = [head + i + tail for i in li]                
        
    elif (all(isinstance(x, int) for x in li)) and \
        (isinstance(head, int)):

            if head != '':
                results = [head+i for i in li]
            else:
                results = li   
                
            
    return results


# ..........
    

def li_join_as_str(li, divider=', '):
    '''
    將List合併成為字串
    '''
    
    if isinstance(li, str):
        return li
    
    li = li_conv_ele_type(li, to_type='str') 
    li = divider.join(li)
    
    return li


# ..........
    

def li_search(li, regex):
    
    import re
    
    # 移除Null和nan，不確定哪一個比較快 ......
    # 方法一
    li = [i for i in li if (i) and (i==i)]
    
    # 方法二
    # li = list(filter(None, li))     
    
    if len(li) == 0:
        return []
    
    r = re.compile(regex)    
    results = list(filter(r.match, li)) 
    
    return results


# ..............
    

def li_check_nested(li, check_tuple=True):
    
    if check_tuple == True:
        results = any(isinstance(i, list) for i in li) \
            or any(isinstance(i, tuple) for i in li)
    else:
        results = any(isinstance(i, list) for i in li)
        
    return results


# ............
    

def li_to_dict(key_list, value_list):
    '''
    將兩組list轉成dict
    '''
    results = {key_list[i]: value_list[i] for i in range(len(key_list))} 
    return results



# %% 字串系列 -------------
    
def str_replace_special(string, value=''):
    '''
    取代文字中的特殊符號，中文字會被排除
    '''
    import re
    results = re.sub('[^a-zA-Z0-9 \n\.]', value, string)    
    
    return results


# ........
    
def str_replace_space(string, value=''):
    resutls = string.replace(' ', value)
    return resutls


# ........
    
def str_remove_non_ascii(string):
    import re
    resutls = re.sub(r'[^\x00-\x7f]',r'', string) 
    return resutls


# ........


def str_clean_spec(string):
    '''
    清理商品規格資料
    '''    
    string = string.lower()

    string = string.replace('公克', 'g')
    string = string.replace('克', 'g')
    string = string.replace('公斤', 'kg')
    
    string = string.replace('毫升', 'ml')
    return string
    

# .............
    

def str_filter_chinese(context):

    '''
    只留下中文，會排除特殊符號和Emoji
    https://chenyuzuoo.github.io/posts/28001/
    '''
    
    import re    
    decode = True
    
    try:
        # 如果str已經被decode，再次執行時會出錯
        # convert context from str to unicode
        context = context.decode("utf-8") 
    except:
        decode=False
        # print('decode失敗')        
        
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
    
    if decode==True:
        context = context.encode("utf-8") # convert unicode back to str    
    
    return context


# .............
    

def str_get_punc():
    '''
    取得標點符號(punctuation), 但這裡的標點符號目前只包含英文中的標點符號，缺少
    中文中獨的有符號，如《》。
    '''
    import string 
    punc = list(string.punctuation)    
    results = pd.DataFrame({'PUNC':punc})
    
    return results


# .........


def str_conv_half_width(obj):
    '''
    此function只會把全形(full-width)轉為半形(half-width)，但不會排除特殊符號及
    表情符號
    '''
    import unicodedata
    # obj = '１２３ａｂｄ中文 data mining'
    results = unicodedata.normalize('NFKC', obj)
    
    return results


# .........
    

def str_remove_emoji(text):
    '''
    移除文字中的emoji

    Parameters
    ----------
    text : str
        

    Returns
    -------
    str

    '''
    import re
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


# .........
    

def str_clean_dev(obj, punc=True, emoji=True):
    '''
    未完成，可以使用str_clean取代。
    '''
    import re
    if punc == True:
        punc = ""        
        
        # 英文標點符號 ...
        en_punc = str_get_punc()
        en_punc = en_punc['PUNC'].tolist()
        en_punc = li_join_as_str(en_punc, divider='')
    
        # 中文標點符號 ...
        # 可以使用zhon.hanzi.punctuation取得
        cht_punc = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿
        ｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—
        ‘'‛“”„‟…‧﹏"""
        
        punc = en_punc + cht_punc
        punc_regex = "[{}] ".format(punc)
        obj = re.sub(punc_regex, "", obj)

    if emoji == True:
        obj = str_remove_emoji(obj)

    return ''


def str_clean(obj, remove_en=False, remove_ch=False, remove_num=False, 
              remove_blank=False): 
    '''
    清理文字內容，只保留英文、中文和數字。
    (1) Bug, 字串中的文字也會被當成str處理，因此目前無法確實排除。

    Parameters
    ----------
    obj : str
    remove_en : boolean
        移除英文
    remove_ch : boolean
        移除中文        
    remove_num : boolean
        移除數字
    remove_blank : boolean
        移除空格可能會造成英文中的錯誤

    Returns
    -------
    obj : str
    '''
    import re
    # regex = "^a-zA-Z0-9\u4e00-\u9fa5"
    regex = "^"

    if remove_en == False:
        regex = regex + "a-zA-Z"

    if remove_num == False:
        regex = regex + "0-9"
        
        
    if remove_ch == False:
        # 中文範圍是到9fa5還是9fbf
        regex = regex + "\u4e00-\u9fa5"
        
            
    if remove_blank == False:
        regex = regex + " + u+2800"
    
    rule = re.compile(u"[" + regex + "]") 
    obj = rule.sub('', obj) 
    return obj


# %% Tuple系列 --------

def tuples_reverse(tup): 
    new_tup = tup[::-1] 
    return new_tup 


# %% DataFrame資料處理 -----
    
def df_add_agg(df, cols, group_by=[], size=True, min_val=True, max_val=True,
               mean_val=True, std=True, med=True, mode=True, drop=False,
               append_col=''):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.
    group_by : TYPE, optional
        DESCRIPTION. The default is [].
    min_val : TYPE, optional
        DESCRIPTION. The default is True.
    max_val : TYPE, optional
        DESCRIPTION. The default is True.
    mean_val : TYPE, optional
        DESCRIPTION. The default is True.
    std : TYPE, optional
        DESCRIPTION. The default is True.
    med : TYPE, optional
        DESCRIPTION. The default is True.
    mode : TYPE, optional
        DESCRIPTION. The default is True.
    drop : boolean, optional
        False時會回傳相同的列數，True時會回傳合併後的列數
    append_col : str, optional
        要插入在原有欄位名稱及agg method之間的文字. The default is True.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    '''
    
    merge_keys = df_get_cols_except(df, except_cols=cols)
    
    # ......
    group_by = conv_to_list(group_by)
    
    if len(group_by) == 0:
        group_by = ['TEMP_KEY']
        df['TEMP_KEY'] = 1
        
    
    # 全部加總的方式 ......
    agg_method = []
    
    if size == True:
        agg_method.append('size')
    
    if min_val == True:
        agg_method.append('min')

    if max_val == True:
        agg_method.append('max')
        
    if mean_val == True:
        agg_method.append('mean')    
        
    if std == True:
        agg_method.append('std')    

    if med == True:
        agg_method.append('median')    

    if mode == True:
        agg_method.append('mode')    

    
    # 先複製一份Results，避免df的資料量在loop中一直變大
    # 移除重複列
    if drop:    
        drop_cols = df_get_cols_except(df=df, except_cols=group_by)    
        results = df.drop(drop_cols, axis=1) \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    else:
        results = df.copy()
        
        
    for i in range(len(agg_method)):
        
        if agg_method[i] != 'size':
            
            temp_df = df[group_by + cols] \
                .groupby(group_by) \
                .agg('mean') \
                .reset_index()
                
            # 更改欄位名稱 ......            
            new_cols = [j+append_col+'_'+agg_method[i].upper() for j in cols]  

                
            rename_dict = li_to_dict(cols, new_cols)            
            temp_df = temp_df.rename(columns=rename_dict)   
            
        else:
            temp_df = df[group_by] \
                        .groupby(group_by) \
                        .size() \
                        .reset_index(name='SIZE'+append_col)

        # 合併物件        
        results = results.merge(temp_df, how='left', on=group_by)            
    

    return results


# ......................
 

   
def df_add_rank(df, value, group_by=[], sort_ascending=False, 
                rank_ascending=True,
                rank_name='RANK',
                rank_method='min', inplace=False):
    '''    
    增加DataFrame的組內排名
    Update, 當有同名資料列時作出提醒
    '''

    
    group_by = conv_to_list(group_by)
    py_version = os_get_py_version()
    
    
    if inplace:
        loc_df = df
    else:
        loc_df = df.copy()
    
    
    # 排序
    if len(group_by) > 0:
        sort_key = li_join_flatten(group_by, value)
        loc_df = loc_df.sort_values(by=sort_key, ascending=sort_ascending) \
                .reset_index(drop=True)
    else:
        loc_df = loc_df.sort_values(by=value, ascending=sort_ascending) \
                .reset_index(drop=True)
           
    
    # 主工作區 .....
    # if len(group_by) > 0:
        
    #     if py_version >= 3.8:
    #         loc_df[rank_name] = loc_df \
    #                         .groupby(group_by)[value] \
    #                         .rank(rank_method, ascending=rank_ascending)
    #     else:
    #         loc_df[rank_name] = loc_df \
    #                         .groupby(group_by)[value] \
    #                         .transform('rank', method=rank_method,
    #                                    ascending=rank_ascending) 
    # else:
    #     if py_version >= 3.8:
    #         loc_df[rank_name] = loc_df[value].rank(ascending=rank_ascending)                        
                                
    #     else:
    #         loc_df[rank_name] = loc_df[value].transform('rank', method=rank_method,
    #                                             ascending=rank_ascending) 
    

    if len(group_by) > 0:
        
        loc_df[rank_name] = loc_df \
                            .groupby(group_by)[value] \
                            .rank(rank_method, ascending=rank_ascending)
    else:
        loc_df[rank_name] = loc_df[value].rank(ascending=rank_ascending)
        
        
    return loc_df


# ...........
    

def df_add_ratio(df, value, col_name='RATIO', by=None, drop_total=True):
    '''
    計算欄位的占比。
    
    Update, 讓value可以輸入多個欄位
    '''
    
    if by != None:
        df['TOTAL'] = df.groupby(by)[value].transform(sum)
        df[col_name] = df[value] / df['TOTAL']
    else:
        df['TOTAL'] = df[value].sum()
        df[col_name] = df[value] / df['TOTAL']
    
    
    if drop_total == True:
        df = df.drop('TOTAL', 1)
    
    return df


# .................


def df_add_size(df, group_by=None, col_name='SIZE'):
    
    '''
    執行DataFrame的size
    '''

    
    if group_by != None:
        group_key = conv_to_list(group_by)
        
        size_df = df.groupby(group_by) \
                    .size() \
                    .reset_index(name=col_name)
    
        df = df.merge(size_df, how='left', on=group_key)
        
    else:
        size_col = len(df)
        df['SIZE'] = size_col

    return df    



# ...........
    

def df_change_row_index(df, original , to):
    '''
    變更DataFrame中某一列的index
    '''
    
    index_li = df.index.tolist()
    index = index_li.index(original)
    index_li[index] = to
    df.index = index_li        
    
    return df


# ...........




# .....................
    
def df_cal_weekday(df, col):
    '''
    col   : 日期欄位名稱
    '''
    results = df.copy()
    
    # 待優化
    # if results[col].dtype:
    results[col] = results[col].apply(ymd)
    
    # 星期一是0，星期日是6
    results['WEEKDAY'] = results[col].dt.weekday
    results['WEEKEND'] = results['WEEKDAY'].isin([5,6])
    
    return results


# ..............
    

def df_cross_join(df1, df2):

    loc_df1 = df1.copy()
    loc_df2 = df2.copy()
    
    loc_df1['CROSS_KEY'] = 1
    loc_df2['CROSS_KEY'] = 1
           
    results = loc_df1.merge(loc_df2, on='CROSS_KEY') \
                .drop('CROSS_KEY', 1) \
                .reset_index(drop=True)
                    
    
    return results


# ..................


def df_flatten_columns(df, keep_level=[], divider='_', upper=True, key_ahead=True):
    
    import re
    
    # 不是Nested時繼續執行會出錯
    # Check，確認第二個reset_index該不該加
    # 應該要用list(df.columns)?
    if li_check_nested(list(df.columns)) == False:
        df = df \
            .reset_index() \
            .reset_index(drop=True)
        return df
    
    
    # cols_raw = list(df.columns.values)
    cols_raw = list(df.columns)
    
    
    # 只保留某個level的header
    if keep_level != []:
        
        keep_level = conv_to_list(keep_level)
        new_cols = []
        
        for i in range(len(cols_raw)):
            
            new_ele = []
            for j in range(len(keep_level)):
                new_ele.append(cols_raw[i][keep_level[j]])
                
            new_cols.append(new_ele)
        
        cols_raw = new_cols
        

    
    if key_ahead == True:
        cols_raw_new = []
        
        for i in range(0, len(cols_raw)):
            temp = tuples_reverse(cols_raw[i])
            cols_raw_new.append(temp)

        cols_raw = cols_raw_new

    
    # 列出所有的欄位名稱
    flatten_cols_pre = [divider.join(col).strip() \
                    for col in cols_raw]

    
    # 當key_ahead為False時，後面會多divider；True時，前面會多divider。在這裡排除。
    flatten_cols = []
    for j in flatten_cols_pre:
        
        if key_ahead == True:
            new = re.sub('^'+divider, '', j)            
        else:
            new = re.sub(divider+'$', '', j)
        flatten_cols.append(new)


    # 轉為大寫    
    if upper==True:
        flatten_cols = li_upper(flatten_cols)


    df.columns = flatten_cols
    return df    


# .............
    

def df_fillna(df, cols, group_by=[], method='mean'):
    '''
    Update, group_by還沒寫

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.
    group_by : TYPE, optional
        DESCRIPTION. The default is [].
    method : TYPE, optional
        DESCRIPTION. The default is 'mean'.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    cols = conv_to_list(cols)
    group_by = conv_to_list(group_by)
    
    
    for c in cols:
        
        if method == 'mean':
            if len(group_by) == 0:
                col_mean = df[c].mean().to_dict()
                df.fillna(value={c:col_mean}, inplace=True)
            else:

                # mean_col = c + '_MEAN'
                # mean_series = df.groupby(group_by).agg({c:'mean'}) \
                #     .rename(columns={c:mean_col})
                
                # df = df.merge(mean_series, how='left', on=group_by)
                # df[c] = np.where(df[c].isna(), df[mean_col], df[c])
                
                # df = df.drop(mean_col, axis=1)
                
                
                # 這個寫法沒有用
                # https://stackoverflow.com/questions/46391128/pandas-fillna-using-groupby
                df[c] = df.groupby(group_by)[c] \
                    .apply(lambda x: x.fillna(x.mean()))
                    
    return df


# .............
    

def df_clean_spec(df, col):
    '''
    清理商品規格資料
    '''
    loc_df = df.copy()
    spec_series = loc_df[col]
    spec_series = spec_series.str.lower()

    spec_series = spec_series.replace('公克', 'g')
    spec_series = spec_series.replace('克', 'g')
    spec_series = spec_series.replace('公斤', 'kg')
    
    spec_series = spec_series.replace('毫升', 'ml')

    loc_df[col] = spec_series
    
    return loc_df


# ................
    

def df_date_cal(df, amount, cols, new_cols=None, suffix=None, 
                unit="m", simplify=True):
    '''
    '''
    loc_df = df.copy()
    cols = conv_to_list(cols)
    
    # 新欄位名稱 .............
    # 如果不指定新欄位名稱，也不指定後綴，則直接回寫原本的欄位
    if new_cols != None:
        new_cols = conv_to_list(new_cols)    
        
    elif suffix != None:
        new_cols = li_ele_add(cols, tail=suffix)
        
    elif (new_cols == None) and (suffix==None):
        new_cols = cols
        print('df_date_cal覆寫日期欄位')
    
    
    for i in range(0, len(cols)):
    
        if unit=="y":
            loc_df[new_cols[i]] = loc_df[cols[i]] \
                                + pd.offsets.DateOffset(years=amount)
        elif unit=="m":
            loc_df[new_cols[i]] = loc_df[cols[i]] \
                                + pd.offsets.DateOffset(months=amount)
        elif unit=="w":
            # loc_df[new_cols[i]] = loc_df[cols[i]] + datetime.timedelta(weeks=amount)
            loc_df[new_cols[i]] = loc_df[cols[i]] \
                                + pd.offsets.DateOffset(weeks=amount)            
        elif unit=="d":
            # loc_df[new_cols[i]] = loc_df[cols[i]] + datetime.timedelta(days=amount)
            loc_df[new_cols[i]] = loc_df[cols[i]] \
                                + pd.offsets.DateOffset(days=amount)            
        
    
    return loc_df


# ..............
    

def df_date_diff(df, col1, col2, name='DAYS', absolute=False, inplace=False):
    
    if inplace == False:
        col1_pre = col1
        col2_pre = col2
        
        col1 = col1 + '_TEMP'
        col2 = col2 + '_TEMP'
        
        df[col1] = df[col1_pre]
        df[col2] = df[col2_pre]
    
    df = df_ymd(df=df, cols=[col1, col2])
    
    
    if absolute:
        df[name] = abs((df[col1] - df[col2]).dt.days)
    else:
        df[name] = (df[col1] - df[col2]).dt.days
    

    # 刪除暫時欄位
    # if inplace == False:
    df = df.drop([col1, col2], axis=1)
    
    return df


# ...........


def df_get_cols_contain(df, string=[]):
    
    string = conv_to_list(string)
    cols = list(df.columns)
    loc_df = pd.DataFrame({'COLS':cols})
    target_cols = []
    
    for s in string:
        temp = loc_df[loc_df['COLS'].str.contains(s)]
        new_cols = temp['COLS'].tolist()
        target_cols = target_cols + new_cols
    
    return target_cols


# .............
    

def df_get_cols_except(df, except_cols=[]):
    
    except_cols = conv_to_list(except_cols)
    
    cols = df.columns
    cols = list(cols)
    cols = li_remove_items(cols, except_cols)
    
    return cols  


# ......
    

def df_li_search(df, cols, regex,  col_head='', col_tail=''):
    
        
    if col_head == '' and (col_tail == ''):
        print('df_li_search覆寫欄位')    
    
    cols = conv_to_list(cols)
    loc_df = df.copy().reset_index(drop=True)

    # ......
    for i in range(0, len(cols)):
        loc_df[col_head + cols[i] + col_tail] = loc_df[cols[i]] \
                            .apply(li_search, regex='最愛分店_')        

    return loc_df


# ..................
    

def df_chk_col_na(df, cols=None, positive_only=False, return_obj=True,
                  alert=False, alert_obj=None):
    '''
    '''
    loc_df = df.copy()
    
    if cols == None:
        cols = list(loc_df.columns)
    
    results_col = []
    results_na = []
    
    
    for i in cols:
        temp = loc_df[loc_df[i].isna()]
        results_col.append(i)
        results_na.append(len(temp))
        
    
    results = pd.DataFrame(data={'COLUMN':results_col, 
                                 'NA_COUNT':results_na})
    
    if positive_only:
        results = results[results['NA_COUNT']>0] \
                    .reset_index(drop=True)
    
    if alert and len(results):
        msg = 'df_chk_col_na - ' + str(alert_obj) + '中有na'
        print(msg)
        print(results)
    
    if return_obj:
        return results



# ..............


def df_add_level(df, start, stop, step, col, col_name='LEVEL', unit='',
                 method=1):
    '''
    共有兩種方式，method=1為含下不含上，method=2為上下都含
    '''
    
    
    if method == 1:
        lv_list = list(range(start, stop+step, step))
        
        # 轉成str，並補上leading 0，避免產出的結果沒辦法排序
        lv_name = [str(i).zfill(len(str(stop))) for i in  lv_list]
        
        for i in range(0, len( lv_list)+1):
            
            if i == 0:
                df.loc[df[col]< lv_list[i], col_name] = \
                    '-' + str( lv_name[i]) + unit
            
            elif i == len( lv_list):
                df.loc[df[col]>= lv_list[i-1], col_name] = \
                    str( lv_list[i-1]) + unit + '-'
            else:
                df.loc[(df[col]>= lv_list[i-1]) & (df[col]< lv_list[i]),
                       col_name] = \
                    str( lv_name[i-1]) + '-' + str( lv_name[i]) + unit
            
                
    elif method == 2:
        
        lv_list_min = list(range(start, stop+step, step))
        lv_list_max = [i - 1 for i in lv_list_min]
        
        lv_list_min = ['NA'] + lv_list_min
        lv_list_max = lv_list_max + ['NA']
        
        # 轉成str，並補上leading 0，避免產出的結果沒辦法排序
        lv_min_name = [str(i).zfill(len(str(stop))) for i in  lv_list_min]
        lv_max_name = [str(i).zfill(len(str(stop))) for i in  lv_list_max]
        
        
        for i in range(0, len(lv_list_min)):
            
            if i == 0:
                df.loc[df[col]<=lv_list_max[i], col_name] \
                    = '-' + str(lv_max_name[i]) + unit
            
            elif i == len(lv_list_min) - 1:
                df.loc[df[col]>=lv_list_min[i], col_name] \
                    = str(lv_min_name[i]) + unit + '-'
            
            else:
                df.loc[(df[col]>=lv_list_min[i]) \
                           & (df[col]<=lv_list_max[i]), col_name] \
                    = str(lv_min_name[i]) + '-' + str(lv_max_name[i]) + unit
            
    return df




# def df_add_level_20210617(df, start, stop, step, col, col_name='LEVEL', unit='',
#                  method=1):
#     '''
#     共有兩種方式，method=1為含下不含上，method=2為上下都含
#     '''
    
    
    
#     if method == 1:
#         level_list = list(range(start, stop+step, step))
        
#         for i in range(0, len(level_list)+1):
            
#             if i == 0:
#                 df.loc[df[col]<level_list[i], 
#                            col_name] = '<' + str(level_list[i]) + unit
            
#             elif i == len(level_list):
#                 df.loc[df[col]>=level_list[i-1], 
#                            col_name] = '>=' + str(level_list[i-1]) + unit
#             else:
                
#                 df.loc[(df[col]>=level_list[i-1]) \
#                            & (df[col]<level_list[i]), col_name] = \
#                     str(level_list[i-1]) + '-' + str(level_list[i]) + unit
                
#     elif method == 2:
        
#         level_list_min = list(range(start, stop+step, step))
#         level_list_max = [i - 1 for i in level_list_min]
        
        
#         level_list_min = ['NA'] + level_list_min
#         level_list_max = level_list_max + ['NA']
        
#         for i in range(0, len(level_list_min)):
            
#             if i == 0:
#                 df.loc[df[col]<=level_list_max[i], 
#                            col_name] = '<=' + str(level_list_max[i]) + unit
            
#             elif i == len(level_list_min) - 1:
#                 df.loc[df[col]>=level_list_min[i], 
#                            col_name] = '>=' + str(level_list_min[i]) + unit
            
#             else:
#                 df.loc[(df[col]>=level_list_min[i]) \
#                            & (df[col]<=level_list_max[i]), col_name] = \
#                     str(level_list_min[i]) + '-' + str(level_list_max[i]) + unit
            
#     return df






# def df_add_level_backup20210401(df, start, stop, step, col, col_name='LEVEL', unit=''):
    
    
#     loc_df = df.copy()
#     level_list = list(range(start, stop+step, step))
    
#     for i in range(0, len(level_list)+1):
        
#         if i == 0:
#             loc_df.loc[loc_df[col]<level_list[i], 
#                        col_name] = '<' + str(level_list[i]) + unit
#             continue
        
#         if i == len(level_list):
#             loc_df.loc[loc_df[col]>=level_list[i-1], 
#                        col_name] = '>=' + str(level_list[i-1]) + unit
#             break            
        
#         loc_df.loc[(loc_df[col]>=level_list[i-1]) \
#                    & (loc_df[col]<level_list[i]), col_name] = \
#             str(level_list[i-1]) + '-' + str(level_list[i]) + unit
            
    
#     return loc_df

# .............
    

def df_date_simplify(df, cols):
    '''
    '''
    loc_df = df.copy()
    cols = conv_to_list(cols)
    
    for i in range(0, len(cols)):
        
        loc_df[cols[i]] = loc_df[cols[i]].apply(date_simplify)
        
    return loc_df


# ............
    

def df_conv_date_to_week_head(df, cols):
    '''
    將日期欄位轉換成每周的第一天，通常的使用時機是圖表要用周為單位，但X軸又想以
    日期顯示，而不是用周次。

    Update, 目前cols只能輸入一欄
    '''
    
    loc_df = df.copy()
    cols = conv_to_list(cols)
    
    # 讀取日期資料 ......
    if isinstance(loc_df[cols[0]], int):
        calendar = get_rt_calendar(simplify=True)
    else:
        calendar = get_rt_calendar(simplify=False)
    
    
    # 避免dataframe和calendar的欄位名稱不一樣。
    calendar = calendar.rename(columns={'WORK_DATE':cols[0]})
    
    # 每周第一天的日期
    week_head = calendar.copy()
    week_head['WEEK_HEAD'] = week_head \
                            .groupby(['WEEK_ID'])['WORK_DATE'] \
                            .transform(min)

    # 合併資料 ......
    loc_df = loc_df.merge(week_head, how='left', on=cols)
    drop_key = li_join_flatten(['YEAR', 'WEEKDAY', 'WEEK_NUM'], cols)
    
    loc_df = loc_df.drop(drop_key, axis=1)
    loc_df = loc_df.rename(columns={'WEEK_HEAD':cols[0]})
    
    return loc_df


# .............
    

def df_conv_col_type(df, cols, to, ignore=False):
    '''
    一次轉換多個欄位的dtype
    '''
    loc_df = df.copy()
    cols = conv_to_list(cols)
    
    for i in range(len(cols)):

        if ignore :
            try:
                loc_df[cols[i]] = loc_df[cols[i]].astype(to)
            except:
                print('df_conv_col_type - ' + cols[i] + '轉換錯誤')
                continue
        else:
            loc_df[cols[i]] = loc_df[cols[i]].astype(to)
 
    
    return loc_df


# ............
    

def df_add_quantile(df, cols, group_by=[], suffix='_PR'):
    '''
    增加DataFrame的百分位數
    (1) 由於pd.qcut的值必須為unique，所以不採用。
    '''
    
    print('df_add_quantile的算法有BUG')
    return 'df_add_quantile的算法有BUG'
    
    
    cols = conv_to_list(cols)
    group_by = conv_to_list(group_by)
    
    if len(group_by) == 0:
        
        for i in range(0, len(cols)):
            temp_col = cols[i] + '_PR'
            df = df \
                .sort_values(by=cols[i]) \
                .reset_index(drop=True) \
                .reset_index()
            
            # 計算PR值
            df[temp_col] = (df['index'] / len(df)) * 100
            df[temp_col] = df[temp_col].apply(np.floor).astype(int)
            
            df = df.drop('index', axis=1)
        results = df.copy()
        
    else:
        group_key = df[group_by] \
                    .sort_values(by=group_by[0]) \
                    .drop_duplicates() \
                    .reset_index(drop=True) \
                    .reset_index() \
                    .rename(columns={'index':'GRP_KEY'})
                            
        loc_df = df.merge(group_key, how='left', on=group_by)
        
        
        # 計算百分位數
        for i in range(0, len(cols)):
            results = pd.DataFrame()
            
            for j in range(len(group_key)):
                temp_col = cols[i] + '_PR'
                temp_df = loc_df[loc_df['GRP_KEY']==j]
                
                temp_df = temp_df \
                    .sort_values(by=cols[i]) \
                    .reset_index(drop=True) \
                    .reset_index()
                
                # 計算PR值
                temp_df[temp_col] = (temp_df['index'] / len(temp_df)) * 100
                temp_df[temp_col] = temp_df[temp_col] \
                                    .apply(np.floor) \
                                    .astype(int)
                
                temp_df = temp_df.drop('index', axis=1)
                results = results.append(temp_df)
                
            results = results.reset_index(drop=True)
            loc_df = results.copy()
        
    return results



def df_add_quantile_backup_20210426(df, cols, suffix='_PR'):
    '''
    增加DataFrame的百分位數
    (1) 由於pd.qcut的值必須為unique，所以不採用。
    '''
    
    loc_df = df.copy()
    cols = conv_to_list(cols)
    
    for i in range(0, len(cols)):
        
        temp_col = cols[i] + '_PR'
        
        loc_df = loc_df \
                .sort_values(by=cols[i]) \
                .reset_index(drop=True) \
                .reset_index()
        
        # 計算PR值
        loc_df[temp_col] = (loc_df['index'] / len(loc_df)) * 100
        loc_df[temp_col] = loc_df[temp_col].apply(np.floor).astype(int)
        
        loc_df = loc_df.drop('index', axis=1)
        
    return loc_df


# .............
    

def df_str_col_subtract(df, col1, col2):
    '''
    將A欄位中的文字減掉B欄位，如「台北市內湖區」減掉「內湖區」。
    '''
    
    loc_df = df.copy()
    loc_df[col1] = [a.replace(b, '').strip() \
                      for a, b in zip(loc_df[col1], loc_df[col2])]
    
    return loc_df
    

# ...........
    

def df_anti_merge(df1, df2, on):
    '''
    Anti-merge / anti-join
    '''
    loc_df1 = df1.copy()
    loc_df2 = df2.copy()
    loc_df2['ANTI_LIST'] = True
    
    # 合併資料
    results = loc_df1.merge(loc_df2, how='left', on=on)
    
    results = results[results['ANTI_LIST'].isna()] \
                .drop('ANTI_LIST', axis=1) \
                .reset_index(drop=True)
    
    return results


# ...............
    

def df_str_clean(df, cols, remove_en=False, remove_ch=False, remove_num=False, 
                 remove_blank=False, drop=False):
    '''
    清理DataFrame中的文字內容，只保留英文、中文和數字。
    (1) Bug, 字串中的文字也會被當成str處理，因此目前無法確實排除。
    
    Parameters
    ----------
    df : DataFrame
    cols : str
        此function一次只套用至單一欄位
    remove_en : boolean
    remove_ch : boolean
    remove_num : boolean
    remove_blank : boolean
    drop : boolean
        排除為空值的欄位

    Returns
    -------
    loc_df : DataFrame

    '''
    loc_df = df.copy()
    loc_df[cols] = loc_df[cols].apply(str_clean,
                                      remove_en=remove_en, 
                                      remove_ch=remove_ch, 
                                      remove_num=remove_num, 
                                      remove_blank=remove_blank)
    
    if drop == True:
        loc_df = loc_df[(~loc_df[cols].isna()) & (loc_df[cols]!='')] \
                .reset_index(drop=True)

    return loc_df


# ...........


def df_add_shift(df, cols, shift, group_by=[], suffix='_PRE', 
                 remove_na=False):
    
    loc_df = df.copy()
    cols = conv_to_list(cols)
    
    shift_cols = cols
    shift_cols = li_ele_add(shift_cols, head='', tail=suffix)
    
    group_by = conv_to_list(group_by)

    for i in range(len(cols)):
        
        if len(group_by) > 0:
            loc_df[cols[i]+suffix] = loc_df \
                                    .groupby(group_by)[cols[i]] \
                                    .shift(shift)
        else:
            loc_df[cols[i]+suffix] = loc_df[cols[i]] \
                                    .shift(shift)
                                    
                                    
        # if len(group_by) > 0:
        #     loc_df.loc[loc_df.index, cols[i]+suffix] = \
        #         loc_df \
        #         .groupby(group_by)[cols[i]] \
        #         .shift(shift)
        # else:
        #     loc_df.loc[loc_df.index, cols[i]+suffix] = \
        #         loc_df[cols[i]].shift(shift)
                                    

    if remove_na==True:
        loc_df = loc_df.dropna().reset_index(drop=True)

    return loc_df, shift_cols


#　..............
    

def df_add_number_comma(df, cols):    
    '''
    四捨五入取到整數後，加上千分位逗號。
    使用這個function後，數值和逗號會被轉換成文字，匯出excel或csv時會無法計算。
    '''
    
    cols = conv_to_list(cols)
    
    for i in range(len(cols)):
        df = df.round({cols[i]: 0})
        df[cols[i]] = df[cols[i]].astype(int)
        df[cols[i]] = df[cols[i]].apply(lambda x : "{:,}".format(x))
    
    return df


# ...................
   

def df_read_excel(file, skiprows=None, sheet_name=None):
    '''
    pd.read_excel有時候會出錯，因此用這個function作為替代方案。

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    skip_rows : TYPE, optional
        DESCRIPTION. The default is 0.
    sheet_name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    from xlrd import open_workbook
    book = open_workbook(file, on_demand=True)
    df = pd.DataFrame()

    
    if sheet_name == None:
        sheet_name = book.sheet_names()
        sheet_name = sheet_name[0]


    # 由於pd.read_excel中的skiprows預設也是None，為了方便切換，因此這支function
    # 預設的skiprows也設為None
    if skiprows == None:
        skiprows = 0


    # 讀取資料表 ......
    sheet = book.sheet_by_name(sheet_name)
    df = pd.DataFrame()
    cols = sheet.ncols 
    
    for col_id in range(cols): 
        col = sheet.col(col_id)
        li = []

        for cell in col:
            li.append(cell.value)

        li = li[skiprows:]
        df[li[0]] = li[1:]    
            
    return df


# ............
    

def df_ymd(df, cols):
    
    cols = li_join_flatten(cols)
    
    for col in cols:
        df.loc[:, col] = df[col].apply(ymd)
    
    return df


# ............
    

def df_col_upper(df):
    
    cols = list(df.columns)
    cols = li_upper(cols)
    df.columns = cols
    
    return df


# %% DataFrame資料清理 -----

def df_replace_special(df, cols):
    
    loc_df = df.copy()
    for c in cols:
        loc_df[c] = loc_df[c].apply(str_replace_special)
    
    return loc_df

    


def df_shift_fill_na(df, loop_times, cols, group_by=[], forward=True, 
                     backward=True):
    
    cols = conv_to_list(cols)
    group_by = conv_to_list(group_by)

    # Prevent group_by empty ......
    if len(group_by) == 0:
        group_by = ['TEMP']
        df['TEMP'] = 1
        
    # Loop ......
    if forward:
        for i in range(loop_times):
            for j in cols:
                df.loc[df[j].isna(), j] = df.groupby(group_by)[j].shift(1)

    if backward:                
        for i in range(loop_times):
            for j in cols:
                df.loc[df[j].isna(), j] = df.groupby(group_by)[j].shift(-1) 
                

    if group_by[0] == 'TEMP':
        df = df.drop('TEMP', axis=1)
    
    return df


# ..............


def df_conv_all_inf(df, value=np.nan, drop=False):
    
    loc_df = df.copy()
    cols = list(loc_df)
    
    for i in range(0, len(cols)):
        loc_df[cols[i]] = loc_df[cols[i]] \
                .replace([np.inf, -np.inf], np.nan)
                
    if drop==True:
        loc_df = loc_df.dropna(subset=cols, how="all")

    return loc_df


# ................


def df_conv_inf(df, cols, value=np.nan, drop=False):
    
    loc_df = df.copy()
    cols = conv_to_list(cols)
    
    for i in range(0, len(cols)):
        loc_df[cols[i]] = loc_df[cols[i]] \
                .replace([np.inf, -np.inf], np.nan)
                
    if drop==True:
        loc_df = loc_df.dropna(subset=cols, how="all")

    return loc_df


# ...........
    

def df_conv_all_na(df, value=0):
    
    loc_df = df.copy()
    
    # 取得有NA的欄位
    cols = df_chk_col_na(loc_df)
    cols = cols[cols['NA_COUNT']>0]
    cols = cols['COLUMN'].tolist()

    # 轉換NA
    loc_df = df_conv_na(loc_df, cols=cols, value=value)
    
    return loc_df


# ..........


def df_conv_na(df, cols=None, value=0):
    '''
    轉換NAN 
    待新增其他版本，如series
    df
    cols   : list
    value  : int or str, default 0.
    '''
    
    if cols == None:
        cols = list(df.columns)
        cols = conv_to_list(cols)
    else:
        cols = conv_to_list(cols)
    
    for i in range(0, len(cols)):
        df.loc[(df[cols[i]].isna()) |
                    (df[cols[i]].isnull()), cols[i]] = value
    
    return df




# %% CSV系列 ............


def df_read_csv(file, delimiter=',', quotechar='"', header=True, 
                errors='ignore'):
    '''
    pd.read_csv出錯時的替代方案，常見的錯誤訊息如下，可能原因是特殊符號
    'utf-8' codec can't decode byte 0xaa in position 103: invalid start byte        
    '''
    import csv   
    
    
    try:
        results = pd.read_csv(file)
        return results
    except:
        print('pd.read_csv error.')
    
    
    row_list = []
    with open(file, newline='', errors=errors) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter, 
                                quotechar=quotechar)
        
        for row in spamreader:
            row_list.append(row)


    # 轉成DataFrame ......
    if header == True:
        cols = row_list[0]
        df_row = row_list[1:]
        results = pd.DataFrame(columns=cols,
                              data=df_row)            
    else:
        results = pd.DataFrame(data=row_list)     

    return results




def df_write_csv(df, file, encoding='big5', ignore_error=True):

    # 由於pandas v1.0.1中的to_csv沒有errors這個參數，在轉換encoding='big5'時常常出錯
    import csv

    with open(file, 'w', newline='', encoding=encoding) as csvfile:

        errors = []
        fieldnames = df.columns
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        df_dict = df.to_dict('records')

        for i in range(len(df_dict)):

            if ignore_error:
                try:
                    writer.writerow(df_dict[i])
                except Exception as e: 
                    errors.append(e)
            else:
                writer.writerow(df_dict[i])

    print('df_write_csv共有' + str(len(errors)) + '筆資料因轉換錯誤而忽略')    
    





# %% Excel系列...............
    

def excel_add_df(df, sht, startrow, startcol, header=True, index=False):
    '''
    1.使用pd.ExcelWriter。
    2. df.to_excel()會新增一個新的sheet，當使用多個df.to_excel()時不會有問
    題，但當合併df.to_excel()及worksheet.write('A1', 'TEXT')時會出現
    錯誤訊息：
    Sheetname 'name', with case ignored, is already in use.
    3.此function會用兩層for讀取dataframe，並一格一格塞進cell中。
    '''
    
    df = df.reset_index(drop=True)
    df = df_conv_all_inf(df, value='') # 將inf轉成na
    
    if header == True:
        values = list(df.columns)
        for col in range(len(values)):
            sht.write(startrow, startcol + col, values[col])
        startrow = startrow + 1
        
        
    if index == True:
        values = list(df.index)
        for row in range(len(values)):
            sht.write(startrow + row, startcol, values[row])
        startcol = startcol + 1
    
    
    # 填入Dataframe的數值
    for row in range(len(df)):
        values = df.iloc[row, :]
        
        for col in range(len(values)):
             
            # 避免數值為na
            if values[col] == values[col]:
                sht.write(startrow + row, startcol + col, values[col])
    
    return ''



def excel_add_format(sht, cell_format, startrow, endrow, 
                     startcol, endcol):
    '''
    增加數字格式
    
    https://xlsxwriter.readthedocs.io/format.html
    num_format = workbook \
        .add_format({'num_format': '_-* #,##0_-;-* #,##0_-;_-* "-"??_-;_-@_-'})
    
    digi_format = workbook.add_format({'num_format':'0.00'})
    sku_format = workbook.add_format({'num_format':'0'})
    date_format = workbook.add_format({'num_format':'yyyy/mm/dd'})    
    '''
    
    for i in range(startrow, endrow + 1):
        row_values = sht.table.get(i, None)
        
        for j in range(startcol, endcol + 1):
            
            if row_values == None:
                sht.write(i, j, '', cell_format) 
            else:
                cell_values = row_values.get(j, None)
                try:
                    value = cell_values.number
                except:
                    continue
                sht.write(i, j, value, cell_format)

    return ''



# %% Math系列 ------------

def df_cor(df, group, value, p_thld=0.05, method=0):
    '''
    計算Correlation
    1. Update, 如果兩邊的length不一樣時會出錯
    
    method: 0:Pearson; 1:Spearman; 2:Kendall
    
    '''
    import scipy.stats
    permute = df_permute(df=df, group=group)
    

    # 主工作區 ......
    results = pd.DataFrame()
    for i in range(len(permute)):
        
        grp_l = permute.loc[i, 'STOCK_SYMBOL_L']
        grp_r = permute.loc[i, 'STOCK_SYMBOL_R']
        
        x = df[df[group]==grp_l]
        x = x[value].to_numpy()
        
        y = df[df[group]==grp_r]
        y = y[value].to_numpy()
        
        # 計算Correlcation ......
        if method == 0:
            data = scipy.stats.pearsonr(x, y)    # Pearson's r
        elif method == 1:
            data = scipy.stats.spearmanr(x, y)   # Spearman's rho
        elif method == 2:
            data = scipy.stats.kendalltau(x, y)  # Kendall's tau    
    
    
        # 合併資料 ......
        data = [list(data)]
        new_results = pd.DataFrame(data=data, columns=['COEF', 'P_VALUE'])
        new_results['GROUP_L'] = grp_l
        new_results['GROUP_R'] = grp_r
        results = results.append(new_results)
        

    results = results[results['P_VALUE']<=p_thld] \
                .reset_index(drop=True)     
       
    results = results[['GROUP_L', 'GROUP_R', 'COEF', 'P_VALUE']]
    
    
    print('df_permute - 計算筆數為' + str(len(permute)) \
          + '，p值達標的筆數為' + str(len(results)))
        
    return results


    
def df_permute(df, group):
    '''
    不重複排列

    '''    
    
    group_li = conv_to_list(group)
    key_pre = df[group_li].drop_duplicates() \
            .reset_index(drop=True) \
            .reset_index()
    

    key_l = key_pre.rename(columns={'index':'INDEX_L',
                                    group:group+'_L'})
    
    key_r = key_pre.rename(columns={'index':'INDEX_R',
                                    group:group+'_R'})

    key = df_cross_join(key_l, key_r)
    key = key[key['INDEX_L']<key['INDEX_R']].reset_index(drop=True)
    
    return key




# %% Text系列 ................
    

def txt_export(obj, file):
    '''
    obj   : list or series.
    file  : 
    '''
    # 新增txt檔 ......
    # 必須以utf-8寫入，否則程式讀取或手動開啟時可能會有亂碼
    import codecs
    txt_file = codecs.open(file, "w", "utf-8")    
        
    for i in range(0, len(obj)):
        
        # 第一行不需要加空格
        if i > 0:
            txt_file. write("\n")
        
        txt_file.write(obj[i])
    txt_file.close()
    
    return txt_file



# %% 統計系列 ------------


def df_summary(df, cols, group_by=[], add_mean=True, add_min=True, add_max=True,
            add_median=True, add_std=True, add_skew=True, add_count=True):
    '''
    計算常用的統計指標，包括mean、min、max、median、sum、skew、count、mode
    Update，要讓cols也可以是list
    
    df     : DataFrame
    groups : list
    cols   : str
    '''
    
    loc_df = df.copy()
    cols = conv_to_list(cols)
    # pd.Series.mode，加上mode會出錯    
    
    # 確認計算方式 ......
    method = []
    
    if add_mean:
        method.append('mean')
        
    if add_min:
        method.append('min')
        
    if add_max:
        method.append('max')

    if add_median:
        method.append('median')

    if add_std:
        method.append('std')

    if add_skew:
        method.append('skew')

    if add_count:
        method.append('count')        
       
        
    # 計算區 ......
    if len(group_by) == 0:
        loc_df['TEMP'] = 1
        group_by = ['TEMP']
        
    
    for i in range(len(cols)):
        
        c = cols[i]
        temp = loc_df \
                .groupby(group_by)[c] \
                .agg(method) \
                .reset_index()

        # 重新命名 ...
        new_name = [c + '_' + m for m in method]
        rename_dict = li_to_dict(method, new_name)
        temp = temp.rename(columns=rename_dict)
            
        # 合併 ...
        if i > 0:
            result = result.merge(temp, how='outer', on=group_by)
        else:
            result = temp.copy()
            
        
    # 將欄位名稱變成大寫        
    result = df_col_upper(result)
    
        
    # 移除臨時欄位 ......            
    if group_by[0] == 'TEMP':
        result = result.drop('TEMP', axis=1)
    
    return result



# def summary_20210729(df, cols, group_by=[], add_mean=True, add_min=True, add_max=True,
#             add_median=True, add_std=True, add_skew=True, add_count=True):
#     '''
#     計算常用的統計指標，包括mean、min、max、median、sum、skew、count、mode
#     Update，要讓cols也可以是list
    
#     df     : DataFrame
#     groups : list
#     cols   : str
#     '''
    
#     loc_df = df.copy()
#     # pd.Series.mode，加上mode會出錯    
    
#     if len(group_by) == 0:
#         loc_df['TEMP'] = 1
#         group_by = ['TEMP']
        
    
#     results = loc_df \
#         .groupby(group_by)[cols] \
#         .agg(['mean', 'min', 'max', 'median', 'std', 
#               'skew', 'count']) \
#         .reset_index()
        
#     if group_by[0] == 'TEMP':
#         results = results.drop('TEMP', axis=1)
    
#     return results


def df_add_mode(df, group_key, value, name='MODE'):
    
    df[name] = df \
        .groupby(group_key)[value] \
        .transform(lambda x: x.value_counts().idxmax())
    
    return df



# %% 通用分析系列 ------------



def df_add_ma(df, cols, group_by, date_col, values=[5,20], wma=False):
    '''
    這支function會取代原本的物件

    Parameters
    ----------
    loc_df : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.
    group_by : TYPE
        DESCRIPTION.
    date_col : TYPE
        DESCRIPTION.
    values : TYPE, optional
        DESCRIPTION. The default is [5,20].

    Returns
    -------
    loc_df : TYPE
        DESCRIPTION.
    ma_cols : TYPE
        DESCRIPTION.

    '''
    
    # 5MA：短期操作
    # 20MA：月均線
    # 60MA：季均線，常用來觀察整體趨勢
    
    
    loc_df = df.copy()
    cols = conv_to_list(cols)
    group_by = conv_to_list(group_by)
    date_li = conv_to_list(date_col)    
    
    
    if len(group_by) == 0:
        loc_df['TEMP_GROUP'] = 1
        group_by = ['TEMP_GROUP']
    
    
    values = conv_to_list(values)
    values.sort(reverse=True)
    

    # 為了算出完整的MA，資料區間一定會往前推, 這段時間只是為了探供資料，但
    # 不需要算MA      
    date_divider = loc_df[[date_col]] \
                    .drop_duplicates() \
                    .sort_values(by=date_col) \
                    .reset_index(drop=True)
                    
    date_divider = date_divider.loc[values[0] - 1, date_col]
    
    
    # Calculate MA values ......    
    ma_cols = []
    for i in range(len(cols)):
        
        shift_cols = []        
        col = cols[i]
        
        
        # Add Shift Columns ......
        for j in range(1, values[0] + 1):
            shift_cols.append(str(j))
            loc_df[str(j)] = loc_df.groupby(group_by)[col].shift(j)        
        
        
        # Calculate MA
        for v in range(len(values)):
            value = values[v]
            
            if v == 0:
                temp_cols = [str(k) for k in range(1, value + 1)]
                
                temp_main = loc_df[group_by + date_li + temp_cols]
                temp_main = temp_main.melt(id_vars=group_by+date_li)
                temp_main = temp_main.dropna(subset=['value'], axis=0)
                
                temp_main['variable'] = temp_main['variable'].astype(int)
                temp_main = temp_main[temp_main['WORK_DATE']>date_divider]
                
            # Remove NA Value ......
            temp = temp_main[temp_main['variable']<=value]
            
            
            # Calculate For Weighted Moving Average
            if wma:
                period_sum = sum(range(1, value + 1))
                temp['value'] = temp['value'] * temp['variable']
                
                temp = temp \
                        .groupby(group_by + date_li) \
                        .agg({'value':'sum'}) \
                        .reset_index() 
                        
                temp['value'] = temp['value'] / period_sum
                temp = temp.rename(columns={'value':col + '_MA_' + str(value)})                
                
            else:
                # Calcuate Average ......
                temp = temp \
                        .groupby(group_by + date_li) \
                        .agg({'value':'mean'}) \
                        .reset_index() \
                        .rename(columns={'value':col + '_MA_' + str(value)})
                    
                    
            loc_df = loc_df.merge(temp, how='left', on=group_by+date_li)
            ma_cols.append(col + '_MA_' + str(value))
            
            
            # Delete Objects
            if v == len(values) - 1:
                del temp_main
                gc.collect()            
            

        # Drop shift columns
        loc_df = loc_df.drop(shift_cols, axis=1)
        print('add_ma - ' + str(i) + '/' + str(len(cols)-1))
        
        
    if group_by[0] == 'TEMP_GROUP':
        loc_df = loc_df.drop('TEMP_GROUP', axis=1)
    
    return loc_df, ma_cols




# def df_add_ma_20210717(df, cols, group_by, date_col, values=[5,20], wma=False):
#     '''
#     這支function會取代原本的物件

#     Parameters
#     ----------
#     df : TYPE
#         DESCRIPTION.
#     cols : TYPE
#         DESCRIPTION.
#     group_by : TYPE
#         DESCRIPTION.
#     date_col : TYPE
#         DESCRIPTION.
#     values : TYPE, optional
#         DESCRIPTION. The default is [5,20].

#     Returns
#     -------
#     df : TYPE
#         DESCRIPTION.
#     ma_cols : TYPE
#         DESCRIPTION.

#     '''
    
#     # 5MA：短期操作
#     # 20MA：月均線
#     # 60MA：季均線，常用來觀察整體趨勢
    
#     cols = conv_to_list(cols)
#     group_by = conv_to_list(group_by)
#     date_li = conv_to_list(date_col)    
    
#     values = conv_to_list(values)
#     values.sort(reverse=True)
    

#     # 為了算出完整的MA，資料區間一定會往前推, 這段時間只是為了探供資料，但
#     # 不需要算MA      
#     date_divider = df[[date_col]] \
#                     .drop_duplicates() \
#                     .sort_values(by=date_col) \
#                     .reset_index(drop=True)
                    
#     date_divider = date_divider.loc[values[0] - 1, date_col]
    
    
#     # Calculate MA values ......    
#     ma_cols = []
#     for i in range(len(cols)):
        
#         shift_cols = []        
#         col = cols[i]
        
        
#         # Add Shift Columns ......
#         for j in range(1, values[0] + 1):
#             shift_cols.append(str(j))
#             df[str(j)] = df.groupby(group_by)[col].shift(j)        
        
        
#         # Calculate MA
#         for v in range(len(values)):
#             value = values[v]
            
#             if v == 0:
#                 temp_cols = [str(k) for k in range(1, value + 1)]
                
#                 temp_main = df[group_by + date_li + temp_cols]
#                 temp_main = temp_main.melt(id_vars=group_by+date_li)
#                 temp_main = temp_main.dropna(subset=['value'], axis=0)
                
#                 temp_main['variable'] = temp_main['variable'].astype(int)
#                 temp_main = temp_main[temp_main['WORK_DATE']>date_divider]
                
#             # Remove NA Value ......
#             temp = temp_main[temp_main['variable']<=value]
            
            
#             # Calculate For Weighted Moving Average
#             if wma:
#                 period_sum = sum(range(1, value + 1))
#                 temp['value'] = temp['value'] * temp['variable']
                
#                 temp = temp \
#                         .groupby(group_by + date_li) \
#                         .agg({'value':'sum'}) \
#                         .reset_index() 
                        
#                 temp['value'] = temp['value'] / period_sum
#                 temp = temp.rename(columns={'value':col + '_MA_' + str(value)})                
                
#             else:
#                 # Calcuate Average ......
#                 temp = temp \
#                         .groupby(group_by + date_li) \
#                         .agg({'value':'mean'}) \
#                         .reset_index() \
#                         .rename(columns={'value':col + '_MA_' + str(value)})
                    
                    
#             df = df.merge(temp, how='left', on=group_by+date_li)
#             ma_cols.append(col + '_MA_' + str(value))
            
            
#             # Delete Objects
#             if v == len(values) - 1:
#                 del temp_main
#                 gc.collect()            
            

#         # Drop shift columns
#         df = df.drop(shift_cols, axis=1)
#         print('add_ma - ' + str(i) + '/' + str(len(cols)-1))
    
#     return df, ma_cols



# ..................



def df_add_peaks(df, by, value, asced=True, since_zero=True):
    '''
    增加Peak欄位
    df    : DataFrame 
    by    : 
    value :  
    '''
    from scipy.signal import find_peaks
    loc_df = df.copy()

    # 增加迴圈用的KEY    
    loop_key = loc_df[by].drop_duplicates().reset_index(drop=True)
    loop_key['PEAK_KEY'] = range(0, len(loop_key))
    loc_df = loc_df.merge(loop_key, on=by)
    
    # 分析Peaks
    results = pd.DataFrame()
    
    for i in loop_key['PEAK_KEY']:
        
        temp = loc_df.copy()
        temp = temp[temp['PEAK_KEY']==i].reset_index(drop=True)
        
        temp_series = temp[value]
        peaks, peak_values = find_peaks(temp_series, prominence=1)
        
        # 合併資料
        temp_df = pd.DataFrame(peak_values)
        
        rank = li_rank(peak_values['prominences'], 
                         asced=asced, 
                         since_zero=since_zero)
        
        temp_df['PEAK_RANK'] = pd.Series(rank)
        
        temp_df.index = peaks
        temp_df.columns = temp_df.columns.str.upper()
        
        temp_df = temp.merge(temp_df, 
                             how='left',
                             left_index=True, 
                             right_index=True)
        
        results = results.append(temp_df)

    return results



# %% 日期系列 -------------

def date_conv_roc_to_ad(obj):
    '''
    從民國年轉成西元年
    
    1. Update, 目前只能轉換全數字的格式
    '''
    
    # 移除特殊符號，如110/5/1並補0
    
    # 轉換
    obj = str(obj)
    year = str(int(obj[0:3]) + 1911)
    date = obj[3:7]
    ad_date = int(year + date)
    
    return ad_date


    
def get_rt_calendar(begin_date=20190101, end_date=20251231, simplify=False,
                    tail_begin=False):
    '''
    TAIL_BEGIN :  True時保留tail_begin欄位。
    '''
    
    begin_date = ymd(begin_date)
    end_date = ymd(end_date)
    delta = end_date - begin_date
    date_list = pd.date_range(begin_date, periods=delta.days + 1, freq='d')    
    
    results = pd.DataFrame({'WORK_DATE':date_list})
    results['YEAR'] = results['WORK_DATE'].dt.year
    results['MONTH'] = results['WORK_DATE'].dt.month
    
    
    
    # 標記每月的第一天及最後一天 ......
    # 如果begin_date不是從1號開始的話會出錯 ......
    results['MONTH_BEGIN'] = results \
        .groupby(['YEAR', 'MONTH'])['WORK_DATE'] \
        .transform('min')
    
    results['MONTH_END'] = results \
        .groupby(['YEAR', 'MONTH'])['WORK_DATE'] \
        .transform('max')    
    
    results['MONTH_BEGIN'] = results['WORK_DATE'] == results['MONTH_BEGIN']
    results['MONTH_END'] = results['WORK_DATE'] == results['MONTH_END']    
    
    
    # 將星期日轉為每周第一天 ......
    results['WEEKDAY'] = results['WORK_DATE'].dt.weekday + 1
    results.loc[results['WEEKDAY']==7, 'WEEKDAY'] = 0
    
    
    # 增加WEEK_ID
    # WEEK_ID是個流水號，主要是用來處理同一周但跨年度的情形。
    results = results.reset_index().rename(columns={'index':'WEEK_ID'})
    results['WEEK_ID'] = results['WEEK_ID'] + results.loc[0, 'WEEKDAY']
    results['WEEK_ID'] = results['WEEK_ID'] / 7
    results['WEEK_ID'] = results['WEEK_ID'].apply(np.floor)
    results['WEEK_ID'] = results['WEEK_ID'].astype(int)
    
    
    # 增加周次 ......
    # py_version = os_get_py_version()
    
    # if py_version >= 3.8:
    #     results['WEEK_NUM'] = results['WORK_DATE'].dt.isocalendar().week
    # else:
    #     results['WEEK_NUM'] = results['WORK_DATE'].dt.week
    
    # 在3.8中有時候會出現錯誤訊息，因此上面先移除
    # AttributeError: 'DatetimeProperties' object has no attribute 'isocalendar'    
    results['WEEK_NUM'] = results['WORK_DATE'].dt.week
    
        
    results.loc[results['WEEKDAY']==0, 
                'WEEK_NUM'] = results \
                                .groupby('YEAR')['WEEK_NUM'] \
                                .shift(-1)
    
    # Shift完後，如果12/31是星期日，則WEEK_NUM會變成NA
    results.loc[results['WEEK_NUM'].isna(),
                'WEEK_NUM'] = results \
                                .groupby('YEAR')['WEEK_NUM'] \
                                .shift(1) + 1
    
    # isocalendar().week的結果，2016年第一周的week_num為53。以下排除這個問題
    results.loc[(results['MONTH']==1) \
             & (results['WEEK_NUM']>50), 'TAIL_BEGIN_WEEK'] = True
        
    results['TAIL_BEGIN_YEAR'] = results \
                                .groupby(['YEAR'])['TAIL_BEGIN_WEEK'] \
                                .transform(max)
                                
    results = df_conv_na(results, 
                         cols=['TAIL_BEGIN_WEEK', 'TAIL_BEGIN_YEAR'],
                         value=False)
    
    results.loc[results['TAIL_BEGIN_YEAR']==True, 
                'WEEK_NUM'] = results['WEEK_NUM'] + 1
    
    results.loc[results['TAIL_BEGIN_WEEK']==True, 'WEEK_NUM'] = 1
    results = df_conv_col_type(df=results, cols='WEEK_NUM', to=int)
    
    results['WEEK_NUM'].unique()
    
    # ......

    if simplify == True:
        results = df_date_simplify(df=results, cols='WORK_DATE')
        
        
    if tail_begin == False:
        results = results.drop(['TAIL_BEGIN_WEEK', 'TAIL_BEGIN_YEAR'], 
                               axis=1)          
        
    return results



# ............


def ymd(series):
    '''
    轉換成日期格式
    '''
    
    if isinstance(series, str):
        
        explode = False
        
        if '-' in series:
            series = series.split('-')
            explode = True
            
        if '/' in series: 
            series = series.split('/')
            explode = True
            
        if explode:            
            series = ["{:02d}".format(int(i)) for i in series]
            series = series[0] + series[1] + series[2]
            
        series = int(series)

    
        # or np.int64 == np.dtype(type(series)).type \
    if isinstance(series, int) \
        or isinstance(series, np.int32) \
        or isinstance(series, np.int64) \
        or isinstance(series, str):
            
        series = str(series)
        series = datetime.datetime(year = int(series[0:4]), 
                        month = int(series[4:6]), 
                        day = int(series[6:8]))
    
    return series


# ......................


def date_cal(obj, amount, unit="m", simplify=True):
    '''
    '''
    # from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    
    obj = ymd(obj)
    
    if unit=="y":
        results = obj + relativedelta(years=amount)
    elif unit=="m":
        results = obj + relativedelta(months=amount)
    elif unit=="w":
        results = obj + datetime.timedelta(weeks=amount)        
    elif unit=="d":
        results = obj + datetime.timedelta(days=amount)
        
    if simplify == True:
        results = date_simplify(results)
    
    return results


# ......................


def date_simplify(obj):
    '''
    將日期轉換成數字
    '''
    
    # 如果obj為Na，obj!=obj為True
    if (isinstance(obj, int)) or (obj!=obj):
        return obj

    # if isinstance(obj, int):
    #     return obj
    
    if (isinstance(obj, datetime.datetime)) or (isinstance(obj, datetime.date)):
    # if isinstance(obj, datetime):
        obj = obj.strftime('%Y%m%d')
        
    if isinstance(obj, str):
        obj = str_replace_special(obj)
    
    obj = int(obj)
    
    return obj

# .....................


def df_add_month_begin_end(df, date_col='WORK_DATE', simplify=True):
    '''    
    '''  
    import calendar
    
    loc_df = df.copy()
    new_cols = {'MONTH_BEGIN', 'MONTH_END'}
    existing_cols = loc_df.columns
    
    if len(new_cols.intersection(existing_cols)) > 0:
        raise ValueError('MONTH_BEGIN與MONTH_END欄位已存在')
    
    
    # Workarea .........
    month_begin = []
    month_end = []
    
    for i in range(0, len(loc_df)):

        cur_date = loc_df.loc[i, date_col]
        end = calendar.monthrange(cur_date.year,
                                          cur_date.month)[1]
        
        begin_new = datetime.datetime(year=cur_date.year, 
                                      month=cur_date.month, 
                                      day=1)
        month_begin.append(begin_new)
        

        end_new = datetime.datetime(year=cur_date.year, 
                                    month=cur_date.month, 
                                    day=end)
        month_end.append(end_new)        
        
    loc_df['MONTH_BEGIN'] = pd.Series(month_begin)
    loc_df['MONTH_END'] = pd.Series(month_end)
    
    
    if simplify == True:
        loc_df['MONTH_BEGIN'] = loc_df['MONTH_BEGIN'].apply(date_simplify)
        loc_df['MONTH_END'] = loc_df['MONTH_END'].apply(date_simplify)
    
    
    return loc_df


# .....................
    


def date_get_seq(begin_date, end_date=None, seq_length=None, interval=1, 
                 unit='d', simplify_date=False,
                 simplify_format=False, rename=True):
    '''
    

    Parameters
    ----------
    begin_date : TYPE
        DESCRIPTION.
    end_date : TYPE, optional
        DESCRIPTION. The default is None.
    seq_length : int
        最後回傳的資料列數
    interval : int, optional
        回傳的每個日期之間的間隔，可為正或負
    unit : TYPE, optional
        DESCRIPTION. The default is 'd'.
    simplify_date : TYPE, optional
        DESCRIPTION. The default is False.
    simplify_format : TYPE, optional
        DESCRIPTION. The default is False.
    rename : TYPE, optional
        DESCRIPTION. The default is True.
    ascending : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    
    在一般情況下，只有by day時會用到skip這個參數，而且在其他單位時，也比較容易
    先算出完整seq後再用%排除，因此目前skip只會在by day時使用。
    '''
    
    begin_date = date_simplify(begin_date)
    
    if end_date == None:
        
        if interval >= 0:
            end_date = date_cal(obj=begin_date, unit='d',
                                amount=(seq_length - 1) * (interval + 1))
            
        elif interval < 0:
            end_date = date_cal(obj=begin_date, unit='d',
                                amount=(seq_length - 1) * (interval - 1))
            
    elif end_date != None and seq_length == None:
        seq_length = int(abs(date_diff(begin_date, end_date)) / interval)

    
    # 讀取日曆 ......
    # 確認Calendar的開始與結束時間
    if begin_date > end_date:
        calendar_begin = end_date
        calendar_end = begin_date
        ascending = False
    else:
        calendar_begin = begin_date
        calendar_end = end_date 
        ascending = True        
        
    calendar = get_rt_calendar(begin_date=calendar_begin, 
                               end_date=calendar_end,
                               simplify=True)
    
    calendar = calendar.sort_values(by='WORK_DATE', ascending=ascending)
    
    if unit == 'm':
        calendar = calendar[['YEAR', 'MONTH', 'WORK_DATE']] \
                    .drop_duplicates(subset=['YEAR', 'MONTH'])
                    
    elif unit == 'w':
        calendar = calendar[['YEAR', 'WEEK_NUM', 'WORK_DATE']] \
                    .drop_duplicates(subset=['YEAR', 'WEEK_NUM'])
    
    
    # 排除不必要的資料列 ......
    keep_index = [i * (abs(interval)+1) for i in range(seq_length)]
    
    calendar = calendar.reset_index(drop=True).reset_index()
    calendar.loc[0, 'BEGIN'] = True
    calendar = calendar[calendar['index'].isin(keep_index)] \
                .reset_index(drop=True)
    
    results = calendar[['WORK_DATE']]
    results.columns = ['TIME_UNIT']


    # 資理資料 .....
    if not simplify_date:
        results = df_ymd(df=results, cols='TIME_UNIT')  
    
    if unit == 'm':
        if simplify_format:
            results['TIME_UNIT_SHORT'] = results['TIME_UNIT_SHORT'].astype(str)
            results['TIME_UNIT_SHORT'] = results['TIME_UNIT_SHORT'] \
                                            .str.slice(stop=6)            
    else:
        results['TIME_UNIT_SHORT'] = results['TIME_UNIT']    
    
    # 排序
    # results = results \
    #             .sort_values(by='TIME_UNIT', ascending=ascending) \
    #             .reset_index(drop=True)
        
    if rename == True:
        results = results.rename(columns={'TIME_UNIT':'WORK_DATE'})

    return results



# def date_get_seq_20210729(begin_date, end_date=None, seq_length=None, interval=1, 
#                  unit='d', simplify_date=False,
#                  simplify_format=False, rename=True, 
#                  ascending=False):
#     '''
    

#     Parameters
#     ----------
#     begin_date : TYPE
#         DESCRIPTION.
#     end_date : TYPE, optional
#         DESCRIPTION. The default is None.
#     seq_length : int
#         每一段時間區間的長度，如果有填end_date就不需要填
#     interval : int, optional
#         回傳的每個日期之間的間隔，可為負值
#     unit : TYPE, optional
#         DESCRIPTION. The default is 'd'.
#     simplify_date : TYPE, optional
#         DESCRIPTION. The default is False.
#     simplify_format : TYPE, optional
#         DESCRIPTION. The default is False.
#     rename : TYPE, optional
#         DESCRIPTION. The default is True.
#     ascending : TYPE, optional
#         DESCRIPTION. The default is False.

#     Returns
#     -------
#     results : TYPE
#         DESCRIPTION.

    
#     在一般情況下，只有by day時會用到skip這個參數，而且在其他單位時，也比較容易先算
#     出完整seq後再用%排除，因此目前skip只會在by day時使用。
#     '''
    
#     begin_date = date_simplify(begin_date)
    
#     if end_date == None:
#         end_date = date_cal(begin_date, -seq_length, 'd')
        
#     elif end_date != None and seq_length == None:
#         seq_length = date_diff(begin_date, end_date) + 1
        
    
#     # 讀取日曆 ......
#     calendar = get_rt_calendar(begin_date=begin_date, end_date=end_date,
#                                simplify=True)
    
#     if unit == 'm':
#         calendar = calendar[['YEAR', 'MONTH', 'WORK_DATE']] \
#                     .drop_duplicates(subset=['YEAR', 'MONTH'])
                    
#     elif unit == 'w':
#         calendar = calendar[['YEAR', 'WEEK_NUM', 'WORK_DATE']] \
#                     .drop_duplicates(subset=['YEAR', 'WEEK_NUM'])
    
#     # 排除不必要的資料列 ......
#     calendar = calendar.reset_index(drop=True)
#     calendar.loc[0, 'BEGIN'] = True
#     keep_index = [0 + i * interval for i in range(seq_length)]
    
#     calendar = calendar.reset_index(drop=True).reset_index()
#     calendar = calendar[calendar['index'].isin(keep_index)]
    
#     results = calendar[['WORK_DATE']]
#     results.columns = ['TIME_UNIT']

#     # 資理資料 .....
#     if not simplify_date:
#         results = df_ymd(df=results, cols='TIME_UNIT')  
    
#     if unit == 'm':
#         if simplify_format:
#             results['TIME_UNIT_SHORT'] = results['TIME_UNIT_SHORT'].astype(str)
#             results['TIME_UNIT_SHORT'] = results['TIME_UNIT_SHORT'] \
#                                             .str.slice(stop=6)            
#     else:
#         results['TIME_UNIT_SHORT'] = results['TIME_UNIT']    
    
#     # 排序
#     results = results \
#                 .sort_values(by='TIME_UNIT', ascending=ascending) \
#                 .reset_index(drop=True)
        
#     if rename == True:
#         results = results.rename(columns={'TIME_UNIT':'WORK_DATE'})

#     return results



# ....................
    

def date_get_year_month_seq(year_num, month_num, periods, ascending=False):
    '''
    產生年/月的list

    Parameters
    ----------
    year_num : TYPE
        DESCRIPTION.
    month_num : TYPE
        DESCRIPTION.
    periods : TYPE
        DESCRIPTION.
    ascending : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    '''

    calendar = get_rt_calendar()
    calendar = calendar[['YEAR', 'MONTH']] \
                .drop_duplicates() \
                .reset_index(drop=True)
                
                
    if year_num == None or month_num == None:
        now = datetime.datetime.now()
        year_num = now.year
        month_num = now.month
                
        
    index_begin = calendar[(calendar['YEAR']==year_num) \
                & (calendar['MONTH']==month_num)].index

    index_begin = index_begin[0]    
    index_end = index_begin + periods
        
    
    if index_begin < index_end:
        index_list = list(range(index_begin, index_end))
    else:
        index_list = list(range(index_end + 1, index_begin + 1))
    
    
    calendar = calendar[calendar.index.isin(index_list)] \
                .sort_values(by=['YEAR', 'MONTH'], ascending=ascending) \
                .reset_index(drop=True)
    
    
    results = []
    for i in range(len(index_list)):
        results.append([calendar.loc[i, 'YEAR'], calendar.loc[i, 'MONTH']])
        
    return results


# .....................


def date_diff(d1, d2, absolute=False):
    
    if isinstance(d1, int) == True \
        or isinstance(d1, np.int64) == True \
        or isinstance(d1, np.int32) == True \
        or isinstance(d1, str) == True:
        d1 = ymd(d1)
        
    if isinstance(d2, int) == True \
        or isinstance(d2, np.int64) == True \
        or isinstance(d2, np.int32) == True \
        or isinstance(d2, str) == True:
        d2 = ymd(d2)
    
    if absolute == True:
        results = abs((d2 - d1).days)
    else:
        results = (d2 - d1).days

    return results


# ....................
    

def get_time_serial(with_time=False, divider='_', to_int=False):
    '''    
    取得當前時間的時間序列。
    '''
    import pytz
    central = pytz.timezone('Asia/Taipei')
    cur = datetime.datetime.now(central)
    
    if with_time == True:
        
        results = str(cur.strftime('%Y')) \
            + str(cur.strftime('%m')) \
            + str(cur.strftime('%d')) \
            + divider \
            + str(cur.strftime('%H')) \
            + str(cur.strftime('%M')) \
            + str(cur.strftime('%S'))          

    else:
        results = str(cur.strftime('%Y')) \
            + str(cur.strftime('%m')) \
            + str(cur.strftime('%d'))
           
            
    if to_int == True:
        results = int(results)

    return results

    

# def get_time_serial(with_time=False, divider='_'):
#     '''    
#     取得當前時間的時間序列。
#     '''
    
#     if with_time == True:
#         results = str(arrow.now().year) + \
#                   str('%02d' % arrow.now().month) + \
#                   str('%02d' % arrow.now().day) + \
#                   divider + \
#                   str('%02d' % arrow.now().hour) + \
#                   str('%02d' % arrow.now().minute)
#     else:
#         results = str(arrow.now().year) + \
#                   str('%02d' % arrow.now().month) + \
#                   str('%02d' % arrow.now().day)
    
#     return results


# ...................
    
def cal_age(date):
    '''
    計算年齡
    
    範例 
    df['WORK_DATE'].apply(cal_age)
    '''
    # from datetime import datetime
    import math
    
    try:
        if not isinstance(date, datetime.datetime):
            date = ymd(date)
            
        age = math.floor((datetime.datetime.today() - date).days / 365)    
    except:
        return np.NaN
    
    return age


# ..................
    

def get_date_from_now(amount=0, unit='d', simplify=True):
    
    now = datetime.datetime.now()
    results = date_cal(now, amount, unit, simplify)
    return results


# ..............
    

def date_get_today(with_time=False, simplify=True, to_datetime=False):

    '''
    取得今日日期，並指定為台北時區
    '''

    import pytz
    central = pytz.timezone('Asia/Taipei')
    
    if with_time == True:
        now = datetime.datetime.now(central)
    else:
        now = datetime.datetime.now(central).date()

    
    if simplify == True:
        now = date_simplify(now)
        
    if to_datetime == True:
        now = datetime.datetime.combine(now, datetime.datetime.min.time())
    
    return now


# .................
    

def get_time_seq_list(obj, periods=-3, unit='m', ascending=False):
    '''
    通常搭配SQL使用
    '''
    
    obj = ymd(obj)
    obj = obj.strftime("%Y-%m-%d")
    obj_series = pd.Series(obj)
    
    obj2 = get_time_seq(begin_date=obj, periods=periods, unit=unit, 
                 simplify_date=False, simplify_format=False)
    
    # 合併資料 .....
    results_pre =  obj2['TIME_UNIT'].append(obj_series) \
                .reset_index(drop=True)
                
    results = pd.DataFrame(data={'WORK_DATE':results_pre})
    results['WORK_DATE'] = results['WORK_DATE'].apply(ymd)
    
    results['YEAR_NUM'] = results['WORK_DATE'].dt.year
    results['MONTH_NUM'] = results['WORK_DATE'].dt.month
    
    results = results \
                    .sort_values(by='WORK_DATE', ascending=ascending)
        
        
    results_list = results.values.tolist()
    
    return results_list


# .............
    

def date_gat_calendar(begin_date=20190101, end_date=20211231, 
                      sunday_leading=True):
    '''
    列出每一個日期的相關資訊，這樣當資料表中的WORK_DATE已經被simplify過時，
    不需要再轉換成日期，可以直接join。
    '''
    date_seq = get_time_seq(begin_date=begin_date, 
                                 end_date=end_date,
                                 periods=None, unit='d', 
                                 simplify_date=True,
                                 skip=0, simplify_format=False, 
                                 rename=True)
    
    date_seq['WORK_DATE_TEMP'] = date_seq['WORK_DATE'].apply(ymd)
    date_seq['YEAR'] = date_seq['WORK_DATE_TEMP'].dt.year
    date_seq['MONTH'] = date_seq['WORK_DATE_TEMP'].dt.month
    date_seq['WEEKDAY'] = date_seq['WORK_DATE_TEMP'].dt.weekday
    date_seq['WEEK_NUM'] = date_seq['WORK_DATE_TEMP'].dt.week
    
    
    if sunday_leading == True:
        date_seq.loc[date_seq['WEEKDAY']==6, 'WEEK_NUM'] = \
            date_seq['WEEK_NUM'] + 1
    
    date_seq = date_seq[['WORK_DATE', 'YEAR', 'MONTH',
                         'WEEK_NUM', 'WEEKDAY']]
    
    return date_seq


# .................
    

def date_get_period(predict_begin, predict_end=None, predict_period=None,
                    data_begin=None, data_end=None, data_period=None,
                    shift=None):
    '''
    1. 取得資料期間，以及預測區間
    2. by day
    3. predict_begin為必填
    
    

    Parameters
    ----------
    predict_begin : TYPE, optional
        DESCRIPTION. The default is None.
    predict_end : TYPE, optional
        DESCRIPTION. The default is None.
    predict_period : TYPE, optional
        DESCRIPTION. The default is None.
    data_begin : TYPE, optional
        DESCRIPTION. The default is None.
    data_end : TYPE, optional
        DESCRIPTION. The default is None.
    data_period : TYPE, optional
        DESCRIPTION. The default is None.
    shift : int, default None
        資料處理階段，有時候會計算lag，為了避免前面的資料lag值為na，因此抓資料的時候
        再往前推N天。預設為三倍的predict_period。


    Returns
    -------
    shift_begin : TYPE
        DESCRIPTION.
    shift_end : TYPE
        DESCRIPTION.
    data_begin : TYPE
        DESCRIPTION.
    data_end : TYPE
        DESCRIPTION.
    predict_begin : TYPE
        DESCRIPTION.
    predict_end : TYPE
        DESCRIPTION.

    '''
    
    # Begin
    if data_begin == None and data_period != None and data_end != None:
        data_begin = date_cal(data_end, amount=-1*data_period, unit='d')
    
    elif data_begin == None and data_period != None and data_end == None \
        and predict_begin != None:
        data_end = date_cal(predict_begin, amount=-1, unit='d')
        data_begin = date_cal(data_end, amount=-1*data_period, unit='d')
    
    elif data_end == None and data_period != None:
        data_end = date_cal(data_begin, amount=data_period, unit='d')
    
    
    # Forecast periods
    predict_begin = date_cal(data_end, amount=1, unit='d')
    
    if predict_end == None and predict_period != None:
        predict_end = date_cal(predict_begin,
                                amount=predict_period,
                                unit='d')
    
    # Shift
    if shift == None:
        shift = -3 * predict_period
    
    shift_begin = date_cal(data_begin, shift, 'd')
    shift_end = date_cal(data_begin, -1, 'd')
    
    return shift_begin, shift_end, \
            data_begin, data_end, predict_begin, predict_end


# ...............
    
    
def date_split(begin_date, end_date, chunk, unit='m', simplify=False, 
               full=False):
    
    calendar_raw = get_rt_calendar(begin_date=begin_date, end_date=end_date, 
                               simplify=True)
    
    
    if unit == 'd':
        calendar_pre = calendar_raw.reset_index()
        calendar= calendar_pre[(calendar_pre['index']%chunk==0) \
                            | (calendar_pre['index']%chunk==chunk-1)] 
        
        # 當begin_date和end_date實際差距不足一個chunk的時候，將end_date保留，否則往下
        # 會找不到END_DATE
        if len(calendar) == 1:
            temp = calendar_pre.iloc[-1, :]
            calendar = calendar.append(temp)
            
            
    elif unit == 'm':
        calendar = calendar_raw[(calendar_raw['MONTH_BEGIN']==True) \
                            | (calendar_raw['MONTH_END']==True)] \
                    .reset_index(drop=True) \
                    .reset_index()
                    
        calendar = calendar_pre[(calendar_pre['index']%(chunk*2)==0) \
                            | (calendar_pre['index']%(chunk*2)==chunk*2-1)]
            
      
    results = calendar[['WORK_DATE']].reset_index(drop=True)
    results['TYPE'] = np.where(results.index%2==0, 'BEGIN_DATE', 'END_DATE')
    
    results['INDEX'] = results.index / 2
    results['INDEX'] = results['INDEX'].apply(np.floor)

    
    results = results.pivot_table(index='INDEX',
                                  columns='TYPE',
                                  values='WORK_DATE') \
                .reset_index()

    # 這裡的最後一個END_DATE可能會是na，遇到na時改成argument中的end_date
    results = results[['BEGIN_DATE', 'END_DATE']]
    results = df_conv_na(df=results, cols='END_DATE', value=end_date)
    results = df_ymd(df=results, cols=['BEGIN_DATE', 'END_DATE'])
    
    if simplify:
        results = df_date_simplify(df=results, 
                                        cols=['BEGIN_DATE', 'END_DATE'])
    
    return results
    
    

def date_split_v1(begin_date, end_date, cycle_period=14):
    
    
    date_seq = date_get_seq(begin_date=begin_date, end_date=end_date,
                                 periods=None, unit='d', simplify_date=True,
                                 skip=0, simplify_format=False, rename=True)    
    
    date_seq['INDEX'] = date_seq.index / cycle_period
    date_seq['INDEX'] = date_seq['INDEX'] .apply(np.floor)


    date_seq['BEGIN_DATE'] = date_seq \
                            .groupby('INDEX')['WORK_DATE'].transform(min)
                            
    date_seq['END_DATE'] = date_seq \
                            .groupby('INDEX')['WORK_DATE'].transform(max)                            
    
    
    date_seq = date_seq[['INDEX', 'BEGIN_DATE', 'END_DATE']] \
                .drop_duplicates() \
                .reset_index(drop=True)
    
    
    date_seq = df_conv_col_type(df=date_seq, cols='INDEX', to=int)
    
    
    return date_seq


# ...............
    

def date_get_month_begin_end(begin_date=None, end_date=None,
                             begin_month=None, end_month=None, 
                             simplify=True):
    '''
    date和month只需要挑一個填
    '''

    from calendar import monthrange
    
    # 以date計算month
    if begin_date != None and end_date != None:
        begin_date = str(begin_date)
        end_date = str(end_date)

        begin_month = begin_date[0:6]
        end_month = end_date[0:6]
        
    
    begin_date = str(begin_month) + '01'
    
    last_day = monthrange(int(end_month[0:4]), int(end_month[4:6]))  
    last_day = str(last_day[1]).zfill(2)
    end_date = str(end_month) + last_day
    

    # 讀取日曆    
    calendar = get_rt_calendar(begin_date=begin_date, end_date=end_date)
    
    calendar['DAY'] = calendar['WORK_DATE'].dt.day
    
    calendar['MONTH_MIN'] = calendar \
                    .groupby(['YEAR', 'MONTH'])['DAY'] \
                    .transform(min)

    calendar['MONTH_MAX'] = calendar \
                    .groupby(['YEAR', 'MONTH'])['DAY'] \
                    .transform(max)
                    
    calendar.loc[calendar['DAY']==calendar['MONTH_MIN'], 'TYPE'] = 'BEGIN'
    calendar.loc[calendar['DAY']==calendar['MONTH_MAX'], 'TYPE'] = 'END'
    
    calendar = calendar[~calendar['TYPE'].isna()]
    calendar = calendar[['YEAR', 'MONTH', 'TYPE', 'WORK_DATE']]


    begin = calendar[calendar['TYPE']=='BEGIN'] \
            .drop('TYPE', axis=1) \
            .rename(columns={'WORK_DATE':'BEGIN'})
    
    
    end = calendar[calendar['TYPE']=='END'] \
            .drop('TYPE', axis=1) \
            .rename(columns={'WORK_DATE':'END'})
    

    results = begin.merge(end, on=['YEAR', 'MONTH']) \
                .reset_index(drop=True)
    
    
    if simplify == True:
        results = df_date_simplify(df=results, cols=['BEGIN', 'END'])
    
    
    return results




# %% Plotly系列 ----------------
    

def plotly(df, x, y, groupby='', title="", xaxes="", yaxes="", mode=2):


    from plotly.offline import plot 
    import plotly.graph_objs as go
    
    
    fig = go.Figure() 

    
    # 確認模式
    if mode == 0 or mode == 'markers':
        mode = 'markers'
    elif mode ==1 or mode == 'lines+markers':
        mode = 'lines+markers'
    elif mode == 2 or mode == 'lines':
        mode = 'lines'        
    
    
    
    if groupby == '':
    
        fig.add_trace(go.Scatter(name='NAME',
                                  x=df[x],  
                                  y=df[y],
                                  mode=mode))
        
        fig.update_layout(title_text=title)
        fig.update_xaxes(title_text=xaxes)
        fig.update_yaxes(title_text=yaxes)    

        plot(fig)           
        return ''
    
    
    groupby = conv_to_list(groupby)
    
    
    unique_key = df[groupby].drop_duplicates() \
        .reset_index(drop=True)
        
    
    unique_key['KEY'] = True
    
    agg_key = li_join_flatten(groupby, x)

    
    for i in range(0, len(unique_key)):
        
        temp_key = unique_key[unique_key.index==i]
        temp = df.merge(temp_key, how='left', on=groupby)
        
        temp = temp[temp['KEY']==True]
        
        
        temp = temp.groupby(agg_key) \
            .aggregate({y:'sum'}) \
                .reset_index()
        
        
        # Update，這段寫法有點笨，在上面就先合併所有的欄位。
        legend_name = ''
        for j in range(0, len(groupby)):
            
            if j == 0:
                legend_name = str(temp_key.loc[i, groupby[j]])
            else:
                legend_name = legend_name + '_' \
                                + str(temp_key.loc[i, groupby[j]])
                                 
                                
        fig.add_trace(go.Scatter(name=legend_name,
                                 x=temp[x],  
                                 y=temp[y],
                                 mode=mode))
    
    
    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text=xaxes)
    fig.update_yaxes(title_text=yaxes)    

    plot(fig)    
    
    return ''



# def plotly_v0(df, x, y, title="", xaxes="", yaxes=""):
#     '''
#     20201228移除
#     '''
#     from plotly.offline import plot 
#     import plotly.graph_objs as go
    
#     fig = go.Figure() 
    
#     fig.add_trace(go.Scatter(name='NAME',
#                              x=df[x],  
#                              y=df[y]))
    
#     fig.update_layout(title_text=title)
#     fig.update_xaxes(title_text=xaxes)
#     fig.update_yaxes(title_text=yaxes)    

#     plot(fig)    
    
#     return ''




# %% Matplotlib系列 -----------------
    

def plt_load_chinese_font(plt):
    '''
    預設的字型無法顯示中文。
    '''
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    
    return ''



def plt_settings():
    
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    pylab.rcParams.update(params)
    return ''
        

# %% Geocoding系列 ------------
    

def geocoding_get_member_addr(include_stores=[], exclude_stores=[4,5,8,19], 
                        status=[0,1,5,8,9]):
    '''
    取得拆開欄位的會員地址。
    Update, 把註冊店變成參數
    '''
    
    # 取得店號
    loc_store_list = cbrt.get_store_list(exclude_stores=exclude_stores,
                                    include_stores=include_stores)
    
    status_sql = li_join_as_str(status)
    status_sql = ' and status in (' + status_sql + ') '
        
    
    # 讀取資料 ...    
    sql = (" select store_no, client_no, status, tb1.post_no, " 
            " descr, area_descr, sub_locality, street,  " 
            " addr_num, addr_floor, address" 
            " from  " 
            " ( " 
            "   select store_no, client_no, status, post_no, " 
            "   sub_locality, street, address_hao addr_num, " 
            "   address_lo addr_floor, address " 
            "   from store_clients@csm " 
            "   where client_no != 999999 " 
            "   " + status_sql + " "
            "   and store_no " + loc_store_list['STORE_NOT_IN_SQL'] + " "
            " ) tb1, " 
            " ( " 
            "   select post_no, descr, area_descr " 
            "   from post_area@csm " 
            " ) tb2 " 
            " where tb1.post_no = tb2.post_no ")
        
    data_raw = t.query_sql(sql)
    
    data = data_raw.copy()
    data['ADDRESS'] = data['DESCR'] + data['ADDRESS']
    data = df_str_col_subtract(data, col1='DESCR', col2='AREA_DESCR') 
    

    # None會導致合併string的時候出錯，因此先轉成-1
    data = df_conv_na(data, cols=['STREET', 'SUB_LOCALITY'], value='-1')


    data['GEOCODING_ADDR'] = data['STREET'] + ', ' \
                            + data['SUB_LOCALITY'] + ', ' \
                            + data['AREA_DESCR'] + ', ' \
                            + data['DESCR'] + ', ' \
                            + '台灣'
        
    data['GEOCODING_ADDR'] = data['GEOCODING_ADDR'] \
                            .str.replace('-1, ', '')
                                    
    data = data[~data['GEOCODING_ADDR'].isna()] \
                    .reset_index(drop=True)    
        
    return data


# ..............
    

def geocoding_member_coord(df=None, include_stores=[], 
                             exclude_stores=[4,5,8,19], mode=0, test=False):
    '''
    Update, mode的1、2還沒寫完
    df       : DataFrame. 必須包含geo_get_member_addr中有的欄位
    mode     : int. 0為排除所有已有經緯度的會員；
                    1為排除有經緯度，且地址相同的會員；
                    2為不排除任何會員
    test     : boolean. 測試模式時只會回傳要轉換的會員數量。
    '''
    
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="example app")            
    
    
    # 讀取會員地址 ......
    if not isinstance(df, pd.DataFrame):
        member_addr_raw = geocoding_get_member_addr(exclude_stores= exclude_stores,
                                              include_stores=include_stores)
    else:
        member_addr_raw = df
    
    
    # 以mode篩選 .....
    coord_sql = (" select store_no, client_no, address address_org "
                 " from rdba.member_coordinate")   
    
    coord_data = t.query_sql(coord_sql)
    
    
    if mode == 0:
        coord_member = coord_data[['STORE_NO', 'CLIENT_NO']]
        coord_member['IN_LIST'] = 1
        
        member_addr = member_addr_raw.merge(coord_member, how='left',
                                            on=['STORE_NO', 'CLIENT_NO'])
        
        member_addr = member_addr[member_addr['IN_LIST'].isna()] \
                            .drop('IN_LIST', axis=1) \
                            .reset_index(drop=True)
                    
        print('geocoding_member_address 排除已有資料的會員')
        
    elif mode == 1:
        return 'mode 1 還沒寫完'
        
    elif mode == 2:
        return 'mode 2 還沒寫完'
    
    # 測試模式 ......
    if test == True:
        test_str = 'geocoding_member_address為測試模式，共有' \
                    + str(len(member_addr)) + '筆資料要進行轉換'
        print(test_str)                    
        return test_str

    
    # 呼叫OpenStreetMap的API ......
    geo_results = pd.DataFrame()
    fail_results = pd.DataFrame()
    
    # for i in range(0, 100):
    for i in range(0, len(member_addr)):
        temp_addr = member_addr.loc[i, 'GEOCODING_ADDR']
        
        # 顯示進度
        if i % 100 == 0:
            print(str(i) + '/' + str(len(member_addr)))
        
        # Call OpenStreeMap API
        try:
            new_data = geolocator.geocode(temp_addr).raw
            
            # 刪除不需要的欄位，必須之換的時候
            del new_data['licence']
            del new_data['osm_type']
            del new_data['osm_id']
            del new_data['boundingbox']
            del new_data['display_name']
            del new_data['class']
            del new_data['type']
            del new_data['importance']
                
            new_df = pd.DataFrame({'STORE_NO':[member_addr.loc[i, 'STORE_NO']],
                                   'CLIENT_NO':[member_addr.loc[i, 'CLIENT_NO']],
                                   'LAT':[new_data['lat']],
                                   'LON':[new_data['lon']]})

            geo_results = geo_results.append(new_df)
        except:
            new_df = pd.DataFrame({'STORE_NO':[member_addr.loc[i, 'STORE_NO']],
                                   'CLIENT_NO':[member_addr.loc[i, 'CLIENT_NO']],
                                   'LAT':['NA'],
                                   'LON':['NA']})

            fail_results = fail_results.append(new_df)            
            continue
    
    print('Geocoding轉換完畢')
    
    # 資料整理 ......
    # 轉換失敗的會員也要寫進資料庫，不然每次都會嘗試轉換這些人，卻又沒有結果。    
    geo_results_df = geo_results.append(fail_results) \
                    .drop_duplicates() \
                    .reset_index(drop=True)

    
    # 避免錯誤
    if len(geo_results_df) == 0:
        print('geocoding_member_address 成功轉換的筆數為0')
        return 'geocoding_member_address 成功轉換的筆數為0'


    # 將欄位名稱變成大寫                    
    geo_results_df.columns = li_upper(list(geo_results_df.columns))

    return geo_results_df


# ..................
    

def cal_dist_by_coord(lon1, lat1, lon2, lat2, digits=2):
    '''
    以經緯度計算直線距離。
    origin        : list. 經緯度
    destination   : list. 經緯度
    '''
    import math
    # lon1, lat1 = start
    # lon2, lat2 = end
    radius = 6371 # km
    
    
    # lat1 = float(lat1)
    # lon1 = float(lon1)
    # lat2 = float(lat2)
    # lon2 = float(lon2)
    

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    dist = radius * c
    dist = round(dist, digits)

    return dist


def list_cal_dist_by_coord(li, digits=2):
    '''
    以經緯度計算直線距離。
    origin        : list. 經緯度
    destination   : list. 經緯度
    '''
    import math
    lon1, lat1, lon2, lat2 = li
    radius = 6371 # km
    
    
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    dist = radius * c
    dist = round(dist, digits)

    return dist


# ................




def df_cal_dist_by_coord(df, start_lon, start_lat, end_lon, end_lat, 
                         digits=2):
    '''
    以經緯度計算直線距離。
    start_col : list. 經緯度
    end_col   : list. 經緯度
    '''
    

    import math
    radius = 6371 # km


    df.loc[:, 'DLON'] = df[end_lon] - df[start_lon]
    df['DLON'] = df['DLON'].apply(math.radians)
     
    df.loc[:, 'DLAT'] = df[end_lat] - df[start_lat]
    df['DLAT'] = df['DLAT'].apply(math.radians)    


    df['A'] = (df['DLAT'] / 2).apply(math.sin) \
                * (df['DLAT'] / 2).apply(math.sin) \
                + df[start_lat].apply(math.radians).apply(math.cos) \
                * df[end_lat].apply(math.radians).apply(math.cos) \
                * (df['DLON'] / 2).apply(math.sin) \
                * (df['DLON'] / 2).apply(math.sin) 
        
    df['C'] = 2 * np.arctan(df['A'].apply(math.sqrt) \
                            / (1 - df['A']).apply(math.sqrt))
        
    df['DIST'] = radius * df['C']
    df.round({'DIST': digits})


    df = df.drop(['DLON', 'DLAT', 'A', 'C'], axis=1)
    
    # a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
    #     * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    # c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # dist = radius * c
    # dist = round(dist, digits)    
    
    
    return df





# def df_cal_dist_by_coord_backup20210406(df, start_col, end_col, digits=2):
#     '''
#     以經緯度計算直線距離。
#     start_col : list. 經緯度
#     end_col   : list. 經緯度
#     '''
    
#     df['TEMP_COL'] = df[start_col] + df[end_col]
#     df['DIST'] = df['TEMP_COL'].apply(list_cal_dist_by_coord, digits)
#     df = df.drop('TEMP_COL', axis=1)
            
#     return df


# ...............
    

def get_member_dist(exclude_stores=[4,5,8,19], include_stores=[]):
    '''
    計算全部會員與某間店之間的距離
    (1) Update，匯入會員df，只計算特定會員

    Parameters
    ----------
    exclude_stores : TYPE, optional
        DESCRIPTION. The default is [4,5,8,19]. 排除的分店，但不直接排除會員
    include_stores : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    member_coord : TYPE
        DESCRIPTION.

    '''

    # 取得分店經緯度
    store_coord = cbrt.get_store_coord(exclude_stores=exclude_stores, 
                                       include_stores=include_stores)
    
    store_coord = store_coord[['STORE_NO', 'LONGITUDE', 'LATITUDE']]
    store_coord.columns = ['DIST_STORE_NO', 'STORE_LON', 'STORE_LAT']
    
    
    # 會員經緯度 
    member_coord_sql = (" select distinct store_no, client_no, lon, lat "
                        " from rdba.member_coordinate"
                        " where lat != 'NA' ")
    
    member_coord = t.query_sql(member_coord_sql)
    member_coord = df_conv_col_type(df=member_coord, cols=['LON', 'LAT'],
                                    to=float)
    
    member_coord = member_coord.rename(columns={'LON':'MEMBER_LON',
                                                'LAT':'MEMBER_LAT'})
    
    # member_coord['MEMBER_COORD'] = member_coord[['LON', 'LAT']] \
    #                                 .values.tolist()    
    
    # member_coord = member_coord.drop(['LON', 'LAT'], axis=1)
    

    # 主要計算區
    member_coord = df_cross_join(member_coord, store_coord)
            
    member_coord = df_cal_dist_by_coord(df=member_coord,
                                        start_lon='STORE_LON', 
                                        start_lat='STORE_LAT',
                                        end_lon='MEMBER_LON',
                                        end_lat='MEMBER_LAT')

    return member_coord


# ...............
    

def get_catchment_area(exclude_stores=[4,5,8,19], include_stores=[], 
                       remove_dup=True, remove_area_e=False,
                       export_file=True, load_file=False, path=None,
                       file_name='get_catchment_area'):

    # 讀取暫存檔
    if path != None and load_file:
        try:
            results = pd.read_csv(path + '/' + file_name + '.csv')
            print('讀取暫存檔 ' + path + '/' + file_name + '.csv')
            return results
        except:
            print('get_catchment_area無法讀取暫存檔')

    
    # 讀取資料 ......
    dist_data = get_member_dist(exclude_stores=exclude_stores,
                                include_stores=include_stores)    
    
    dist_data = dist_data[['STORE_NO', 'CLIENT_NO', 'DIST_STORE_NO', 'DIST']]
    
    
    # 只保留距離最近的店 ......
    if remove_dup == True:
        dist_data['MIN_DIST'] = dist_data \
                                .groupby(['STORE_NO', 'CLIENT_NO'])['DIST'] \
                                .transform(min)
        
        dist_data = dist_data[dist_data['MIN_DIST']==dist_data['DIST']] \
                    .drop('MIN_DIST', axis=1) \
                    .reset_index(drop=True)
    
    
    # 排除商圈E，即>=20
    if remove_area_e:
        dist_data = dist_data[dist_data['DIST']<20].reset_index(drop=True)
    
    
    dist_data = df_add_level(df=dist_data, start=0, stop=20, step=5,
                             col='DIST', col_name='LEVEL', unit='')
    
    dist_data['AREA'] = np.select([dist_data['LEVEL']=='00-05',
                                   dist_data['LEVEL']=='05-10',
                                   dist_data['LEVEL']=='10-15',
                                   dist_data['LEVEL']=='15-20',
                                   dist_data['LEVEL']=='20-'],
                                  ['A', 'B', 'C', 'D', 'E'])

    dist_data = dist_data \
                .reset_index(drop=True) \
                .rename(columns={'DIST_STORE_NO':'NEAREST_STOER'})
             
    dist_data = dist_data[['STORE_NO', 'CLIENT_NO', 'NEAREST_STOER', 
                           'DIST', 'AREA']]   

    # 儲存暫存檔 ......
    if export_file and path != None:
        try:
            dist_data.to_csv(path + '/' + file_name + '.csv', 
                             index=False, encoding='utf-8')
        except:
            print('get_catchment_area無法儲存檔案')
                
    return dist_data


# %% ML系列 ----------------


def df_get_dummies(df, cols, expand_col_name=True, inplace=False):
    
    if inplace:
        loc_df = df
    else:
        loc_df = df.copy()
    
    # ......
    cols = conv_to_list(cols)
    
    for i in range(len(cols)):
        col = cols[i]
        dummy_df = pd.get_dummies(loc_df[col])
        
        # 把原本的變數名稱與數值合併
        if expand_col_name:
            new_cols = list(dummy_df.columns)
            new_cols = [col + '_' + str(i) for i in new_cols]
            dummy_df.columns = new_cols
        
        loc_df = loc_df.drop(col, axis=1)
        loc_df = pd.concat([loc_df, dummy_df], axis=1)

    return loc_df


# ..............


def df_normalize(df, cols, groupby=[], method=0, show_progress=False):
    '''
    Update, add new method

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    groupby : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    RESULTS : TYPE
        DESCRIPTION.
    
    ORIGINAL : TYPE
        DESCRIPTION.
    
    GROUPBY : TYPE
        DESCRIPTION.        

    '''
    cols = conv_to_list(cols)
    groupby = conv_to_list(groupby)
    
    # 取得Groupby的欄位
    if len(groupby) > 0:
        groupby = conv_to_list(groupby)
        groups = df[groupby] \
                .drop_duplicates() \
                .reset_index(drop=True) \
                .reset_index() \
                .rename(columns={'index':'GROUP_ID'})
                
        df = df.merge(groups, how='left', on=groupby)       
         
    else:
        groups = pd.DataFrame({'GROUP_ID':[0]})
        df.loc[df.index, 'GROUP_ID'] = 0
        df.loc[df.index, 'TEMP_KEY'] = 0
        groupby = ['TEMP_KEY']
        
    orig_key = li_join_flatten(groupby, ['GROUP_ID', 'NORM_MIN', 'NORM_MAX'])
    
    
    # 主工作區 ......
    original = pd.DataFrame()


    # Get min and max of each columns. ......
    for i in range(0, len(cols)):    

        col = cols[i]
        temp_df = df[groupby + ['GROUP_ID', col]]
        
        temp_df.loc[temp_df.index, 'NORM_MIN'] = \
            temp_df.groupby('GROUP_ID')[col].transform('min')
            
        temp_df.loc[temp_df.index, 'NORM_MAX'] = \
            temp_df.groupby('GROUP_ID')[col].transform('max')


        # Normalize ......
        # If col_min is equal to col_max, then the results of
        # normalization will be na
        if method == 0:
            temp_df.loc[temp_df.index, 'NORM_VALUES'] = \
                np.where(temp_df['NORM_MIN']==temp_df['NORM_MAX'],
                         temp_df['NORM_MIN'],
                         (temp_df[col] - temp_df['NORM_MIN']) \
                             / (temp_df['NORM_MAX'] - temp_df['NORM_MIN']))
        
        df.loc[df.index, col] = temp_df['NORM_VALUES']
        
        
        # 保留原始值 ......
        original_new = temp_df[orig_key].drop_duplicates()
        original_new.loc[original_new.index, 'COLUMN'] = col
        original = original.append(original_new)
        
        print('df_normalize - ' + str(i) + '/' + str(len(cols) - 1))
        
        
    # 整料整理 ......
    original = original.reset_index(drop=True)  
    df = df.drop('GROUP_ID', axis=1).reset_index(drop=True)
    
    if groupby[0] == 'TEMP_KEY':
        groupby = []
        df = df.drop('TEMP_KEY', axis=1)
        
    return df, original, groupby, method
    

# .............
    

def df_normalize_restore(df, original, groupby=[], method=0):
    
    cols = list(df.columns)
    groupby = conv_to_list(groupby)
    execute_times = 0
    
    # ......
    if len(groupby) == 0:
        groupby = ['TEMP_KEY']
        df['TEMP_KEY'] = 0
    
    
    # 還原數值 ......
    orig_cols = original['COLUMN'].unique().tolist()
    
    for i in range(0, len(orig_cols)):
        col = orig_cols[i]

        if col not in cols:
            continue            
        
        temp_original = original[original['COLUMN']==col]
        df = df.merge(temp_original, how='left', on=groupby)
        
        if method == 0:
            df[col] = df[col] \
                        * (df['NORM_MAX'] - df['NORM_MIN']) \
                        + df['NORM_MIN']
        
        df = df.drop(['GROUP_ID', 'NORM_MIN', 'NORM_MAX', 'COLUMN'], axis=1)
        execute_times = execute_times + 1        


    if execute_times == 0:
        print('df_normalize_restore - 執行次數為0')    
        
        
    if groupby[0] == 'TEMP_KEY':
        df = df.drop('TEMP_KEY', axis=1)

    return df    


# .............


def ml_bulk_run_clf_model(X_train, y_train, X_test, y_test,
                          logreg=True, rf=True, knn=True, svm=True, gnb=True,
                          xgb=True):
    
    '''
    https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0
    '''

    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    from sklearn import model_selection
    from sklearn.utils import class_weight
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pddef     
    
    dfs = []
    models = []    
    
    if logreg:
        models.append(('LogReg', LogisticRegression()))
        
    if rf:
        models.append(('RF', RandomForestClassifier()))
    
    if knn:
        models.append(('KNN', KNeighborsClassifier()))
        
    if svm:
        models.append(('SVM', SVC()))

    if gnb:
        models.append(('GNB', GaussianNB()))

    if xgb:
        models.append(('XGB', XGBClassifier()))
    
    
    
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 
               'f1_weighted', 'roc_auc']
    
    target_names = ['malignant', 'benign']
    
    
    
    for name, model in models:
        
        kfold = model_selection.KFold(n_splits=4, shuffle=True, 
                                      random_state=90210)
        
        cv_results = model_selection.cross_validate(model, X_train, y_train, 
                                                    cv=kfold, scoring=scoring)
        
        model_fit = model.fit(X_train, y_train)
        y_pred = model_fit.predict(X_test)
        
        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))
    
        
        results.append(cv_results)
        names.append(name)
        
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)

    return final




def ml_bulk_run_reg_model(X_train, y_train, X_test, y_test, rf=True, kr=True,
                          lreg=True, svr=True, dt=True, sgd=True, knn=True,
                          xgb=True):
    
    '''
    
    Total List
    https://scikit-learn.org/stable/supervised_learning.html
    
    https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    
    '''

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from xgboost import XGBClassifier
    
    from sklearn import model_selection
    from sklearn.utils import class_weight
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pddef     
    
    dfs = []
    models = []    
    
    if rf:
        models.append(('RF', RandomForestRegressor()))
        
    if kr:
        models.append(('KR', KernelRidge()))
        
    if lreg:
        models.append(('LogReg', LinearRegression()))
        
    if svr:
        models.append(('SVR', SVR(C=1.0, epsilon=0.2)))
        
    if dt:
        models.append(('DT', DecisionTreeRegressor()))
        
    if sgd:
        models.append(('SGD', SGDRegressor()))

    if knn:
        models.append(('KNN', KNeighborsRegressor()))
        
    if xgb:
        models.append(('XGB', XGBClassifier()))
    
    
    results = []
    names = []
    scoring = ['r2']

    # target_names = ['malignant', 'benign']
    
    for name, model in models:
        
        kfold = model_selection.KFold(n_splits=4, shuffle=True, 
                                      random_state=90210)


        cv_results = model_selection.cross_validate(model, X_train, y_train, 
                                                    cv=kfold, scoring=scoring)        
        
        model_fit = model.fit(X_train, y_train)
        y_pred = model_fit.predict(X_test)
        # print(name)
        # print(classification_report(y_test, y_pred, target_names=target_names))
    
        
        results.append(cv_results)
        names.append(name)
        
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
    
    final = pd.concat(dfs, ignore_index=True)
    
    return final


def ml_bulk_run_clus_model(X_train, y_train, X_test, y_test, km=True,
                           ap=True):
    
    '''
    
    Total List
    https://scikit-learn.org/stable/supervised_learning.html
    
    https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    
    '''


    from sklearn.cluster import KMeans
    from sklearn.cluster import AffinityPropagation
    # from sklearn.cluster import OPTICS
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.kernel_ridge import KernelRidge
    # from sklearn.linear_model import LinearRegression
    # from sklearn.svm import SVR
    # from sklearn.tree import DecisionTreeRegressor
    # from sklearn.linear_model import SGDRegressor
    # from sklearn.neighbors import KNeighborsRegressor
    # from xgboost import XGBClassifier
    
    from sklearn import model_selection
    from sklearn.utils import class_weight
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pddef     
    
    dfs = []
    models = []    
    
    if km:
        models.append(('KM', KMeans()))
    
    if ap:
        models.append(('AP', AffinityPropagation()))
        
    
    
    results = []
    names = []
    scoring = ['v_measure_score']

    # target_names = ['malignant', 'benign']
    
    for name, model in models:
        
        kfold = model_selection.KFold(n_splits=4, shuffle=True, 
                                      random_state=90210)


        cv_results = model_selection.cross_validate(model, X_train, y_train, 
                                                    cv=kfold, scoring=scoring)        

        model_fit = model.fit(X_train, y_train)
        y_pred = model_fit.predict(X_test)

        
        # print(name)
        # print(classification_report(y_test, y_pred, target_names=target_names))
    
        
        results.append(cv_results)
        names.append(name)
        
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
    
    final = pd.concat(dfs, ignore_index=True)
    
    return final



def ml_conv_to_nparray(obj, reshape=True):
    '''
    將list轉成np.array
    reshape : boolean. True時會將只有一個feature的array reshape，否則丟進模型中
              會很容易出錯。
    '''
    
    # if isinstance(obj, pd.Series):
    #     arr = obj.to_numpy()
    #     arr = arr.reshape(-1, 1)
        
    # elif isinstance(obj, pd.Series):
    #     arr = np.array(obj) 
    
    #     if not isinstance(obj[0], list):
    #         arr = arr.reshape(-1, 1)


    if isinstance(obj, pd.Series):
        arr = obj.to_numpy()
        arr = arr.reshape(-1, 1)
        
    elif isinstance(obj, list):
        arr = np.array(obj)
        arr = arr.reshape(-1, 1)
    else:
        arr = ''


    return arr


# ............
    

def ml_data_process(df, ma=True, normalize=True, lag=True, 
                    ma_group_by=[], norm_group_by=[], lag_group_by=[], 
                    date_col='WORK_DATE', 
                    ma_cols_contains=[], ma_except_contains=[],
                    norm_cols_contains=[], norm_except_contains=[],
                    lag_cols_contains=[], lag_except_contains=[], 
                    drop_except_contains=[],
                    ma_values=[7,14], lag_period=3):

    # Convert To List ......
    ma_group_by = conv_to_list(ma_group_by)
    norm_group_by = conv_to_list(norm_group_by)
    lag_group_by = conv_to_list(lag_group_by)
    
    # MA
    ma_cols_contains = conv_to_list(ma_cols_contains)
    ma_except_contains = conv_to_list(ma_except_contains)
    
    # Normalize
    norm_cols_contains = conv_to_list(norm_cols_contains)
    norm_except_contains = conv_to_list(norm_except_contains)
    
    # Lag
    lag_cols_contains = conv_to_list(lag_cols_contains)
    lag_except_contains = conv_to_list(lag_except_contains)
    
    drop_except = df_get_cols_contain(df, drop_except_contains)
    
    date_col_list = [date_col]
    drop_except = drop_except + date_col_list
    
    cols = list(df.columns)    
    loc_main = df.copy()
    
    
    # Calculate MA ......
    if ma:
        ma_cols = df_get_cols_contain(loc_main, ma_cols_contains)
        ma_except = df_get_cols_contain(loc_main, ma_except_contains)        
        
        if len(ma_cols) == 0:
            cols = li_remove_items(cols, 
                                   ma_group_by+ma_except+date_col_list)
        else:
            cols = ma_cols
            
            
        # 防呆，WORK_DATE不該被加入ma_group_by中
        ma_group_by = li_remove_items(ma_group_by, ['WORK_DATE'])
            
        loc_main, ma_cols = df_add_ma(df=loc_main, cols=cols, 
                                      group_by=ma_group_by, 
                                      date_col=date_col, values=ma_values,
                                      wma=False)
        
        drop_cols = li_remove_items(cols, drop_except)
        loc_main = loc_main.drop(drop_cols, axis=1)


    # Normalize ......
    # 要把標準化之後的物件回傳，才能執行df_normalize_restore
    if normalize:
        norm_cols = df_get_cols_contain(loc_main, norm_cols_contains)
        norm_except = df_get_cols_contain(loc_main, norm_except_contains)        
        
        if len(norm_cols) == 0:
            norm_cols = df_get_cols_except(df=loc_main, 
                                           except_cols=norm_group_by + norm_except)
        
        loc_main, norm_orig, norm_group, norm_method = \
            df_normalize(df=loc_main, cols=norm_cols,
                         groupby=norm_group_by, show_progress=True)       
    else:
        norm_orig = []
        norm_group = []
        norm_method = ''
          
        
    # Lag ......
    if lag:
        lag_cols = df_get_cols_contain(loc_main, lag_cols_contains)
        lag_except = df_get_cols_contain(loc_main, lag_except_contains)        
        
        if len(lag_cols) == 0:        
            lag_cols = df_get_cols_except(df=loc_main, 
                                          except_cols=lag_group_by+lag_except)
        
        loc_main, _ = df_add_shift(df=loc_main, cols=lag_cols, 
                                        shift=lag_period,
                                        group_by=lag_group_by,
                                        suffix='_LAG', 
                                        remove_na=False)    

        drop_cols = li_remove_items(lag_cols, drop_except)
        loc_main = loc_main.drop(drop_cols, axis=1)

    cols = df_get_cols_except(df=loc_main, 
                              except_cols=ma_group_by+norm_group_by \
                                          +lag_group_by+date_col_list)
        

    return loc_main, cols, norm_orig, norm_group, norm_method



# def ml_data_process(df, ma_group_by=[], norm_group_by=[], lag_group_by=[], 
#                     date_col='WORK_DATE', 
#                     ma=True, normalize=True, lag=True, 
#                     ma_cols=[], ma_except=[], norm_cols=[], norm_except=[],
#                     lag_cols=[], lag_except=[], drop_except=[],
#                     ma_values=[7,14], lag_period=3):

#     # Convert To List ......
#     ma_group_by = conv_to_list(ma_group_by)
#     norm_group_by = conv_to_list(norm_group_by)
#     lag_group_by = conv_to_list(lag_group_by)
    
#     # MA
#     ma_cols = conv_to_list(ma_cols)
#     ma_except = conv_to_list(ma_except)
    
#     # Normalize
#     norm_cols = conv_to_list(norm_cols)
#     norm_except = conv_to_list(norm_except)
    
#     # Lag
#     lag_cols = conv_to_list(lag_cols)
#     lag_except = conv_to_list(lag_except)
    
#     drop_except = conv_to_list(drop_except)
    
#     date_col_list = [date_col]
#     drop_except = drop_except + date_col_list
    
#     cols = list(df.columns)    
#     loc_main = df.copy()
    
    
#     # Calculate MA ......
#     if ma:
#         if len(ma_cols) == 0:
#             cols = li_remove_items(cols, 
#                                         ma_group_by+ma_except+date_col_list)
#         else:
#             cols = ma_cols
            
#         loc_main, ma_cols = df_add_ma(df=loc_main, cols=cols, 
#                                            group_by=ma_group_by, 
#                                            date_col=date_col, values=ma_values,
#                                            wma=False)
        
#         drop_cols = li_remove_items(cols, drop_except)
#         loc_main = loc_main.drop(drop_cols, axis=1)
#         cols = ma_cols

#     # Normalize ......
#     if normalize:
#         if len(norm_cols) == 0:
#             norm_cols = df_get_cols_except(df=loc_main, 
#                                                 except_cols=norm_group_by + norm_except)
        
#         loc_main, _, _, _ = df_normalize(df=loc_main,
#                                               cols=norm_cols,
#                                               groupby=norm_group_by,
#                                               show_progress=True)        
#     # Lag ......
#     if lag:
#         if len(lag_cols) == 0:        
#             lag_cols = df_get_cols_except(df=loc_main, 
#                                                except_cols=lag_group_by+lag_except)
        
#         loc_main, _ = df_add_shift(df=loc_main, cols=lag_cols, 
#                                         shift=lag_period,
#                                         group_by=lag_group_by,
#                                         suffix='_LAG', 
#                                         remove_na=False)    

#         drop_cols = li_remove_items(lag_cols, drop_except)
#         loc_main = loc_main.drop(drop_cols, axis=1)

#     cols = df_get_cols_except(df=loc_main, 
#                                    except_cols=ma_group_by \
#                                        +norm_group_by+lag_group_by)
        

#     return loc_main, cols




# ............
    

def cluster_sort_group(df, value_col, group_col, center):
    '''
    (1) 分群出來的編號通常與數值大小無關，因此將編號重新排序，方便閱讀。
    (2) 原本分群的編號是從0開始，但這支function出來的結果會從1開始
    (3) 目前這支function只適合單一維度的數值特徵
    
    df         : DataFrame. 包含分群用的數值，以及分群後的群集編號
    value_col  : int or float. 分群用的數值
    group_col  : int. 群集編號
    center     : list. 
    '''
    
    print('cluster_sort_group - update, 要考量group_col為None的情形')
    
    new_order = df[[value_col, group_col]] \
                .drop_duplicates(subset=group_col) \
                .reset_index(drop=True)
    
    new_order = df_add_rank(df=new_order, value=value_col, 
                            rank_name='NEW_GROUP')
    
    new_order = new_order.drop(value_col, axis=1)

    # rank的default是float，轉成int
    new_order = df_conv_col_type(new_order, cols='NEW_GROUP', to=int)
    new_order['NEW_GROUP'] = new_order['NEW_GROUP']
    
    
    # 合併資料
    cluster_results = df.merge(new_order, how='left', on=group_col)
    cluster_results = cluster_results.drop(group_col, axis=1)
    cluster_results = cluster_results.rename(columns={'NEW_GROUP':group_col})
    
    
    # 群集中心 ......
    center_results = pd.DataFrame(data=center,
                                  columns=['CENTER'])
    
    center_results = center_results.reset_index()
    center_results = center_results.rename(columns={'index':group_col})
    center_results = center_results.merge(new_order, on=group_col)
    center_results = center_results.drop(group_col, axis=1)
    center_results = center_results.rename(columns={'NEW_GROUP':group_col})
    

    export = {'CLUSTER':cluster_results,
              'CENTER':center_results}

    return export



# %% NLP系列 -----------------

def jieba_cut_item_list_v0(path='D:/tool',
                        dict_file='rt_dict', stopword_file='rt_stopwords',
                        divisions=[], sections=[], output_level=5,
                        groups=[], subgroups=[], items=[], 
                        item_status=[1,5,6,8,9], batch=2500):
    '''
    (1) 為避免像「綠巨人珍珠玉米粒 340g*3罐/組」這樣的商品名稱被誤判，因此切
        商品表之前不先用df_str_clean()排除任何的特殊符號，而是用stopword的方式處理
    (2) Update, remove special character or integer
    
    
    divisions      : list. 指定處別
    sections       : list. 指定課別
    groups         : list. 指定大分類
    subgroups      : list. 指定小分類
    items          : list. 指定商品
    cut_section    : boolean. 是否切課名
    cut_group      : boolean. 是否切大分類名稱
    cut_subgroup   : boolean. 是否切小分類名稱
    cut_item       : boolean. 是否切商品名稱
    '''
    import jieba
    jieba.initialize()
    
    jieba.load_userdict(path + "/" + dict_file + '.txt')
    
    
    hier_output = get_item_hier_sql_output(output_level=output_level)
    
    item_list_raw = get_item_list(divisions=divisions, 
                                sections=sections,
                                groups=groups, 
                                subgroups=subgroups, 
                                items=items,
                                output_level=output_level,
                                item_status=[1,5,6,8,9])
    
    item_list = item_list_raw[hier_output['KEY_NAME_LIST']]
        
    # 確認要切詞的分類
    levels = []
    
    
    # Ver 1
    # 實務上切詞只會切item_name，因為分類名稱的數量太少，沒有代表性。
    # cut_section = False
    # cut_group = False
    # cut_subgroup = False 
    # cut_item=True
    
    # if cut_section == True:
    #     levels.append(['SECTION_NO', 'SECTION_NAME'])
    
    # if cut_group == True:
    #     levels.append(['HIER_GRP', 'GRP_NAME'])        
        
    # if cut_subgroup == True:
    #     levels.append(['HIER_SUBGRP', 'SUBGRP_NAME'])
    
    # if cut_item == True:
    #     levels.append(['ITEM_NO', 'ITEM_NAME'])     
    


    # Ver 2    
    if output_level == 2:
        levels.append(['SECTION_NO', 'SECTION_NAME'])
    
    elif output_level == 3:
        levels.append(['HIER_GRP', 'GRP_NAME'])        
        
    if output_level == 4:
        levels.append(['HIER_SUBGRP', 'SUBGRP_NAME'])
    
    if output_level == 5:
        levels.append(['ITEM_NO', 'ITEM_NAME'])     
    
        

    # Iterate each level .......
    for i in range(0, len(levels)):
        
        target_code = levels[i][0]
        target_name = levels[i][1]
        
        temp_item_list = item_list[[levels[i][0], levels[i][1]]] \
                        .drop_duplicates() \
                        .reset_index(drop=True)
        
        # Iterate each row .......
        cut_name_list = []
        
        for j in range(0, len(temp_item_list)):
        
            # temp_code = temp_item_list[target_code][j]
            temp_name = temp_item_list[target_name][j]
            
            new_cut = list(jieba.cut(temp_name, cut_all=False, HMM=True)) 
            cut_name_list.append(new_cut)
                    
        cut_name_series = pd.Series(cut_name_list)
        temp_item_list[target_name+'_CUT'] = cut_name_series
        
        print(str(i) + '/' + str(len(levels)))
        
        
        # 合併資料 
        item_list = item_list.merge(temp_item_list, how='left',
                                    on=[target_code, target_name])


    # 計算詞頻 ......
    item_list_lite = item_list[['HIER_SUBGRP', 'ITEM_NAME_CUT']]
    
    unique_subgrp = item_list_lite['HIER_SUBGRP'] \
                    .drop_duplicates() \
                    .reset_index(drop=True)
    
    
    word_count_raw = pd.DataFrame()
    
    for i in range(0, len(unique_subgrp)):
        
        temp_data_pre = item_list_lite[
            item_list_lite['HIER_SUBGRP']==unique_subgrp[i]] \
            .reset_index(drop=True)
    
        temp_data = temp_data_pre['ITEM_NAME_CUT'].tolist()
        loop_times = int(np.ceil(len(temp_data) / batch))     
        
        # print(i)
    
        # 把recursive list變成flatten
        # 因為python有recursion depth limit，約為3000，因此必須拆成多次
        # 執行cbyz.li_flatten
        item_words_li = []
        
        for j in range(0, loop_times):
            temp_data_batch = temp_data[j*batch : (j+1)*batch]
            temp_li_batch_flatten = li_flatten(temp_data_batch)
            item_words_li = item_words_li + temp_li_batch_flatten
    
        # 資料整理 ......
        new_results = pd.DataFrame({'WORD':item_words_li})
        
        new_results = new_results.groupby('WORD') \
                    .size() \
                    .reset_index(name='COUNT')
    
        new_results['HIER_SUBGRP'] = unique_subgrp[i]
        word_count_raw = word_count_raw.append(new_results)
        
        
    # 排除stopwords ......
    stopwords = pd.read_csv(path + "/" + stopword_file + '.txt',
                            sep=" ", header=None)
    stopwords.columns = ['WORD']

    word_count_remove_stopwords = df_anti_merge(word_count_raw,
                                   stopwords, on='WORD')
        
    # 重新整理 ......        
    word_count = word_count_remove_stopwords \
                .sort_values(by=['HIER_SUBGRP', 'COUNT'],
                             ascending=[True, False]) \
                .reset_index(drop=True)
        
    word_count = word_count[['HIER_SUBGRP', 'WORD', 
                             'COUNT']]        
        
    
    summary = word_count.groupby('WORD') \
                .aggregate({'COUNT':sum}) \
                .reset_index() \
                .sort_values(by='COUNT', ascending=False) \
                .reset_index(drop=True)

        
    return_dict = {'ITEM_LIST':item_list,
                   'WORD_COUNT':word_count,
                   'SUMMARY':summary} 
        
    return return_dict


# ..............
    


def jieba_cut_item_list(path='D:/tool',
                        dict_file='rt_dict', stopword_file='rt_stopwords',
                        divisions=[], sections=[], output_level=5,
                        groups=[], subgroups=[], items=[], 
                        item_status=[1,5,6,8,9], batch=2500):
    '''
    (1) 為避免像「綠巨人珍珠玉米粒 340g*3罐/組」這樣的商品名稱被誤判，因此切
        商品表之前不先用df_str_clean()排除任何的特殊符號，而是用stopword的方式處理
    (2) Update, remove special character or integer
    (3) Update, title_word_count也要增加post_title_word_count，這樣之後要filter
    with website和channel才有辦法執行
    
    
    divisions      : list. 指定處別
    sections       : list. 指定課別
    groups         : list. 指定大分類
    subgroups      : list. 指定小分類
    items          : list. 指定商品
    cut_section    : boolean. 是否切課名
    cut_group      : boolean. 是否切大分類名稱
    cut_subgroup   : boolean. 是否切小分類名稱
    cut_item       : boolean. 是否切商品名稱
    '''
    import jieba
    jieba.initialize()
    jieba.load_userdict(path + "/" + dict_file + '.txt')
    
    
    # hier_output = get_item_hier_sql_output(output_level=output_level)
    
    item_list_raw = get_item_list(divisions=divisions, 
                                sections=sections,
                                groups=groups, 
                                subgroups=subgroups, 
                                items=items,
                                output_level=output_level,
                                item_status=[1,5,6,8,9])

        
    # 確認要切詞的分類
    levels = []
 
    if output_level == 2:
        levels = ['SECTION_NO', 'SECTION_NAME']
    
    elif output_level == 3:
        levels = ['HIER_GRP', 'GRP_NAME']
        
    if output_level == 4:
        levels = ['HIER_SUBGRP', 'SUBGRP_NAME']
    
    if output_level == 5:
        levels = ['ITEM_NO', 'ITEM_NAME']
    
    
    cut_col = levels[1]+'_CUT'
        

    # Iterate each level .......
    item_list = item_list_raw[levels] \
                            .drop_duplicates() \
                            .reset_index(drop=True)
    
    
    # Iterate each row .......
    cut_name_list = []
    
    for i in range(0, len(item_list)):
    
        temp_name = item_list[levels[1]][i]
        
        new_cut = list(jieba.cut(temp_name, cut_all=False, HMM=True)) 
        cut_name_list.append(new_cut)
                
    cut_name_series = pd.Series(cut_name_list)
    item_list[cut_col] = cut_name_series
    


    # 計算詞頻 ......
    word_count_raw = item_list.copy()
    word_count_raw = word_count_raw[cut_col].tolist()
    loop_times = int(np.ceil(len(word_count_raw) / batch))         
    

    # 把recursive list變成flatten
    # 因為python有recursion depth limit，約為3000，因此必須拆成多次
    # 執行cbyz.li_flatten
    words_list = []
    
    for i in range(0, loop_times):
        word_list_batch = word_count_raw[i*batch : (i+1)*batch]
        temp_li_batch_flatten = li_flatten(word_list_batch)
        words_list = words_list + temp_li_batch_flatten

    # 資料整理 ......
    word_count_pre = pd.DataFrame({'WORD':words_list})
    
    word_count_pre = word_count_pre.groupby('WORD') \
                    .size() \
                    .reset_index(name='COUNT')    
    
        
    # 排除stopwords ......
    stopwords = pd.read_csv(path + "/" + stopword_file + '.txt',
                            sep=" ", header=None)
    stopwords.columns = ['WORD']

    word_count_remove_stopwords = df_anti_merge(word_count_pre,
                                   stopwords, on='WORD')
        
    # 重新整理 ......        
    word_count = word_count_remove_stopwords \
                .sort_values(by=['COUNT'], ascending=[False]) \
                .reset_index(drop=True)
        
      
    return_dict = {'ITEM_LIST':item_list,
                   'WORD_COUNT':word_count} 
        
    return return_dict



# ..............
    

def jieba_cut_single_file(content_file,
                          dict_file='D:/tool/rt_dict.txt', 
                          stopword_file='D:/tool/rt_stopwords.txt',
                          stopword_ahead=False,
                          stopword_behind=True,
                          cut_all=False, skiprows=None,
                          export_content=False):
    '''    
    
    Update, 移除特殊符號和特殊符號的編號\u3000
    Update, 待把資料源換成CHT api
    Update, CHT_POST_ID很長，確認在資料儲存上是否會造成問題   
    
    (1) 先排除特殊符號可能會提高切詞的錯誤率，因此流程為先切詞，再排除特殊符號。

    use_stopword_head    : boolean. Jieba中有讀取stopword的function，但
                          這個function有時候會讀取失敗，因此，use_stopword_tail
                          為True時代表另外用merge排除。
    '''
    
    import jieba
    jieba.initialize()
    jieba.load_userdict(dict_file)
    
    
    # 載入stopwords
    if stopword_ahead == True:
        import jieba.analyse
        jieba.analyse.set_stop_words(stopword_file)
    
    
    if content_file == None:
        print('讀取CHT API')
    else:
        
        content = pd.read_excel(content_file, skiprows=skiprows)  
        
        if len(content) == 0:
            content = df_read_excel(content_file, 
                                    skiprows=skiprows, 
                                    sheet_name=None)
        
        if len(content) == 0:
            print('content_file列數為0，請確認是否需要設定skiprows')
            return 'content_file列數為0，請確認是否需要設定skiprows'
    
    content = content \
            .rename(columns={'輿情ID':'POST_ID',
                             '輿情標題':'TITLE',
                             '輿情網站':'WEBSITE',
                             '輿情頻道':'CHANNEL',
                             '日期':'DATE',
                             '原始網址':'URL',
                             '輿情內文':'CONTENT'})
    
    # 轉換成半型
    content['TITLE'] = content['TITLE'] \
                            .apply(str_conv_half_width)
                            
    content['CONTENT'] = content['CONTENT'] \
                            .apply(str_conv_half_width)

    
    # 主工作區 ......
    post_list_pre = pd.DataFrame()
    title_word_count_pre = pd.DataFrame()
    post_word_count_pre = pd.DataFrame()
    
    
    for i in range(0, len(content)):
        
        post_id = content.loc[i, 'POST_ID']
        title = content.loc[i, 'TITLE']
        website = content.loc[i, 'WEBSITE']
        channel = content.loc[i, 'CHANNEL']
        date = content.loc[i, 'DATE']
        link = content.loc[i, 'URL']
        post_content = content.loc[i, 'CONTENT']
        
        # 計算標題詞頻 ......
        title_cut = list(jieba.cut(title, cut_all=cut_all, HMM=True))
        title_word_count_df = pd.DataFrame({'WORD':title_cut})
        
        title_new_word_count = title_word_count_df \
                                .groupby('WORD') \
                                .size() \
                                .reset_index(name='COUNT')
        
        title_word_count_pre = title_word_count_pre \
                                .append(title_new_word_count)

        # 計算整體詞頻 ......
        content_cut = list(jieba.cut(post_content, cut_all=cut_all, HMM=True))
        post_word_count_df = pd.DataFrame({'WORD':content_cut})
        
        new_word_count = post_word_count_df \
                        .groupby('WORD') \
                        .size() \
                        .reset_index(name='COUNT')
    
        new_word_count['POST_ID'] = post_id
        post_word_count_pre = post_word_count_pre.append(new_word_count)
    

        # 文章總表 ......
        if export_content == False:
            content_cut = []
        
        new_record = pd.DataFrame({'POST_ID':[post_id],
                                   'TITLE':[title],
                                   'CONTENT_CUT':[content_cut],
                                   'WEBSITE':[website],
                                   'CHANNEL':[channel],
                                   'DATE':[date],
                                   'LINK':[link]})
    
        post_list_pre = post_list_pre.append(new_record)
        # print(str(i) + '/' + str(len(content)-1))
    
    
    # 複製物件，避免更動到原始資料
    title_word_count = title_word_count_pre.copy()
    post_word_count = post_word_count_pre[['POST_ID', 'WORD', 'COUNT']]
    
    
    # 排除特殊符號 ......
    title_word_count = df_str_clean(title_word_count, cols='WORD',
                                     remove_en=False, remove_ch=False, 
                                     remove_num=False, remove_blank=False, 
                                     drop=True)     

    post_word_count = df_str_clean(post_word_count, cols='WORD',
                                     remove_en=False, remove_ch=False, 
                                     remove_num=False, remove_blank=False, 
                                     drop=True)  
    
    # 排除stopwords ......
    if stopword_behind == True:
        stopwords = pd.read_csv(stopword_file, sep=" ", header=None)
        stopwords.columns = ['WORD']
    
        title_word_count = df_anti_merge(title_word_count, 
                                         stopwords, on='WORD')
        
        post_word_count = df_anti_merge(post_word_count, 
                          stopwords, on='WORD')

    # 整理資料 ......
        
    # 文章清單 ...    
    post_list = post_list_pre.reset_index(drop=True)
    
    # 計算標題詞頻 ...
    title_word_count = title_word_count \
                        .groupby('WORD') \
                        .aggregate({'COUNT':sum}) \
                        .reset_index()

    # 計算內文詞頻 ...
    post_word_count = post_word_count[
                        (~post_word_count['WORD'].isna()) \
                        & (post_word_count['WORD']!='')]                 
                            
    word_count = post_word_count \
                .groupby('WORD') \
                .aggregate({'COUNT':sum}) \
                .reset_index()

    # 合併成dict
    return_dict = {'POST_LIST':post_list,
                   'POST_WORD_TOTAL':word_count,
                   'TITLE_WORD':title_word_count,
                   'POST_WORD':post_word_count}
    
    return return_dict


# .............
    

def jieba_get_specific_channel(obj, website='facebook粉絲團', 
                         channel='蝦皮購物 (Shopee)'):
    
    
    # POST_LIST ......
    new_post_list = obj['POST_LIST'].copy()
    
    new_post_list = new_post_list[
        (new_post_list['WEBSITE']==website) \
        & (new_post_list['CHANNEL']==channel)]
    
        
    # Target posts
    target_posts = new_post_list[['POST_ID']].drop_duplicates() \
        .reset_index(drop=True)
            
    target_posts['IN_LIST'] = True
        
    # TITLE_WORD_COUNT
    # Update, title_word_count也要增加post_title_word_count，這樣之後要filter
    # with website和channel才有辦法執行
    # new_title_word_count = cut_multi_file['TITLE_WORD_COUNT'].copy()

    
    # POST_WORD_COUNT
    new_post_word_count = obj['POST_WORD_COUNT'].copy()
    
    new_post_word_count = new_post_word_count.merge(target_posts,
                                                    how='left',
                                                    on='POST_ID')

    new_post_word_count = new_post_word_count[
                            new_post_word_count['IN_LIST']==True]

    # WORD_COUNT ......
    new_word_count = new_post_word_count \
                    .groupby('WORD') \
                    .aggregate({'COUNT':sum}) \
                    .reset_index() \
                    .sort_values(by='COUNT', ascending=False) \
                    .reset_index(drop=True)

    return_dict = {'POST_LIST':new_post_list,
                   'TITLE_WORD_COUNT':'未完成',
                   'POST_WORD_COUNT':new_post_word_count,
                   'WORD_COUNT':new_word_count}
    
    return return_dict


# .............
    

def nlp_dict_add_word(path, file_name='rt_dict', 
                      word=None, freq=None, tag=None):
    '''
    word     : str or DataFrame.
    '''
    
    csv_dir = path+'/'+file_name+'.csv'
    txt_dir = path+'/'+file_name+'.txt'
    
    dict_csv = pd.read_csv(csv_dir)
    

    # 建立備份資料夾 ......
    backup_path = path + '/Dict_Backup'
    
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)

    
    # 備份原始檔
    eqserial = get_time_serial(with_time=True)
    backup_file = backup_path + '/' + file_name \
                    + '_backup_' + time_serial + '.csv'
                    
    dict_csv.to_csv(backup_file, index=False, encoding='utf-8-sig')
    
    
    # 更新字典檔 -------
    
    # 檢查是否已存在
    if isinstance(word, str): 
        
        chk_existing = dict_csv[dict_csv['WORD'] == word]
        # word = conv_to_list(word)
        
        if len(chk_existing) > 0:
            print('這個詞已經存在')
            return '這個詞已經存在'
        
        new_word = pd.DataFrame(data={'WORD':[word],
                                      'FREQ':[freq],
                                      'TAG':[tag]})        
        
    elif isinstance(word, pd.DataFrame):
        
        if list(word.columns) != ['WORD']:
            word.columns = ['WORD']
        
        chk_existing = dict_csv[['WORD']]
        chk_existing['EXISTING'] = True
        
        word = word[['WORD']] \
            .drop_duplicates() \
            .reset_index(drop=True)
        
        if len(word) == 0:
            print('nlp_dict_add_word 輸入的DataFrame列數為0')
            return 'nlp_dict_add_word 輸入的DataFrame列數為0'
        
        new_word = word.merge(chk_existing, how='left', on='WORD')
        
        new_word = new_word[new_word['EXISTING'].isna()] \
                    .drop('EXISTING', axis=1) \
                    .drop_duplicates() \
                    .reset_index(drop=True)

    else:
        print('請轉成str或DataFram')
    

    
    dict_csv = dict_csv \
            .append(new_word) \
            .reset_index(drop=True)
    
    
    dict_csv.to_csv(csv_dir, index=False, encoding='utf-8-sig')
    
    del dict_csv
    
    # append的空值是None，重新讀取後會變nan，為了方便處理，重新讀取檔案
    dict_csv = pd.read_csv(csv_dir)

    
    # 另存成txt ......
    dict_csv['FREQ'] = dict_csv['FREQ'].astype(str)
    dict_csv['TAG'] = dict_csv['TAG'].astype(str)
    
    dict_csv['FREQ'] = dict_csv['FREQ'].str.replace('nan', '')
    dict_csv['TAG'] = dict_csv['TAG'].str.replace('nan', '')
    
    
    dict_csv.loc[dict_csv['FREQ']!='', 'FREQ'] = ' ' + dict_csv['FREQ']
    dict_csv.loc[dict_csv['TAG']!='', 'TAG'] = ' ' + dict_csv['TAG']
    
    dict_concat = dict_csv['WORD'] + dict_csv['FREQ'] + dict_csv['TAG']
    
    # 匯出txt檔 ......
    txt_export(obj=dict_concat, file=txt_dir)
    
    return dict_concat


# ................


def nlp_dict_drop_duplicate(path, file_name='rt_dict'):
    '''
    
    '''
    
    csv_dir = path+'/'+file_name+'.csv'
    txt_dir = path+'/'+file_name+'.txt'
    
    dict_csv = pd.read_csv(csv_dir)
    

    # 建立備份資料夾 ......
    backup_path = path + '/Dict_Backup'
    
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)

    
    # 備份字典檔
    time_serial = get_time_serial(with_time=True)
    backup_file = backup_path + '/' + file_name \
                    + '_backup_' + time_serial + '.csv'
                    
    dict_csv.to_csv(backup_file, index=False, encoding='utf-8-sig')
    
    
    # 排除重複 -------
    results = dict_csv.copy()
    results = results.sort_values(by='FREQ', ascending=False) \
            .reset_index(drop=True)
        
    results = results \
            .drop_duplicates(subset='WORD') \
            .reset_index(drop=True)


    results.to_csv(csv_dir, index=False) 
    
    
    # 匯出txt檔 ......
    results['FREQ'] = results['FREQ'].astype(str)
    results['TAG'] = results['TAG'].astype(str)
    
    results['FREQ'] = results['FREQ'].str.replace('nan', '')
    results['TAG'] = results['TAG'].str.replace('nan', '')
    
    
    results.loc[results['FREQ']!='', 'FREQ'] = ' ' + results['FREQ']
    results.loc[results['TAG']!='', 'TAG'] = ' ' + results['TAG']
    
    results_series = results['WORD'] + results['FREQ'] + results['TAG']
    
    # 匯出    
    txt_export(obj=results_series, file=txt_dir)
    
    return results


# ...............
    

def nlp_dict_remove_word(word, path, file_name='rt_dict'):
    '''
    word     : str or DataFrame.
    '''
    
    csv_dir = path+'/'+file_name+'.csv'
    txt_dir = path+'/'+file_name+'.txt'
    
    dict_csv = pd.read_csv(csv_dir)
    

    # 建立備份資料夾 ......
    backup_path = path + '/Dict_Backup'
    
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)

    
    # 備份字典檔
    time_serial = get_time_serial(with_time=True)
    backup_file = backup_path + '/' + file_name \
                    + '_backup_' + time_serial + '.csv'
                    
    dict_csv.to_csv(backup_file, index=False, encoding='utf-8-sig')
    
    # 移除字元
    dict_csv = dict_csv[dict_csv['WORD']!=word]

    
    dict_csv.to_csv(csv_dir,
                    index=False,
                    encoding='utf-8-sig')
    
    del dict_csv
    
    # append的空值是None，重新讀取後會變nan，為了方便處理，重新讀取檔案
    dict_csv = pd.read_csv(csv_dir)

    
    # 另存成txt ......
    dict_csv['FREQ'] = dict_csv['FREQ'].astype(str)
    dict_csv['TAG'] = dict_csv['TAG'].astype(str)
    
    dict_csv['FREQ'] = dict_csv['FREQ'].str.replace('nan', '')
    dict_csv['TAG'] = dict_csv['TAG'].str.replace('nan', '')
    
    
    dict_csv.loc[dict_csv['FREQ']!='', 'FREQ'] = ' ' + dict_csv['FREQ']
    dict_csv.loc[dict_csv['TAG']!='', 'TAG'] = ' ' + dict_csv['TAG']
    
    dict_concat = dict_csv['WORD'] + dict_csv['FREQ'] + dict_csv['TAG']
    
    # 匯出
    txt_export(obj=dict_concat, file=txt_dir)

    
    return dict_concat


# ...............
    

def nlp_stopwords_add_word(path='D:/tool', file_name='rt_stopwords', 
                           word=None):
    '''
    word     : str or DataFrame.
    '''
    
    csv_dir = path+'/'+file_name+'.csv'
    txt_dir = path+'/'+file_name+'.txt'
    
    stopword_csv = pd.read_csv(csv_dir)
    

    # 建立備份資料夾 ......
    backup_path = path + '/Dict_Backup'
    
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)


    # 備份原始檔
    time_serial = get_time_serial(with_time=True)
    backup_file = backup_path + '/' + file_name \
                    + '_backup_' + time_serial + '.csv'
                    
    stopword_csv.to_csv(backup_file, index=False, encoding='utf-8-sig')
    
    
    # 更新字典檔 -------
    
    # 檢查是否已存在
    if isinstance(word, str): 
        
        chk_existing = stopword_csv[stopword_csv['STOPWORD'] == word]
        
        if len(chk_existing) > 0:
            print('這個詞已經存在')
            return '這個詞已經存在'
        
        new_word = pd.DataFrame(data={'STOPWORD':[word]})        
        
    elif isinstance(word, pd.DataFrame):

        if list(word.columns) != ['STOPWORD']:
            word.columns = ['STOPWORD']
        
        chk_existing = stopword_csv[['STOPWORD']]
        chk_existing['EXISTING'] = True
        
        word = word[['STOPWORD']] \
            .drop_duplicates() \
            .reset_index(drop=True)
        
        if len(word) == 0:
            print('nlp_dict_add_word 輸入的DataFrame列數為0')
            return 'nlp_dict_add_word 輸入的DataFrame列數為0'
        
        new_word = word.merge(chk_existing, how='left', on='STOPWORD')
        
        new_word = new_word[new_word['EXISTING'].isna()] \
                    .drop('EXISTING', axis=1) \
                    .drop_duplicates() \
                    .reset_index(drop=True)
                    
        if len(word) == 0:
            print('nlp_dict_add_word 輸入的DataFrame列數為0')
            return 'nlp_dict_add_word 輸入的DataFrame列數為0'                    

    else:
        print('請轉成str或DataFram')
    

    stopword_csv = stopword_csv \
            .append(new_word) \
            .reset_index(drop=True)
    

    stopword_csv.to_csv(csv_dir, index=False, encoding='utf-8-sig')
    del stopword_csv
    
    
    # append的空值是None，重新讀取後會變nan，為了方便處理，重新讀取檔案
    stopword_csv = pd.read_csv(csv_dir)

    # 另存成txt ......    
    stopword_series = stopword_csv['STOPWORD']

    # 匯出
    txt_export(obj=stopword_series, file=txt_dir)    

    return stopword_series


# ..............
    

def nlp_stopwords_drop_duplicate(path='D:/tool', file_name='rt_stopwords'):
    '''
    '''
    
    csv_dir = path+'/'+file_name+'.csv'
    txt_dir = path+'/'+file_name+'.txt'
    
    stopwords_csv = pd.read_csv(csv_dir)
    

    # 建立備份資料夾 ......
    backup_path = path + '/Dict_Backup'
    
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)

    
    # 備份字典檔
    time_serial = get_time_serial(with_time=True)
    backup_file = backup_path + '/' + file_name \
                    + '_backup_' + time_serial + '.csv'
                    
    stopwords_csv.to_csv(backup_file, index=False, encoding='utf-8-sig')

    
    # 排除重複 -------
    results = stopwords_csv \
                .copy() \
                .drop_duplicates(subset='STOPWORD') \
                .reset_index(drop=True)

    results.to_csv(csv_dir, index=False, encoding='utf-8-sig') 

    
    # 輸出成文字檔
    txt_export(obj=results['STOPWORD'], file=txt_dir)
    
    return results


# .................

    
def nlp_cht_get_store_term(exclude_stores=[4,5,8,19], include_stores=[]):
    '''
    針對中華電信輿情平台，列出特定分店的搜尋條件。    

    Parameters
    ----------
    exclude_stores : TYPE, optional
        DESCRIPTION. The default is [4,5,8,19].
    include_stores : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    store_li : TYPE
        DESCRIPTION.

    '''
    
    target_stores_raw = get_store_list(exclude_stores=exclude_stores, 
                                            include_stores=include_stores)
    
    target_stores = target_stores_raw['STORE_IN_SERIES']
    
    
    store_name = get_store_name()
    store_name['STORE_NAME'] = store_name['STORE_NAME'].str.replace('店', '')
    
    
    store_name = store_name[store_name['STORE_NO'].isin(target_stores)]
    
    
    store_li = store_name['STORE_NAME'].tolist()
    store_li = li_join_as_str(store_li, divider='|')
    store_li = '大潤發&(' + store_li + ')'
    
    return store_li



# %% 加密演算法 ----------


def df_hash(df, cols):
    '''
    1. 使用hash function加密DataFrame，目前用sha256演算法
    2. 有NA的話會出錯

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    import hashlib    
    cols = conv_to_list(cols)

    for i in range(len(cols)):        
        col = cols[i]
        df[col] = df[col].str.encode('utf-8') \
            .apply(lambda x: (hashlib.sha256(x).hexdigest().upper()))

    return df


# ............


def zip_file(src, src_path, dest, dest_path=None, password=None, 
             compression_level=9, size_limit=-1, multivolume=False):
    '''
    Update, 現在只能加入一個檔案，還沒測試加入整個資料夾
    (1) 因為內建的zipfile沒辦法加密，因此依序測試pyminizip和py7zr
    (2) py7zr可以寫出.zip，但是沒辦法直接用Windows的檔案總管開啟

    Parameters
    ----------
    src : TYPE
        DESCRIPTION.
    src_path : TYPE
        DESCRIPTION.
    dest : 壓縮檔的檔名
        由於壓縮檔可能會是.zip或.7z，因此dest不要加副檔名
    compression_level : TYPE, optional
        DESCRIPTION. The default is 9.
    password : 需為字串
    size_limit : int. 當分成多個檔案時，每個檔案的容量上限，單位為mb，預設為10mb。
    multivolume= : boolean.是否分割成多個字串

    Returns
    -------
    str
        DESCRIPTION.
    '''

    # 不指定dest_path時，預設與src_path相同
    if dest_path == None:
        dest_path = src_path

    # 預設分檔容量為10mb
    if size_limit == -1:
        size_limit = 10 * 1024 * 1024
    else:
        size_limit = size_limit * 1024 * 1024


    # py7zr ......
    if multivolume:
        import multivolumefile
        import py7zr
        
        # volume的單位是bytes，10 * 1024 * 1024代表10mb
        with multivolumefile.open(dest_path + '/' + dest + '.7z', mode='wb', 
                                  volume=size_limit) as target_archive:
            
            with py7zr.SevenZipFile(target_archive, 'w', 
                                    password=password) as archive:
                archive.writeall(src_path + src, str(src))        
                
        return 'zip_file - py7zr完成壓縮'       
    else:      
        
        with py7zr.SevenZipFile(src_path + '/' + dest + '.7z', 'w', 
                                password=password) as archive:
            archive.writeall(src_path + '/' + src)
        return 'zip_file - py7zr完成壓縮'      
    
    
    # pyminizip .....
    try:
        import pyminizip
    except:
        print('No module named pyminizip')
    else:
        pyminizip.compress(src, src_path, src_path + '/' + dest+'.zip', 
                           password, compression_level)              
        return 'os_zip_file - pyminizip完成壓縮'

        
    # zipfile ......
    if password != '':
        from zipfile import ZipFile
        zipObj = ZipFile(src_path + '/' + dest + '.zip', 'w', 
                         compresslevel=compression_level)
        zipObj.write(src_path + '/' + src)
        zipObj.close()
        return 'zip_file - ZipFile完成壓縮'



# %% 其他公開資料 -----------------


def date_get_official_calendar():
    
    
    # 資料源1 - https://data.gov.tw/dataset/14718
    # (1) 2018-2021年json檔
    # (2) 「是否放假」欄位說明:0表示該日要上班,2表示該日放假
    
    calendar_list = ['https://quality.data.gov.tw/dq_download_json.php?nid=14718&md5_url=ece3df01dc1d8ddd727011e05b954560',
                     'https://quality.data.gov.tw/dq_download_json.php?nid=14718&md5_url=45c076a6e949aec5eb8093b3a059371e',
                     'https://quality.data.gov.tw/dq_download_json.php?nid=14718&md5_url=78eba9e4421f1c9d33149f060533691c',
                     'https://quality.data.gov.tw/dq_download_json.php?nid=14718&md5_url=23a64db222b152d6435142aa2e4cbe34']

    
    # 資料源2
    # calendar_list = ['https://data.ntpc.gov.tw/api/datasets/308DCD75-6434-45BC-A95F-584DA4FED251/json?page=0&size=2000']
    
    
    import requests
    results = pd.DataFrame()
    
    for i in range(len(calendar_list)):
        
        # 爬取網頁資料 ......
        r = requests.get(calendar_list[i])    
        
        # r.headers['content-type']
        # r.encoding    
    
        new_data = pd.read_json(r.text)
        results = results.append(new_data)
    
    
    results.columns = ['WORK_DATE', 'WEEK_DAY', 'WORK_DAY', 'NOTE']

    
    results.loc[results['WORK_DAY']==0, 'WORK_DAY'] = 1
    results.loc[results['WORK_DAY']==2, 'WORK_DAY'] = 0
    
    return results


# .............


def get_covid19_data():
    
    # 這個function用的資料源 ......
    # 地區年齡性別統計表-嚴重特殊傳染性肺炎-依個案研判日統計
    # https://data.gov.tw/dataset/120711
    # > 日期有缺值

    # 其他資料源 ......
    # COVID-19全球疫情資訊
    # https://data.gov.tw/dataset/128062
    
    # 台灣COVID-19冠狀病毒檢測每日送驗數
    # https://data.gov.tw/dataset/120451   
    
    # COVID-19台灣最新病例、檢驗統計
    # https://data.gov.tw/dataset/120450    

    
    try:
        link = 'https://od.cdc.gov.tw/eic/Day_Confirmation_Age_County_Gender_19CoV.csv'
        data = pd.read_csv(link)
    except:
        data = pd.read_csv('/Users/Aron/Documents/GitHub/Data/Stock_Forecast/2_Stock_Analysis/Resource/covid.csv')
    
    
        
    data.columns = ['COL1', 'WORK_DATE', 'COL2', 'COL3', 'COL4',
                    'COL5', 'COL6', 'COVID19'] 

    data = data[['WORK_DATE', 'COVID19']]
    data = df_ymd(df=data, cols='WORK_DATE')
    data = df_date_simplify(df=data, cols='WORK_DATE')

    data = data.groupby(['WORK_DATE']).agg({'COVID19':'sum'}).reset_index()
    
    # Calendar ......
    min_data = data['WORK_DATE'].min()
    max_data = data['WORK_DATE'].max()
    calendar = get_rt_calendar(begin_date=min_data, end_date=max_data,
                               simplify=True)
    calendar = calendar[['WORK_DATE']]
    
    # 合併資料 ......
    data = calendar.merge(data, how='left', on='WORK_DATE')
    data = df_conv_na(df=data, cols='COVID19')

    return data


# .............


def get_festival():

      # 元旦，NY
      # CNY，用除夕算
      # 228紀念日，Anniversary228。避免只用數字，dcast變成欄位名稱的時候才不會出問題。
      # 兒童節，Children 
      # 清明節，Tomb
      # 母親節，Mother
      # 端午節，Dragon Boat
      # 中元節，Ghost
      # 中秋節，MOON    
    
    cols = ['YEAR', 'FESTIVAL_NAME', 'FESTIVAL', 'DATE']


    # 節日時間表 ......
    results = \
    pd.DataFrame(data=np.array([[2021, "元旦", "NY", 20210101]]), columns=cols) \
    .append(pd.DataFrame(data=np.array([[2020, "元旦", "NY", 20200101]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "元旦", "NY", 20190101]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "元旦", "NY", 20180101]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "元旦", "NY", 20170101]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "元旦", "NY", 20160101]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "元旦", "NY", 20150101]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2021, "CNY", "CNY", 20210211]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2020, "CNY", "CNY", 20200124]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "CNY", "CNY", 20190204]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "CNY", "CNY", 20180215]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "CNY", "CNY", 20170127]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "CNY", "CNY", 20160207]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "CNY", "CNY", 20150218]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2021, "228紀念日", "Anniv228", 20210228]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2020, "228紀念日", "Anniv228", 20200228]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "228紀念日", "Anniv228", 20190228]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "228紀念日", "Anniv228", 20180228]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "228紀念日", "Anniv228", 20170228]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "228紀念日", "Anniv228", 20160228]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "228紀念日", "Anniv228", 20150228]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2021, "兒童節", "Children", 20210404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2020, "兒童節", "Children", 20200404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "兒童節", "Children", 20190404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "兒童節", "Children", 20180404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "兒童節", "Children", 20170404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "兒童節", "Children", 20160404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "兒童節", "Children", 20150404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2021, "清明節", "Tomb", 20210405]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2020, "清明節", "Tomb", 20200404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "清明節", "Tomb", 20190405]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "清明節", "Tomb", 20180405]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "清明節", "Tomb", 20170404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "清明節", "Tomb", 20160404]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "清明節", "Tomb", 20150405]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2021, "母親節", "Mother", 20210509]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2020, "母親節", "Mother", 20200510]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "母親節", "Mother", 20190512]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "母親節", "Mother", 20180513]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "母親節", "Mother", 20170514]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "母親節", "Mother", 20160508]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "母親節", "Mother", 20150510]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2021, "端午節", "Dragon_Boat", 20210614]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2020, "端午節", "Dragon_Boat", 20200625]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "端午節", "Dragon_Boat", 20190607]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "端午節", "Dragon_Boat", 20180618]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "端午節", "Dragon_Boat", 20170530]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "端午節", "Dragon_Boat", 20160609]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "端午節", "Dragon_Boat", 20150620]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2021, "中元節", "Ghost", 20210822]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2020, "中元節", "Ghost", 20200902]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "中元節", "Ghost", 20190815]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "中元節", "Ghost", 20180825]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "中元節", "Ghost", 20170905]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "中元節", "Ghost", 20160817]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "中元節", "Ghost", 20150828]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2021, "中秋節", "Moon", 20210921]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2020, "中秋節", "Moon", 20201001]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2019, "中秋節", "Moon", 20190913]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2018, "中秋節", "Moon", 20180922]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2017, "中秋節", "Moon", 20171004]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2016, "中秋節", "Moon", 20160915]]), columns=cols)) \
    .append(pd.DataFrame(data=np.array([[2015, "中秋節", "Moon", 20150927]]), columns=cols)) 
    
    
    results = results.reset_index(drop=True)
    results = df_conv_col_type(df=results, cols=['YEAR', 'DATE'], to=int)
    
    
    # 檢查資料是否正確
    chk = results.copy()
    chk = df_ymd(df=results, cols='DATE')
    chk['CHK_YEAR'] = chk['DATE'].dt.year
    chk = chk[chk['YEAR']!=chk['CHK_YEAR']]
    
    if len(chk) > 0:
        print('get_festival資料有誤')
        print(chk)
        return 'get_festival資料有誤'
    
    return results





def pytrends_single(begin_date, end_date, words, by_day=False, hl='zh-TW', 
                    geo='TW'):
    '''
    1. 抓取Google Trend的資料，目前與網頁版比對後，數字相符
    2. 以下使用了兩支function
       (1) pytrend.build_payload：可以一次抓五組關鍵字，但是當資料區間太長時(大約是10個月)，
       回傳的資料會變成by week，而不是by day。
       (2) pytrend.dailydata.get_daily_data：一次只能抓一組關鍵字，但回傳的格式會是by day，

    3. Bug, 使用multiple的時候，如果benchmark在第一round的數值很高，但後面都是0的話會出錯，
    增加一個check
   
    '''
    
    from pytrends.request import TrendReq
    
    # tz為時區timezone，
    # US CST 的時區為-6，在Google的系統顯示為360；台灣的時區為+8，因此應該-480
    pytrend = TrendReq(hl=hl, tz=-480)


    # 設定時間 ......
    # 不知道為什麼，輸入的int有時候會自己變成float
    begin_date = ymd(int(begin_date))
    end_date = ymd(int(end_date))
    

    # 拆分詞組 ......
    # 1. words最多只能有5個字，因此必須拆成多組list
    # 2. 由於Google Trend的資料都是比較值，而不是絕對值，為了讓多個list之間可以有一個比
    #    較基準點(benchmark)，第0組list有五個字，第1~n組都只有四個字，並且會將第一組
    #    list中數值最高的字當成比較基準，加入第1~n組中。
    words = conv_to_list(words)
    words_li = []
    temp_li = []
    
    for i in range(0, len(words)):
        temp_li.append(words[i])

        if (i > 0 and i % 4 == 0) or (i == len(words) - 1):
            words_li.append(temp_li)
            temp_li = []


    # 執行pytrend.build_payload .........    
    benchmark = ''
    results_raw = pd.DataFrame()
    timeframe = begin_date.strftime("%Y-%m-%d") + ' ' \
                    + end_date.strftime("%Y-%m-%d")    
    
    for i in range(0,len(words_li)):
        
        # 增加benchmark
        query_words = words_li[i][:]
        if i > 0:
            query_words.append(benchmark)

        # Query
        pytrend.build_payload(
                kw_list=query_words,
                cat=0,
                timeframe=timeframe,
                geo=geo)
        
        
        # API回傳的格式為pivot          
        data = pytrend.interest_over_time()
        data = data.reset_index()
        data = pd.melt(data, id_vars=['date'], value_vars=query_words)

        data = df_conv_col_type(df=data, cols='value', to='int')
        data['group'] = i

        # 定義benchmark ......
        if i == 0:
            benchmark = data \
                        .groupby(['variable']) \
                        .agg({'value':'mean'}) \
                        .reset_index()
                        
            benchmark['MAX'] = benchmark['value'].max()
            benchmark = benchmark[benchmark['value']==benchmark['MAX']]
            benchmark = benchmark['variable'].tolist()
            benchmark = benchmark[0]
            
            benchmark_data = data[data['variable']==benchmark] \
                            .reset_index()
                            
            benchmark_data = benchmark_data[['date', 'variable', 'value']] \
                            .rename(columns={'value':'benchmark_value'})
                            

        # 標記benchmark ......
        if i == 0:
            data['benchmark'] = 0
        else:
            data['benchmark'] = np.where(data['variable']==benchmark, 1, 0)
        
        results_raw = results_raw.append(data)
        

    # 根據benchmark調整value ......
    if len(results_raw) > 0:
        
        # 計算每一個group要乘上的倍數 ......
        results = results_raw.merge(benchmark_data, how='left', 
                                    on=['date', 'variable'])
        
        results['multfi_times'] = results['benchmark_value'] \
                                / results['value']
                                
        results = df_conv_na(df=results, cols='multfi_times')
        results['group_multfi_times'] = results \
                                .groupby(['group', 'date'])['multfi_times'] \
                                .transform('max')
        
        results['value'] = results['value'] * results['group_multfi_times']
        
        results = results[results['benchmark']==0]
        results = results[['date', 'variable', 'value']].reset_index(drop=True)
        
        
        msg = '''pytrends_single - Bug, 如果benchmar中有數值是0，會導致b
                enchmark_value為NA，且multfi_times為inf，這裡轉換時就會出錯。
                解決方法是當出現NA時，把數值量大的當成benchmark重新query一次。
                '''
        # Bug, 如果benchmar中有數值是0，會導致benchmark_value為NA，且
        # multfi_times為inf，這裡轉換時就會出錯。
        print(msg)
        results = df_conv_col_type(df=results, cols='value', to='int')
        
    else:
        results = results_raw[['date', 'variable', 'value']] \
                    .reset_index(drop=True)

        
    # 將欄位名稱轉成大寫 ......
    cols = list(results.columns)
    cols = [i.upper() for i in cols]
    results.columns = cols        
    
    return results



def pytrends_multi(begin_date, end_date, words, chunk=180, unit='d', 
                   hl='zh-TW', geo='TW'):
    '''
    當unit為m時，chunk建議不要超過10個月，不然pytrends可能會抓不到資料
    '''
    
    date_df = date_split(begin_date=begin_date, end_date=end_date, 
                         chunk=chunk, unit=unit, simplify=True, full=False)
    
    # Query ......
    results = pd.DataFrame()
    
    for i in range(len(date_df)):
        trend = pytrends_single(begin_date=date_df.loc[i, 'BEGIN_DATE'], 
                            end_date=date_df.loc[i, 'END_DATE'], 
                            words=words, by_day=True, hl=hl, 
                            geo=geo)
        
        results = results.append(trend)
        # print('pytrends_multi - ' + str(i) + '/' + str(len(date_df)))
        
        
    # 將全部數值轉成0到100
    ratio = results['VALUE'].max() / 100
    results['VALUE'] = results['VALUE'] / ratio
    
        
    return results



# def df_pytrends_backup_20210531(begin_date, end_date, words, by_day=False, hl='zh-TW', 
#                 geo='TW'):
#     '''
#     以下使用了兩支function
#     1. pytrend.build_payload：可以一次抓五組關鍵字，但是當資料區間太長時(大約是10個月)，
#        回傳的資料會變成by week，而不是by day。
#     2. pytrend.dailydata.get_daily_data：一次只能抓一組關鍵字，但回傳的格式會是by day，
#        這個方式會回傳其他詳細資料，我還沒搞懂它的意思。
       
#    3. Update, 第二支function會有很多欄位，只保留需要的
#    [家樂福_unscaled, 家樂福_monthly, isPartial, scale, 家樂福]
#     '''
    
    
#     from pytrends.request import TrendReq
#     # timezone要幹嘛的？
#     pytrend = TrendReq(hl=hl, tz=360)
    
    
#     # 設定時間 ......
#     begin_date = ymd(begin_date)
#     end_date = ymd(end_date)
#     diff_days = date_diff(begin_date, end_date) + 1
    

#     # 拆分詞組 ......
#     # words最多只能有5個字，因此必須拆成多個list
#     words = conv_to_list(words)
#     words_li = []
#     temp_li = []
    
#     for i in range(0, len(words)):
#         temp_li.append(words[i])

#         if i % 4 == 0 or i == len(words) - 1:
#             words_li.append(temp_li)
#             temp_li = []


#     # 執行pytrend.build_payload .........    
#     results = pd.DataFrame()
    
#     for i in range(0,len(words_li)):
        
#         timeframe = begin_date.strftime("%Y-%m-%d") + ' ' \
#                         + end_date.strftime("%Y-%m-%d")
        
#         pytrend.build_payload(
#                 kw_list=words_li[i],
#                 cat=0,
#                 timeframe=timeframe,
#                 geo=geo)
        
#         data = pytrend.interest_over_time()
#         data = data.drop('isPartial', axis=1).reset_index()
        
#         # API回傳的格式為pivot        
#         cols = df_get_cols_except(df=data, except_cols='date')
#         data = pd.melt(data, id_vars=['date'], value_vars=cols)
        
#         results = results.append(data)
        
#     results = results.reset_index(drop=True)


#     # 執行pytrend.dailydata.get_daily_data .........    
#     if by_day == True and len(results) < diff_days * len(words):

#         print('df_pytrends - 執行pytrend.dailydata.get_daily_data')
#         from pytrends import dailydata        
#         results = pd.DataFrame()
        
#         # 這裡的欄位和和上面不一樣
#         # [家樂福_unscaled, 家樂福_monthly, isPartial, scale, 家樂福]
#         for j in range(len(words)):
#             data = dailydata.get_daily_data(words[j], 
#                                           begin_date.year,
#                                           begin_date.month,
#                                           end_date.year,
#                                           end_date.month,
#                                           geo=geo)
            
#             data = data.drop('isPartial', axis=1).reset_index()   
#             data['WORD'] = words[j]

#             # API回傳的格式為pivot        
#             cols = df_get_cols_except(df=data, 
#                                            except_cols=['date', 'WORD'])
#             data = pd.melt(data, id_vars=['date', 'WORD'], value_vars=cols)
            
#             results = results.append(data)
            
#         results = results.reset_index()
                       
        
#     # 將欄位名稱轉成大寫 ......
#     cols = list(results.columns)
#     cols = [i.upper() for i in cols]
#     results.columns = cols        
    
#     return results





# %% Cheat Sheet -----------------

# import scipy.stats

# 統計檢定 ---------

# 常態分佈    
# dat1 = [14.2,14.0,14.4,14.4,14.2,14.6,14.9,15.0,14.2,14.8]
# dat2 = [13.8,13.6,14.7,13.9,14.3,14.1,14.3,14.5,13.6,14.6]
# scipy.stats.shapiro(dat1)



# 模型評估 ---------

# Classification
# (1) Confusion Matrix
# sklearn.metrics.confusion_matrix(y_true, y_pred)
# (2) ROC Curve and AUC
# sklearn.metrics.roc_curve(y_true, y_score)

# Regression 
# (1) R² or Coefficient of determination
# sklearn.metrics.r2_score(y_true, y_pred)
# (2) Median Absolute Deviation (MAD)
# scipy.stats.median_absolute_deviation(x)


# Clustering
# https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html


# %% 範本 ----------------


def save_file_template(export_file=True, load_file=False, path=None,
                file_name=None):

    '''
    儲存暫存檔的範本
    '''
    
    # 確認暫存檔名稱 ......
    if file_name == None:
        file_name = 'get_tnv_qty'
        

    # 讀取暫存檔 ......  
    if (path != None) and (load_file == True):
        try:
            results = pd.read_csv(path + '/' + file_name + '.csv')
            print('讀取暫存檔 ' + path + '/' + file_name + '.csv')
            return results
        except:
            print('get_tnv_qty無法讀取暫存檔')

    
    
    # 儲存暫存檔 ......
    if (export_file == True) and (path != None):
        
        try:
            results.to_csv(path + '/' + file_name + '.csv', 
                           index=False, 
                           encoding='utf-8')
        except:
            print('get_tnv_qty無法儲存檔案')
        
    
    return results
    

# %% 開發區 -----------------

