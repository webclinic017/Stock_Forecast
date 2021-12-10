


cbyz.df_chk_col_na(df=result)



chk = loc_main[loc_main['OPEN_CHANGE_RATIO_MA_60_LAG'].isna()]


chk = loc_main[loc_main['SYMBOL']=='1101']
chk = loc_main[loc_main['SYMBOL']=='6191']
cbyz.df_chk_col_na(df=chk)

chk = chk[chk['OPEN'].isna()]
