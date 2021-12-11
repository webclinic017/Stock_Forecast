



chk = chk_min_max[chk_min_max['MIN_VALUE']<0]


cbyz.df_chk_col_na(df=market_data_raw,
                        cols=['VOLUME_CHANGE_RATIO', 
                              'VOLUME_CHANGE_ABS_RATIO'])

chk = market_data_raw[market_data_raw['VOLUME_CHANGE_RATIO']==np.inf]
chk


chk['VOLUME_CHANGE_RATIO'].isna()



chk_min_max = cbyz.df_chk_col_min_max(df=market_data_raw)


# 1611中電在20211211-20211219都沒有交易，導致交易量為0；1732毛寶在20190903也是一樣的情形



#                       COLUMN     MIN_VALUE     MAX_VALUE
# 0        VOLUME_CHANGE_RATIO -1.000000e+00           inf
# 1    VOLUME_CHANGE_ABS_RATIO  0.000000e+00           inf
# 2                     VOLUME  0.000000e+00  1.281795e+09
# 3              VOLUME_CHANGE -8.345367e+08  1.109633e+09
# 4          VOLUME_CHANGE_ABS  0.000000e+00  1.109633e+09
# 5          TOTAL_TRADE_VALUE  1.014979e+08  8.871566e+08
# 6         SYMBOL_TRADE_VALUE  0.000000e+00  1.452704e+08
# 7                  WORK_DATE  2.019052e+07  2.021122e+07
# 8                       HIGH  2.710000e+00  1.325000e+03
# 9                       OPEN  2.490000e+00  1.290000e+03
# 10                     CLOSE  2.600000e+00  1.265000e+03
# 11                       LOW  2.490000e+00  1.225000e+03
# 12           HIGH_CHANGE_ABS  0.000000e+00  4.000000e+02
# 13          CLOSE_CHANGE_ABS  0.000000e+00  3.750000e+02
# 14           OPEN_CHANGE_ABS  0.000000e+00  3.740000e+02
# 15            LOW_CHANGE_ABS  0.000000e+00  3.590000e+02
# 16                LOW_CHANGE -3.590000e+02  1.300000e+02
# 17             SHADOW_LENGTH  0.000000e+00  1.170000e+02
# 18               OPEN_CHANGE -3.740000e+02  1.100000e+02
# 19                       BAR  0.000000e+00  1.070000e+02
# 20              CLOSE_CHANGE -3.750000e+02  1.050000e+02
# 21               HIGH_CHANGE -4.000000e+02  1.000000e+02
# 22                TOP_SHADOW -2.000000e+00  7.300000e+01
# 23             BOTTOM_SHADOW -2.500000e+00  6.700000e+01
# 24        CLOSE_CHANGE_LEVEL  1.000000e+00  4.000000e+00
# 25    CLOSE_CHANGE_ABS_RATIO  0.000000e+00  1.268132e+00
# 26          LOW_CHANGE_RATIO -4.998504e-01  1.268132e+00
# 27        CLOSE_CHANGE_RATIO -5.028249e-01  1.268132e+00
# 28      LOW_CHANGE_ABS_RATIO  0.000000e+00  1.268132e+00
# 29     HIGH_CHANGE_ABS_RATIO  0.000000e+00  1.268132e+00
# 30         HIGH_CHANGE_RATIO -5.161290e-01  1.268132e+00
# 31     OPEN_CHANGE_ABS_RATIO  0.000000e+00  1.268132e+00
# 32         OPEN_CHANGE_RATIO -5.020846e-01  1.268132e+00
# 33             K_LINE_TYPE_0  0.000000e+00  1.000000e+00
# 34             K_LINE_TYPE_3  0.000000e+00  1.000000e+00
# 35             K_LINE_TYPE_4  0.000000e+00  1.000000e+00
# 36             K_LINE_TYPE_6  0.000000e+00  1.000000e+00
# 37             K_LINE_TYPE_7  0.000000e+00  1.000000e+00
# 38             K_LINE_TYPE_8  0.000000e+00  1.000000e+00
# 39             K_LINE_TYPE_2  0.000000e+00  1.000000e+00
# 40             K_LINE_TYPE_1  0.000000e+00  1.000000e+00
# 41             K_LINE_TYPE_9  0.000000e+00  1.000000e+00
# 42            K_LINE_COLOR_R  0.000000e+00  1.000000e+00
# 43            K_LINE_COLOR_G  0.000000e+00  1.000000e+00
# 44                  WEEK_NUM  0.000000e+00  1.000000e+00
# 45                   WEEKDAY  0.000000e+00  1.000000e+00
# 46                     MONTH  0.000000e+00  1.000000e+00
# 47                      YEAR  0.000000e+00  1.000000e+00
# 48                DATE_INDEX  0.000000e+00  9.968404e-01
# 49  SYMBOL_TRADE_VALUE_RATIO  0.000000e+00  2.116768e-01



chk = market_data_raw[market_data_raw['WORK_DATE']==20211207]
chk












