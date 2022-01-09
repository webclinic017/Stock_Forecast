# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 10:38:37 2022

@author: 吳雅智 Aron Wu
"""

# % 讀取套件 -------

import pandas as pd
import numpy as np
import sys, time, os, gc


# 設定工作目錄 .....
path = r'D:\Data_Mining\Projects\Codebase_YZ\Dev\\LSTM'
# path = r'/home/rserver/Data_Mining'
path = r'/Users/aron/Documents/GitHub/Stock_Forecast/Dev/lstm'


# Codebase
path_codebase = [path + '/Function',
                 'D:/Data_Mining/Projects/Codebase_YZ',
                 r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ']

for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path

    
import codebase_yz as cbyz
import codebase_ml as cbml



# % 手動設定區 -------

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

 




# Referenece
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    
# https://ithelp.ithome.com.tw/articles/10195400
    

    
    
    
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

import matplotlib.pyplot as plt
dataset = pd.read_csv(path_resource + '/airline-passengers.csv',
                      usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()


# 我們可以觀察到數據有一定的趨勢與波動性，暑假(7~9月)乘客人數比較多，表示有季節效應，我們可以用ACF或PACF確認一下：

# 畫出 ACF 12 期的效應
sm.graphics.tsa.plot_acf(dataset, lags=12)
plt.show()
# 畫出 PACF 12 期的效應
sm.graphics.tsa.plot_pacf(dataset, lags=12)
plt.show()    



# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# 產生 (X, Y) 資料集, Y 是下一期的乘客數
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# 載入訓練資料
dataframe = read_csv(path_resource + '/airline-passengers.csv',
                     usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# 正規化(normalize) 資料，使資料值介於[0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 2/3 資料為訓練資料， 1/3 資料為測試資料
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# 產生 (X, Y) 資料集, Y 是下一期的乘客數(reshape into X=t and Y=t+1)
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 建立及訓練 LSTM 模型
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 預測
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 回復預測資料值為原始數據的規模
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate 均方根誤差(root mean squared error)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# 畫訓練資料趨勢圖
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# 畫測試資料趨勢圖
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# 畫原始資料趨勢圖
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()





# %% Hack ------

from sklearn.model_selection import train_test_split

model_data_raw = pd.read_csv(path_resource + '/model_data.csv')

model_data = cbyz.df_conv_col_type(df=model_data_raw, cols='SYMBOL', to='str')
model_data = model_data[model_data['SYMBOL'].str.slice(0, 2)=='26']
model_data = model_data.drop(['OPEN_CHANGE_RATIO', 
                              'HIGH_CHANGE_RATIO',
                              'LOW_CHANGE_RATIO'],
                             axis=1)

train_data = model_data[~model_data['CLOSE_CHANGE_RATIO'].isna()] \
                .reset_index(drop=True)
train_data = train_data.drop(['SYMBOL', 'WORK_DATE'], axis=1)

pred_data = model_data[model_data['CLOSE_CHANGE_RATIO'].isna()]

# X = train_data.drop('CLOSE_CHANGE_RATIO', axis=1)
X = train_data[['OPEN_CHANGE_RATIO_MA_5_LAG']]

y = train_data['CLOSE_CHANGE_RATIO']
# cbyz.df_chk_col_na(df=X)

X_train, X_test, y_train, y_valid = \
    train_test_split(X, y, test_size=0.1)


X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()


X_train_np = numpy.reshape(X_train_np, (X_train_np.shape[0], 1, X_train_np.shape[1]))
X_test_np = numpy.reshape(X_test_np, (X_test_np.shape[0], 1, X_test_np.shape[1]))


# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# 建立及訓練 LSTM 模型
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train_np, y_train_np, epochs=100, batch_size=1, verbose=2)



pred_train = model.predict(X_train_np)
pred_test = model.predict(X_test_np)


# 回復預測資料值為原始數據的規模
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate 均方根誤差(root mean squared error)
trainScore = math.sqrt(mean_squared_error(y_train_np[0], pred_train[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))



