

# https://www.datacamp.com/community/tutorials/xgboost-in-python
# https://towardsdatascience.com/xgboost-python-example-42777d01001e


from sklearn.datasets import load_boston
boston = load_boston()

print(boston.keys())
# dict_keys(['data', 'target', 'feature_names', 'DESCR'])


print(boston.data.shape)
print(boston.feature_names)


import pandas as pd

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names


data['PRICE'] = boston.target

data.info()
data.describe()


# %% Split train and test set

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

X, y = data.iloc[:,:-1],data.iloc[:,-1]

data_dmatrix = xgb.DMatrix(data=X,label=y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# %% XGBoost -----------
xg_reg = xgb.XGBRegressor(#objective ='reg:squarederror',
                          objective ='reg:linear',
                          colsample_bytree = 0.3, 
                          learning_rate = 0.1,
                          max_depth = 5, 
                          alpha = 10, 
                          n_estimators = 10)


xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


params = {"objective":"reg:linear",
          'colsample_bytree': 0.3,
          'learning_rate': 0.1,
          'max_depth': 5, 
          'alpha': 10}


cv_results = xgb.cv(dtrain=data_dmatrix, 
                    params=params, 
                    nfold=3,
                    num_boost_round=50,
                    early_stopping_rounds=10,
                    metrics="rmse", 
                    as_pandas=True, 
                    seed=123)

cv_results.head()


print((cv_results["test-rmse-mean"]).tail(1))
