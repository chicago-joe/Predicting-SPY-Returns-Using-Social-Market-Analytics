# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:54:04 2019

@author: Zenith Zhou

"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

address = "C:/Users/jloss/PyCharmProjects/SMA-HullTrading-Practicum/Source Code/week 6/"
SPYstat = address+"SPYstationarity.txt"
SPYdaily = address+"SPYdaily.txt"

# convert txt to pandas DF
df_stat_SPY = pd.read_csv(SPYstat,delimiter = ",")
df_daily_SPY = pd.read_csv(SPYdaily, delimiter = ",")

# clean empty cell
df_daily_SPY = df_daily_SPY.dropna()
df_daily_SPY.set_index('Date',inplace=True)

target = df_daily_SPY["next_Return"]
df_daily_SPY = df_daily_SPY.drop(["next_Return"],axis=1)
df_daily_SPY = df_daily_SPY.drop(["today_Return"],axis=1)

# filter for unstationary
features = df_daily_SPY.columns
for name in features:
    if df_stat_SPY[name].bool() == False:
        df_daily_SPY=df_daily_SPY.drop([name],axis=1)
        
X_train, X_test, y_train, y_test = train_test_split(df_daily_SPY, target, 
                                                    test_size=0.2,
                                                    random_state=42)


# Trending for Ridge
## 0-0.5 study
alpha = np.arange(0,0.5,0.005)
ridge_df = pd.DataFrame()

for i in range(len(alpha)):
    ridge = Ridge(alpha=alpha[i])
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    
    ridge_df[float(alpha[i])] = ridge.coef_

ridge_df = ridge_df.transpose()

ridge_df.plot()
plt.ylim(-0.05,0.05)
plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
plt.title("Ridge With 0 - 0.5 Range")
plt.ylabel("Coefficient")
plt.xlabel("Alpha Value")

## 0-0.05 study
alpha = np.arange(0,0.0005,0.000005)
ridge_df = pd.DataFrame()

for i in range(len(alpha)):
    ridge = Ridge(alpha=alpha[i])
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    
    ridge_df[float(alpha[i])] = ridge.coef_

ridge_df = ridge_df.transpose()

ridge_df.plot()
plt.ylim(-0.05,0.05)
plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
plt.title("Ridge With 0 - 0.05 Range")
plt.ylabel("Coefficient")
plt.xlabel("Alpha Value")

# Trending for Lasso