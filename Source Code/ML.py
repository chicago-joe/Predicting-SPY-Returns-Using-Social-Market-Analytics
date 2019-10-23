# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:54:04 2019

@author: Zenith Zhou

"""

import seaborn as sns
import pandas as pd
import time
import numpy as np
import sklearn

address = "C:/Users/yz_ze/Documents/GitHub/SMA-HullTrading-Practicum/Data/"
SPYstat = address+"SPYstationarity.txt"
SPYdaily = address+"SPYdaily.txt"

# convert txt to pandas DF
df_stat_SPY = pd.read_csv(SPYstat,delimiter = ",")
df_daily_SPY = pd.read_csv(SPYdaily, delimiter = ",")

# clean empty cell
df_daily_SPY = df_daily_SPY.dropna()
df_daily_SPY.set_index('Date',inplace=True)

target = df_daily_SPY["next_Return"]
df_daily_SPY.drop(["next_Return"],axis=1)
df_daily_SPY.drop(["today_Return"],axis=1)

# filter for unstationary
features = df_daily_SPY.columns
for name in features:
    if df_stat_SPY[name].bool() == False:
        df_daily_SPY=df_daily_SPY.drop([name],axis=1)