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

address = 
SPYstat = address+"\SPYstationarity.txt"
SPYdaily = address+"\SPYdaily.txt"


df_stat_SPY = pd.read_csv(SPYstat,delimiter = ",")
df_daily_SPY = pd.read_csv(SPYdaily, delimiter = ",")

df_daily_SPY = df_daily_SPY.dropna()
df_daily_SPY.set_index('Date',inplace=True)

target = df_daily_SPY["next_Return"]
df_daily_SPY.drop(columns = ['next_Return']) 