# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:28:21 2019

@author: duany
"""

from readdata import *
from spy15mindata import *
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.stattools import adfuller   
import statsmodels.api as sm
import csv
import seaborn as sns

def distribution(etfsmean):
    for ticker in etfsmean:
        oneticker=etfsmean[ticker]
        result = pd.concat([oneticker, spyprice], axis=1, sort=False,join='inner')
        result = result.drop(columns='date')
        sns.pairplot(result)
        g = sns.PairGrid(result)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.kdeplot, n_levels=6)

def stationarity(result):
    plist=[]
    for col in result:
        if adf_test(result[col])['p-value']<0.05:
            st=True
        else:
            st=False
        plist.append(st)
    return plist

def adf_test(timeseries):
    #print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return(dfoutput)
    
def acf_pacf_plot(ts_log_diff):
    sm.graphics.tsa.plot_acf(ts_log_diff,lags=40) #ARIMA,q
    sm.graphics.tsa.plot_pacf(ts_log_diff,lags=40) #ARIMA,p
    

def outlier(result):
    fig, axs = plt.subplots(4, 4)
    count=0
    for col in result:
    
        axs[math.floor(count/4), count%4].boxplot(result[col])
        axs[math.floor(count/4), count%4].set_title(col)
        count+=1
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=1.5,
                        hspace=0.4, wspace=0.3)
    fig.dpi=100

spyprice=pd.read_csv('SPY Price Data.csv')
spyprice.index=spyprice['Date']
spyprice['Return']=(spyprice['Adj_Close'][1:]-spyprice['Adj_Close'][:-1].values)/spyprice['Adj_Close'][1:]
spyprice=spyprice['Return'][1:]

print('start outlier')

oldtime = time.time()

stlist=[]
for ticker in etfsmean:
    oneticker=etfsmean[ticker]
    result = pd.concat([oneticker, spyprice], axis=1, sort=False,join='inner')
    result = result.drop(columns='date')

    outlier(result)
newtime = time.time()
print('outlier use: %.3fs' % (
        newtime - oldtime
    ))

print('start stationarity')
oldtime = time.time()

stlist=pd.DataFrame(stlist,index=list(etfsmean.keys()),columns=result.columns)
stlist.to_csv('stationarity.csv')
for ticker in etfsmean:
    oneticker=etfsmean[ticker]
    result = pd.concat([oneticker, spyprice], axis=1, sort=False,join='inner')
    result = result.drop(columns='date')

    stlist.append(stationarity(result))
newtime = time.time()
print('stationarity use: %.3fs' % (
        newtime - oldtime
    ))
    
print('start distribution')
oldtime = time.time()

distribution(etfsmean)
newtime = time.time()
print('distribution use: %.3fs' % (
        newtime - oldtime
    ))
