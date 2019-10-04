# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:28:21 2019

@author: duany
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:35:50 2019

@author: duany
"""
import os
import pandas as pd
import time
import numpy as np 
from yahoo_historical import Fetcher
import statsmodels.api as sm
import statsmodels.formula.api as smf               
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller   
import csv
import seaborn as sns
import math

from spyonlydata import *

plt.rcParams['figure.dpi'] = 120
col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','next_Return','today_Return','classret')


###check for the significance
def significance_check(singleticker):
    
    singleticker.columns=col_names
    est_return = smf.ols(formula='next_Return~raw_s+raw_s_mean+raw_volatility+raw_score+s+\
                  s_mean+s_volatility+s_score+s_volume+sv_mean+sv_volatility+sv_score+s_dispersion+\
                  +s_buzz+s_delta', data=singleticker).fit()
    
    est_return_2 = smf.ols(formula='next_Return~sv_mean', data=singleticker).fit()
    print(est_return.summary()) 
    print(est_return_2.summary()) 


def random_forest_classification(singleticker):
    
    from sklearn.ensemble import RandomForestClassifier 
    from sklearn.datasets import make_regression
    from sklearn.model_selection import GridSearchCV 
    from sklearn.metrics import classification_report
    
    warnings.filterwarnings("ignore")
    
    forest = RandomForestClassifier(random_state=42)
    para = {'max_depth':range(1,15), 'min_samples_leaf':range(1,15)}
    
    CV_forest= GridSearchCV(forest,para,cv=6, n_jobs= 4,iid = True,refit= True)
    singleticker_new=singleticker
    singleticker_new=singleticker_new.drop(columns=['today_Return','next_Return','classret'])
            
    CV_forest.fit(singleticker_new,singleticker['classret'] )
    best_leaf = CV_forest.best_params_['min_samples_leaf']
    best_depth = CV_forest.best_params_['max_depth']
    
    forest_best = RandomForestClassifier(random_state=42,min_samples_leaf=best_leaf,max_depth=best_depth,n_jobs=-1)
    forest_best.fit(singleticker_new, singleticker['classret'])
    
    y_pred_rf = forest_best.predict(singleticker_new)
    
    importances = forest_best.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.title('Feature Importance')
    plt.bar(range(singleticker_new.shape[1]), importances[indices], align='center')
    plt.xticks(range(singleticker_new.shape[1]),singleticker_new.columns[indices], rotation=90)
    plt.xlim([-1, singleticker_new.shape[1]])
    plt.tight_layout()
    plt.show()
    
    print(classification_report(singleticker['classret'], y_pred_rf))

def plotting(singleticker):
    
    singleticker.columns=col_names
    
    for col in ['raw_s','s_volume','s_delta']:
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        ax1.set_xlabel('SPY')
        ax1.set_ylabel(col, color=color)
        ax1.plot(singleticker[col], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        
        color = 'tab:blue'
        ax2.set_ylabel('next_Return', color=color) 
        ax2.plot(singleticker['next_Return'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout() 
        plt.xticks(np.arange(0,len(singleticker[col]),20.0))
        
        plt.show()

def distribution(singleticker):
    
    singleticker.columns=col_names
    
    result=singleticker[['raw_s','s_volume','s_delta']]
    sns.pairplot(result)
    g = sns.PairGrid(result)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, n_levels=6)

def stationarity(singleticker):
    
    singleticker.columns=col_names   
    result=singleticker[['raw_s','s_volume','s_delta']]
    result['s_delta'][0]=0

    
    plist=[]
    for col in result:
        st=adf_test(result[col])['p-value']
        plist.append(st)
        
        adf_test(result[col])
        acf_pacf_plot(result[col])
        
    return plist

def adf_test(timeseries):
    
    dftest = adfuller(timeseries, autolag='t-stat')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return(dfoutput)
    
def acf_pacf_plot(ts_log_diff):
    sm.graphics.tsa.plot_acf(ts_log_diff,lags=40) #ARIMA,q
    sm.graphics.tsa.plot_pacf(ts_log_diff,lags=40) #ARIMA,p
    

def outlier(singleticker):
    
    singleticker.columns=col_names   
    result=singleticker[['raw_s','s_volume','s_delta']]
    
    fig, axs = plt.subplots(4, 4)
    count=0
    for col in result:
    
        axs[math.floor(count/4), count%4].boxplot(result[col])
        axs[math.floor(count/4), count%4].set_title(col)
        count+=1
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=1.5,
                        hspace=0.4, wspace=0.3)
    fig.dpi=100

#--------------------------------------------------------------------------------------#
print('start outlier')

oldtime = time.time()
stlist=[]
outlier(SPYdailydata)
newtime = time.time()
print('outlier use: %.3fs' % (
       newtime - oldtime
   ))
#--------------------------------------------------------------------------------------#

print('start stationarity')

oldtime = time.time()
stationarity(SPYdailydata)
newtime = time.time()
print('stationarity use: %.3fs' % (
        newtime - oldtime
    ))

#--------------------------------------------------------------------------------------#
print('start distribution')
oldtime = time.time()

distribution(SPYdailydata)

newtime = time.time()
print('distribution use: %.3fs' % (
       newtime - oldtime
   ))
#--------------------------------------------------------------------------------------#

volume=SPYdailydata['s_volume']
plt.plot(volume)

from datetime import datetime
weekday=[]
date=SPYdailydata.index
for i in date:
    weekday.append(datetime.strptime(i, '%Y-%m-%d').weekday()+1)
season={"weekday" : weekday,
   "volume" : volume}

season=pd.DataFrame(season)

Season_index=[]
for i in range(5):
    Season_index.append(np.mean(season[season['weekday']==(i+1)]['volume'])/np.mean(season['volume']))

for i in range(5):
    if i==0:
        season_ad2=season[season['weekday']==(i+1)]['volume']/Season_index[i]
    else:
        season_ad=season[season['weekday']==(i+1)]['volume']/Season_index[i]
        season_ad2=season_ad2.combine(season_ad,max,fill_value=0)  
season['Vol_adjust']=season_ad2

stationariy_check=adf_test(season['Vol_adjust'])
print(stationariy_check)