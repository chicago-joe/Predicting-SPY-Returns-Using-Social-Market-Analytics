from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense, Activation   
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
os.chdir('/Users/sherry/Box/SMAHULL/Source Code/week 7')
from scipy.stats.mstats import winsorize
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

def stationarity(result):
    plist={}
    for col in result:
        if adf_test(result[col])['p-value']<0.05:
            st=True
        else:
            st=False
        plist[col]=st
    return plist

def adf_test(timeseries):
    #print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return(dfoutput)
    
def using_mstats(s):
    return winsorize(s, limits=[0.005, 0.005])

def winsorize_test_set(df_train,test_x):
    name=[]
    max_=[]
    min_=[]
    for i in df_train.columns:
        name.append(i)
        max_.append(max(df_train[i]))
        min_.append(min(df_train[i]))
    a=pd.DataFrame([max_,min_])
    a.columns=name
    
    for i in test_x.columns:
        test_x.loc[(test_x[i]<a[i][0])][i]=a[i][0]
        test_x.loc[(test_x[i]<a[i][1])][i]=a[i][1]
        
df_daily_SPY = pd.read_csv("SPYdaily.txt", delimiter = ",")
df_train=df_daily_SPY[0:527].drop(columns='Date')
df_test=df_daily_SPY[527:].drop(columns='Date')

df_train = df_train.dropna(axis='rows')
df_train = df_train.apply(using_mstats, axis=0)

slist=(stationarity(df_train))
slist=pd.DataFrame(slist, index=[0])
factors=[]
for i in slist.columns:
    if slist[i][0]==1:
        factors.append(i)

train_x=df_train[factors]
train_y=df_train['next_Return']
test_x=df_test[factors]
test_y=df_test['next_Return']

winsorize_test_set(train_x,test_x)

scaler = StandardScaler()
scaler.fit(train_x)        # compute the mean and std dev which will be used below
X_scaled = scaler.transform(train_x)
X_test_scales=scaler.transform(test_x)

pca = PCA(n_components = 4)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)
X_test_pca=pca.transform(X_test_scales)

plt.figure(figsize=(16, 26))
hm = sns.heatmap(pca.components_,
                 cbar=False,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 10},
                 yticklabels=['1st Comp','2nd Comp','3rd Comp','4th Comp'],
                 xticklabels=train_x.columns)

model = Sequential()
model.add(Dense(20, init='uniform', input_dim=4))
model.add(Activation('linear'))
 
model.add(Dense(20))
model.add(Activation('linear'))

model.add(Dense(10))
model.add(Activation('linear'))
 
model.add(Dense(1))
model.add(Activation('linear'))

#model.compile(loss='mean_squared_error', optimizer="adam", metrics=["accuracy",Accuracy_approx])
sgd = SGD(lr=0.0001, decay=0.00001)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
  
hist=model.fit(X_pca, train_y, batch_size=64, epochs=300, shuffle=True,verbose=0,validation_split=0.2)
plt.plot(hist.history['loss'])
plt.xlabel('epoch')
plt.ylabel('cost function')
plt.show()

model.evaluate(X_pca, train_y, batch_size=10)
model.evaluate(X_test_pca,test_y,batch_size=10)

out=model.predict(X_test_pca)
out=out.reshape(-1,)
len(test_y[(abs((out-test_y)/test_y)<0.1)])/len(test_y)
