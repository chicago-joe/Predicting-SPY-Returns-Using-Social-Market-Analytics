import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir('/Users/sherry/Box/SMAHULL/Source Code/week 8')
train_x ="train_x.txt"
train_y ="train_y.txt"
test_x = "test_x.txt"
test_y ="test_y.txt"

train_x= pd.read_csv(train_x,delimiter=",",index_col=0)
train_y= pd.read_csv(train_y,delimiter=",",index_col=0,header=None)

test_x= pd.read_csv(test_x,delimiter=",",index_col=0)
test_y= pd.read_csv(test_y,delimiter=",",index_col=0,header=None)

scaler = StandardScaler()
scaler.fit(train_x)   
X_scaled = scaler.transform(train_x)
X_test_scaled=scaler.transform(test_x)

table={}
for degree in range(2,3):
    print(degree)
    poly = PolynomialFeatures(degree)
    X_poly=poly.fit_transform(X_scaled)
    X_test_poly=poly.transform(X_test_scaled)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, train_y)
    out=pol_reg.predict(X_test_poly)
    
    up_dir = 0
    down_dir = 0
    for i in range(len(out)):
        if (out[i]>0) and (test_y.iloc[i,0]>0):
            up_dir += 1
        elif ((out[i]<0) and (test_y.iloc[i,0]<0)):
            down_dir += 1
        else:
            continue
    
    up_dir_y = 0
    down_dir_y = 0
    for i in test_y.iloc[:,0]:
        if i > 0:
            up_dir_y += 1
        else:
            down_dir_y += 1
    
    up_dir_pred = 0
    down_dir_pred = 0
    for i in range(len(out)):
        if out[i]>0:
            up_dir_pred += 1
        else:
            down_dir_pred += 1
            
    table["degree: %i"%degree]=[up_dir,down_dir,up_dir_y,down_dir_y,up_dir_pred,down_dir_pred]

r2_score(test_y, out)                   
MSE(test_y, out)    

