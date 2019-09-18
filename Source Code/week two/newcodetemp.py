# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:28:21 2019

@author: duany
"""

from readdata import *
from spy15mindata import *
spyprice=pd.read_csv('SPY Price Data.csv')
spyprice.index=spyprice['Date']
spyprice['Return']=(spyprice['Adj_Close'][:-1]-spyprice['Adj_Close'][1:].values)/spyprice['Adj_Close'][:-1]
spyprice=spyprice['Return'][:-1]
oneticker=etfsmean['SPY']

result = pd.concat([oneticker, spyprice], axis=1, sort=False,join='inner')
print(spydata)
print(D2spydata)
print(etfsmean)
