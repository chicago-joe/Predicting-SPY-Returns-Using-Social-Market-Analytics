# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:35:50 2019

@author: duany
"""
import os
import pandas as pd
import time
import numpy as np 
#from yahoo_historical import Fetcher


def readdata(filename):
    oldtime = time.time()
    f = open(filename)
    s = f.read()
    f.close()
    ticker = s.split('\n')
    dfout = {} 
    itercars = iter(ticker)
    next(itercars)
    for tickerdata in itercars:
        namesplit = tickerdata.split(':')
        tickername=namesplit[0]
        #print(tickername)
        if dfout.get(namesplit[0]):
            tickerout=dfout[namesplit[0]]
        else:
            tickerout = {}
        datesdata = namesplit[1].split('/')
        
        for datedata in datesdata:
            colsdata=datedata.split(',')
            dataout = []
            for coldata in colsdata:
                dataout.append(coldata)
            datename=dataout[len(dataout)-1]
            
            tickerout[datename]=dataout
        dfout[tickername]=tickerout

    newtime = time.time()
    print('read data time: %.3f' % (
        newtime - oldtime
    ))
    oldtime = time.time()
    return dfout

def meanclosetable(ticker,df):
    oldtime = time.time()

    price = Fetcher(ticker, [2015,9,1], [2017,12,31]).getHistorical()
    price=price.set_index((price['Date']))
    for timedata in df[ticker]:
        if sum(price.index==timedata)==1:
            df[ticker][timedata].append(price.loc[timedata][4])
    newtime = time.time()
    print('update data time: %.3f' % (
        newtime - oldtime
    ))
    return df

def submeanclosetableonly(tickerlist,df):
    newdf={}
    for ticker in tickerlist: 
        newdf[ticker] = dict((k, v) for k, v in df[ticker].items() if len(v) >= 17)
    return newdf


def datainsight(tickerlist,df):
    #for ticker in tickerlist: 
    #    df=meanclosetable(ticker,df)
    #newdf=submeanclosetableonly(tickerlist,df)
    allticker={}
    
    for ticker in tickerlist: 
        singleticker=pd.DataFrame.from_dict(df[ticker])
        singleticker=singleticker.transpose()
        col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')
        
        singleticker.columns=col_names
        for names in col_names:
            if names!= 'date':
                singleticker[names] = singleticker[names].astype(float)
        allticker[ticker]=singleticker
    return allticker
       

df=readdata('./meantable12ticker.txt')


tickerlist=['XLK','XLV','XLF','XLY','XLI','XLP','XLE','XLU','VNQ','GDX','VOX','SPY']

etfsmean=datainsight(tickerlist,df)
