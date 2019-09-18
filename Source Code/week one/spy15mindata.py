# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:35:32 2019

@author: duany
"""
import time
import pandas as pd
import numpy as np 
import os


def readfilepath(REPORT_PATH,limit=[]):
    filelist=[[],[],[]]
    fullpathlist=[[],[],[]]
    
    for i in range(len(REPORT_PATH)):
        for dirpath, dirnames, filenames in os.walk(REPORT_PATH[i]):
            for file in filenames:
                if len(limit)!=0:
                    ticker=file.split('.')[0]
                    if ticker not in limit:
                        #print(ticker)

                        continue 
                filelist[i].append(file)
                fullpath = os.path.join(dirpath, file)
                fullpathlist[i].append(fullpath)

    return fullpathlist,filelist


def SPYdata(fullpathlist):
    totalfilecount=0
    outlist={}
    for i in range(len(fullpathlist)):
        totalfilecount+=len(fullpathlist[i])
    count=0
    oldtime = time.time()
    dfout = {}
    tickerkey=''
    for i in range(len(fullpathlist)):
        for j in range(len(fullpathlist[i])):
            count+=1
            print('percent: %.3f' % (
                    count/totalfilecount*100
                ))
            tempdata=pd.read_csv(fullpathlist[i][j], sep="\t",skiprows=5)
            if len(tempdata['ticker'])>0 :
                tempdata=tempdata.set_index(pd.to_datetime(tempdata['date']))
                tickerkey=tempdata['ticker'][0]
                if dfout.get(tickerkey):
                    tempticker=tickerkey
                    dfout[tickerkey]+=1
                else:
                    tempticker=tickerkey
                    dfout[tickerkey]=1

                datelist=(np.unique(np.array([pd.to_datetime(date).date() for date in tempdata.date]))) 
                for k in ((datelist)):
                    outlist[k]=tempdata[str(k)+' 07:29:59':str(k)+' 18:00:01']
                    tempticker=''
        
    newtime = time.time()
    print('mean table use: %.3fs' % (
            newtime - oldtime
        ))
    return outlist

def D2_D1(D2spydata):
    outdf=pd.DataFrame()
    for date in D2spydata:
                    
        outdf=outdf.append(D2spydata[date])
    return outdf

fullpathlist,filelist=readfilepath(['./2015/','./2016/','./2017/'],['SPY'])

D2spydata=SPYdata(fullpathlist)

spydata=D2_D1(D2spydata)