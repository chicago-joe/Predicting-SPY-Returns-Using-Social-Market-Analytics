from readdata import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def maketable(data):

    tickertable = []
    ticker_name = []
    for ticker in data:
        ticker_name.append(ticker)
        singleticker=pd.DataFrame.from_dict(data[ticker])
        singleticker=singleticker.transpose()
        col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
                   's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')

        singleticker.columns=col_names
        for names in col_names:
            if names!= 'date':
                singleticker[names] = singleticker[names].astype(float)
        tickertable.append(singleticker['s_volume'].mean())

    return tickertable, ticker_name


def svol_over_time(data):

    tickertable = []
    ticker_name = []
    for ticker in data:
        ticker_name.append(ticker)
        singleticker=pd.DataFrame.from_dict(data[ticker])
        singleticker=singleticker.transpose()
        col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
                   's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')

        singleticker.columns=col_names
        for names in col_names:
            if names!= 'date':
                singleticker[names] = singleticker[names].astype(float)
        tickertable.append(singleticker['s_volume'])

    return tickertable, ticker_name



alldatamean,all_tickers = maketable(df_alldata)

# convert mean of s_volume table to dataframe
df_all_data = pd.DataFrame(alldatamean,all_tickers)
top10_idx = np.argsort(df_all_data)[-2:]
top10_values = [df_all_data[i] for i in top10_idx]

# make data frame then write to csv
df = pd.DataFrame(top10_values)
# df.to_csv('mean')

########## Now check out the 12 tickers change in s_volume over time ############
# df_12ticker_table = []
ticker_mean,ticker_list = svol_over_time(df_12ticker)
df_12ticker_data = pd.DataFrame(df_12ticker)

df_12ticker_data.columns
df_12ticker_data.index

XLK = df_12ticker_data['XLK']
    # .str.split(', ',expand=True)
# df_XLK = XLK.str.split(', ', expand=True)
# XLK = pd.DataFrame(XLK.str.split(', ', expand=True))

col_names = ('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
                   's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')

# XLK.columns = col_names
# print(XLK)
XLV = df_12ticker_data['XLV']
XLF = df_12ticker_data['XLF']
XLY = df_12ticker_data['XLY']
XLI = df_12ticker_data['XLI']
XLP = df_12ticker_data['XLP']
XLE = df_12ticker_data['XLE']
XLU = df_12ticker_data['XLU']
VNQ = df_12ticker_data['VNQ']
GDX = df_12ticker_data['VOX']
VOX = df_12ticker_data['GDX']

with pd.ExcelWriter('output.xlsx') as writer:
    XLK.to_excel(writer,sheet_name = 'XLK')
    XLF.to_excel(writer,sheet_name = 'XLF')
    XLY.to_excel(writer,sheet_name = 'XLY')
    XLI.to_excel(writer,sheet_name = 'XLI')
    XLP.to_excel(writer,sheet_name = 'XLP')
    XLE.to_excel(writer,sheet_name = 'XLE')
    XLU.to_excel(writer,sheet_name = 'XLU')
    VNQ.to_excel(writer,sheet_name = 'VNQ')
    VOX.to_excel(writer,sheet_name = 'VOX')
    GDX.to_excel(writer,sheet_name = 'GDX')

pd.reade


# tickers=['XLK','XLV','XLF','XLY','XLI','XLP','XLE','XLU','VNQ','GDX','VOX']
#
# for i in df_12ticker_data:
#     for ticker in tickers:
#         df = df_12ticker_data[i]
    # tickers = []
    # tmp = df[i]
    # tickers.append(tmp)














# print(df.head(df_12ticker))
# df.from_items()
# dict.items(df_12ticker)

# def maketable(alldata):
#     for ticker in alldata:
#
#         # print(ticker)
#         singleticker=pd.DataFrame.from_dict(alldata[ticker])
#         singleticker=singleticker.transpose()
#         col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
#                    's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')
#
#         singleticker.columns=col_names
#         for names in col_names:
#             if names!= 'date':
#                 singleticker[names] = singleticker[names].astype(float)
#         return singleticker['s_volume']
#
#
# alldatatable = maketable(df)
