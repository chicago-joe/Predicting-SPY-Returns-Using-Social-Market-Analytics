from readdata import *
import pandas as pd
import numpy as np

# alldata[]

def maketable(alldata):
    # tickertable = pd.DataFrame()
    tickertable = []
    for ticker in alldata:

        # print(ticker)
        singleticker=pd.DataFrame.from_dict(alldata[ticker])
        singleticker=singleticker.transpose()
        col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
                   's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')

        singleticker.columns=col_names
        for names in col_names:
            if names!= 'date':
                singleticker[names] = singleticker[names].astype(float)
        tickertable.append(singleticker['s_volume'].mean())

    return tickertable

alldatatable = maketable(df_12ticker)


# from readdata import *
# import pandas as pd
# import numpy as np
#
#
#
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
#
#
