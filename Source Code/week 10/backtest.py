################################################################
# ensembling.py
# Ensemble Methods
# Created by Joseph Loss on 11/06/2019
#
# Contact: loss2@illinois.edu
###############################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score,accuracy_score,explained_variance_score
import pylab as plot
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 

def SPY():
    
    
    spyprice=pd.read_csv('SPY Price Data.csv')
    spyprice.index=spyprice['Date']
    next_Return=(spyprice['Adj_Close'][:-1].values-spyprice['Adj_Close'][1:])/spyprice['Adj_Close'][1:]
    today_Return=(spyprice['Adj_Close'][:-1]-spyprice['Adj_Close'][1:].values)/spyprice['Adj_Close'][1:].values
    type(today_Return)
    
    sd_Return=today_Return.iloc[::-1].rolling(250).std().iloc[::-1]
    sd_Return=sd_Return.dropna()
    sd_Return=sd_Return[1:]

    classret=[ 2  if next_Return[date]>sd_Return[date]*1.0 else 1 if next_Return[date]>sd_Return[date]*0.05 else 0 if next_Return[date]>sd_Return[date]*-0.05 else -1 if next_Return[date]>sd_Return[date]*-1.0 else -2 for date in sd_Return.index]
    #classret=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in next_Return]
    classret=pd.DataFrame(classret)
    classret.index=sd_Return.index
    classret.columns=['classret']
    
    todayclassret=[ 2  if today_Return[date]>sd_Return[date]*1.0 else 1 if today_Return[date]>sd_Return[date]*0.05 else 0 if today_Return[date]>sd_Return[date]*-0.05 else -1 if today_Return[date]>sd_Return[date]*-1.0 else -2 for date in sd_Return.index]
    #classret=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in next_Return]
    todayclassret=pd.DataFrame(todayclassret)
    todayclassret.index=sd_Return.index
    todayclassret.columns=['todayclassret']
    
    next_Return=pd.DataFrame(next_Return)
    today_Return=pd.DataFrame(today_Return)
    next_Return.columns=['next_Return']
    today_Return.columns=['today_Return']
    return today_Return,next_Return,classret,todayclassret


def readdata():
    colum_names = ['ticker', 'date', 'description', 'sector', 'industry', 'raw_s', 's-volume', 's-dispersion', 'raw-s-delta', 'volume-delta', 'center-date', 'center-time', 'center-time-zone']
    df_2015 = pd.read_csv('SPY2015ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2016 = pd.read_csv('SPY2016ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2017 = pd.read_csv('SPY2017ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
    #aggregating data
    df_temp = df_2015.append(df_2016, ignore_index = True)
    df_aggregate = df_temp.append(df_2017, ignore_index = True)
    df_datetime = df_aggregate['date'].str.split(' ', n = 1, expand = True )
    df_datetime.columns = ['Date', 'Time']
    df = pd.merge(df_aggregate, df_datetime, left_index = True, right_index = True)
    #filtering based on trading hours and excluding weekends
    df = df[(df['Time'] >= '09:30:00') & (df['Time'] <= '16:00:00')]
    #excluding weekends
    #removing empty columns
    df = df.dropna(axis='columns')
    df=df.drop(columns=['ticker','date','description','center-date','center-time','center-time-zone', 'raw-s-delta', 'volume-delta'])
    df["volume_base_s"]=df["raw_s"]/df["s-volume"]
    df["ewm_volume_base_s"] = df.groupby("Date")["volume_base_s"].apply(lambda x: x.ewm(span=390).mean())
    dffinal = df.groupby('Date').last().reset_index()
    dffinal.index=dffinal['Date']
    dffinal["mean_volume_base_s"] = df.groupby("Date")["volume_base_s"].mean()
    dffinal["mean_raw_s"] = df.groupby("Date")["raw_s"].mean()
    dffinal["mean_s_dispersion"] = df.groupby("Date")["s-dispersion"].mean()
    dffinal['volume_base_s_z']=(dffinal['mean_volume_base_s']-dffinal['mean_volume_base_s'].rolling(26).mean())/dffinal['mean_volume_base_s'].rolling(26).std()
    dffinal['s_dispersion_z']=(dffinal['mean_s_dispersion']-dffinal['mean_s_dispersion'].rolling(26).mean())/dffinal['mean_s_dispersion'].rolling(26).std()
    dffinal['raw_s_MACD_ewma12-ewma26'] = dffinal["mean_raw_s"].ewm(span=12).mean() - dffinal["mean_raw_s"].ewm(span=26).mean()
    df1=dffinal.drop(columns=['Date','raw_s','s-volume','s-dispersion','Time','volume_base_s'])
    df1.columns="spy_"+df1.columns

    colum_names = ['ticker', 'date', 'description', 'sector', 'industry', 'raw_s', 's-volume', 's-dispersion', 'raw-s-delta', 'volume-delta', 'center-date', 'center-time', 'center-time-zone']
    df_2015 = pd.read_csv('ES_F2015ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2016 = pd.read_csv('ES_F2016ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2017 = pd.read_csv('ES_F2017ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
    df_temp = df_2015.append(df_2016, ignore_index = True)
    df_aggregate = df_temp.append(df_2017, ignore_index = True)
    df_datetime = df_aggregate['date'].str.split(' ', n = 1, expand = True )
    df_datetime.columns = ['Date', 'Time']
    df = pd.merge(df_aggregate, df_datetime, left_index = True, right_index = True)
    df = df[(df['Time'] >= '09:30:00') & (df['Time'] <= '16:00:00')]
    df = df.dropna(axis='columns')
    df=df.drop(columns=['ticker','date','description','center-date','center-time','center-time-zone', 'raw-s-delta', 'volume-delta'])
    df["volume_base_s"]=df["raw_s"]/df["s-volume"]
    df["ewm_volume_base_s"] = df.groupby("Date")["volume_base_s"].apply(lambda x: x.ewm(span=390).mean())
    dffinal = df.groupby('Date').last().reset_index()
    dffinal.index=dffinal['Date']
    dffinal["mean_volume_base_s"] = df.groupby("Date")["volume_base_s"].mean()
    dffinal["mean_raw_s"] = df.groupby("Date")["raw_s"].mean()
    dffinal["mean_s_dispersion"] = df.groupby("Date")["s-dispersion"].mean()
    dffinal['volume_base_s_delta']=(dffinal['mean_volume_base_s'][1:]-dffinal['mean_volume_base_s'][:-1].values)
    dffinal['s_dispersion_delta']=(dffinal['mean_s_dispersion'][1:]-dffinal['mean_s_dispersion'][:-1].values)
    dffinal['raw_s_MACD_ewma12-ewma26'] = dffinal["mean_raw_s"].ewm(span=12).mean() - dffinal["mean_raw_s"].ewm(span=26).mean()
    df2=dffinal.drop(columns=['Date','raw_s','s-volume','s-dispersion','Time','volume_base_s'])
    df2.columns="future_"+df2.columns
    testdf = pd.concat([df1, df2], axis=1, sort=False)
    today_Return,next_Return,classret,todayclassret=SPY()
    sd_Return=today_Return.iloc[::-1].rolling(250).std().iloc[::-1]
    sd_Return=sd_Return.dropna()
    sd_Return=sd_Return[1:]
    sd_Return
    sd_Return.columns=['sd_Return']
    testdf = pd.concat([testdf, next_Return,today_Return,classret,sd_Return,todayclassret], axis=1, sort=False,join='inner')
    return testdf

testdf=readdata().dropna()

X_train=testdf[0:464]
X_test=testdf[464:]

# Preprocess / Standardize data
sc_X = StandardScaler()
X_train_std = sc_X.fit_transform(X_train)
X_test_std = sc_X.transform(X_test)

y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)

#best_leaf_nodes = None
#best_n = 11

RFmodel = RandomForestRegressor(n_estimators = best_n,max_leaf_nodes=best_leaf_nodes,n_jobs=-1)
#RFmodel = RandomForestClassifier(n_estimators = best_n,max_leaf_nodes=best_leaf_nodes,n_jobs=-1)
RFmodel.fit(X_train_std, y_train)




# predict on in-sample and oos
y_train_pred = RFmodel.predict(X_train_std)
y_test_pred = RFmodel.predict(X_test_std)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

#print('accuracy train: %.3f, test: %.3f' % (
#        accuracy_score(y_train, y_train_pred),
#        accuracy_score(y_test, y_test_pred)))

print('explanined variance train: %.3f, test: %.3f' % (
        explained_variance_score(y_train, y_train_pred),
        explained_variance_score(y_test, y_test_pred)))

# RFmodel.score(X_test_std,y_train_std)

# plot Feature Importance of RandomForests model
featureImportance = RFmodel.feature_importances_
featureImportance = featureImportance / featureImportance.max()    # scale by max importance
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0]) + 0.5
plot.barh(barPos, featureImportance[sorted_idx], align = 'center')      # chart formatting
plot.yticks(barPos, featNames[sorted_idx])
plot.xlabel('Variable Importance')
plot.show()


plt.scatter(y_train_pred.reshape(-1,1),
            (y_train_pred.reshape(-1,1) - y_train.reshape(-1,1)),
            c='steelblue',
            edgecolors = 'white',
            marker='o',
            s=35,
            alpha=0.9,
            label='Training data')
plt.scatter(y_test_pred.reshape(-1,1),
            (y_test_pred.reshape(-1,1) - y_test.reshape(-1,1)),
            c='limegreen',
            edgecolors = 'white',
            marker='s',
            s=35,
            alpha=0.9,
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-0.075, xmax=0.075, lw=2, color='black')
plt.xlim([-0.075,0.075])
plt.show()


up_dir = 0
down_dir = 0
up_0_00=0
up_00_1=0
up_1_5=0
up_5=0
down_0_00=0
down_00_1=0
down_1_5=0
down_5=0
up_0_00_dir=0
up_00_1_dir=0
up_1_5_dir=0
up_5_dir=0
down_0_00_dir=0
down_00_1_dir=0
down_1_5_dir=0
down_5_dir=0

pre_up_0_00_dir=0
pre_up_00_1_dir=0
pre_up_1_5_dir=0
pre_up_5_dir=0
pre_down_0_00_dir=0
pre_down_00_1_dir=0
pre_down_1_5_dir=0
pre_down_5_dir=0
pred_y=y_test_pred
for i in range(len(pred_y)):
    
    if ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0) and (test_y_df.iloc[i,0]<0.001)):
        up_0_00_dir += 1
    elif ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0.001) and (test_y_df.iloc[i,0]<0.01)):
        up_00_1_dir += 1
    elif ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0.01) and (test_y_df.iloc[i,0]<0.05)):
        up_1_5_dir += 1
    elif ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0.05)):
        up_5_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<0) and (test_y_df.iloc[i,0]>-0.001)):
        down_0_00_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<-0.001) and (test_y_df.iloc[i,0]>-0.01)):
        down_00_1_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<-0.01) and (test_y_df.iloc[i,0]>-0.05)):
        down_1_5_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<-0.05)):
        down_5_dir += 1


    if ((pred_y[i]>0.001) and (test_y_df.iloc[i,0]>0) and (test_y_df.iloc[i,0]<0.001)):
        pre_up_0_00_dir += 1
    elif ((pred_y[i]>0.001) and (test_y_df.iloc[i,0]>0.001) and (test_y_df.iloc[i,0]<0.01)):
        pre_up_00_1_dir += 1
    elif ((pred_y[i]>0.001) and (test_y_df.iloc[i,0]>0.01) and (test_y_df.iloc[i,0]<0.05)):
        pre_up_1_5_dir += 1
    elif ((pred_y[i]>0.001) and (test_y_df.iloc[i,0]>0.05)):
        pre_up_5_dir += 1
    elif ((pred_y[i]<-0.001) and (test_y_df.iloc[i,0]<0) and (test_y_df.iloc[i,0]>-0.001)):
        pre_down_0_00_dir += 1
    elif ((pred_y[i]<-0.001) and (test_y_df.iloc[i,0]<-0.001) and (test_y_df.iloc[i,0]>-0.01)):
        pre_down_00_1_dir += 1
    elif ((pred_y[i]<-0.001) and (test_y_df.iloc[i,0]<-0.01) and (test_y_df.iloc[i,0]>-0.05)):
        pre_down_1_5_dir += 1
    elif ((pred_y[i]<-0.001) and (test_y_df.iloc[i,0]<-0.05)):
        pre_down_5_dir += 1

    
    if ((pred_y[i]>0) and (pred_y[i]<0.001) and (test_y_df.iloc[i,0]>0) and (test_y_df.iloc[i,0]<0.001)):
        up_0_00 += 1
    elif ((pred_y[i]>0.001) and (pred_y[i]<0.01) and (test_y_df.iloc[i,0]>0.001) and (test_y_df.iloc[i,0]<0.01)):
        up_00_1 += 1
    elif ((pred_y[i]>0.01) and (pred_y[i]<0.05) and (test_y_df.iloc[i,0]>0.01) and (test_y_df.iloc[i,0]<0.05)):
        up_1_5 += 1
    elif ((pred_y[i]>0.05) and (test_y_df.iloc[i,0]>0.05)):
        up_5 += 1
    elif ((pred_y[i]<0) and (pred_y[i]>-0.001) and (test_y_df.iloc[i,0]<0) and (test_y_df.iloc[i,0]>-0.001)):
        down_0_00 += 1
    elif ((pred_y[i]<-0.001) and (pred_y[i]>-0.01) and (test_y_df.iloc[i,0]<-0.001) and (test_y_df.iloc[i,0]>-0.01)):
        down_00_1 += 1
    elif ((pred_y[i]<-0.01) and (pred_y[i]>-0.05) and (test_y_df.iloc[i,0]<-0.01) and (test_y_df.iloc[i,0]>-0.05)):
        down_1_5 += 1
    elif ((pred_y[i]<-0.05)  and (test_y_df.iloc[i,0]<-0.05)):
        down_5 += 1

    
    if ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0)):
        up_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<0)):
        down_dir += 1


up_dir_y = 0
down_dir_y = 0
up_0_00_y=0
up_00_1_y=0
up_1_5_y=0
up_5_y=0
down_0_00_y=0
down_00_1_y=0
down_1_5_y=0
down_5_y=0
for i in test_y_df.iloc[:,0]:
    if i>0 and i<0.001:
        up_0_00_y+=1
    elif i>0.001 and i<0.01:
        up_00_1_y+=1
    elif i>0.01 and i<0.05:
        up_1_5_y+=1
    elif i>0.05:
        up_5_y+=1
    elif i<0 and i>-0.001:
        down_0_00_y+=1 
    elif i<-0.001 and i>-0.01:
        down_00_1_y+=1 
    elif i<-0.01 and i>-0.05:
        down_1_5_y+=1 
    elif i<-0.05:
        down_5_y+=1 
    if i > 0:
        up_dir_y += 1
    else:
        down_dir_y += 1
        
pre_up_dir_y = 0
pre_down_dir_y = 0
pre_up_0_000_y=0
pre_up_000_00_y=0
pre_up_00_1_y=0
pre_up_1_5_y=0
pre_up_5_y=0
pre_down_0_000_y=0
pre_down_000_00_y=0
pre_down_00_1_y=0
pre_down_1_5_y=0
pre_down_5_y=0
for i in pred_y:
    if i>0 and i<0.0001:
        pre_up_0_000_y+=1
    elif i>0.0001 and i<0.001:
        pre_up_000_00_y+=1
    elif i>0.001 and i<0.01:
        pre_up_00_1_y+=1
    elif i>0.01 and i<0.05:
        pre_up_1_5_y+=1
    elif i>0.05:
        pre_up_5_y+=1
    elif i<0 and i>-0.0001:
        pre_down_0_000_y+=1 
    elif i<-0.0001 and i>-0.001:
        pre_down_000_00_y+=1 
    elif i<-0.001 and i>-0.01:
        pre_down_00_1_y+=1 
    elif i<-0.01 and i>-0.05:
        pre_down_1_5_y+=1 
    elif i<-0.05:
        pre_down_5_y+=1 
    if i > 0:
        pre_up_dir_y += 1
    else:
        pre_down_dir_y += 1
    


pd.DataFrame(pred_y).to_csv('pred_y_rf.csv',sep=',')
test_y_df.to_csv('test_y_df.csv',sep=',',header=None)
