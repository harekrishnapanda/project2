# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:51:22 2018

@author: Harekrishna
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

data = pd.read_csv('data_stocks.csv')
data.head()

# Data Preprocessing
# Drop date and S&P variable since these columns are not stock price columns and wont be usefull
data = data.drop(['DATE'], 1)
data = data.drop(['SP500'], 1)

# Creating the co_relation matrix of each stock prices
df = data.corr()
# plot the co-Relation matrix
plt.matshow(df)

# dropping the values above diagonal values as these are duplicate values of below Diagonal values
v = df.values
i, j = np.tril_indices_from(v, -1)
df1 = pd.Series(v[i, j], [df.index[i], df.columns[j]])
df1= pd.DataFrame(df1)
# Renaming the column name of df1 dataframe
df1.columns = ['Corr_Score']

# considering two stocks are similar if Co-Relation value is more than 90%
df2 = df1[(df1.Corr_Score >= 0.9)]
df2.reset_index(level=0, inplace=True)
df2.reset_index(level=0, inplace=True)
df2.columns = ['stock1','stock2','Corr_Score']
print('The list of similar Performing stocks'+ df2.stock1 + '&'+ df2.stock2)

df4 = df2.sort_values(by='stock1', ascending=True)
# creating the list of simlar stocks 
df3 = pd.DataFrame((df2.stock1.append(df2.stock2).reset_index(drop=True)).unique())
df3.columns = ['Similar_Stocks']

df4= df4.drop(['Corr_Score'],1)



for val_st1 in df4.stock1:
    indx = 0
    for val_st2 in df4.stock2:
        #print("comparing " +val_st2+" and "+val_st1)   
        if val_st1==val_st2:
            print(indx)
            print("index = "+str(indx)+"comparing " +val_st2+" and "+val_st1)
            df4.drop(index=df.index[indx],inplace=True)
        indx+=1

    

df8 = df4#.groupby('stock1').groups
df8.groupby('stock1').size()
df8.stock1.unique(return_counts=True)


df5 = df1[(df1.Corr_Score < 0.9)]
df5.reset_index(level=0, inplace=True)
df5.reset_index(level=0, inplace=True)
df5.columns = ['stock1','stock2','Corr_Score']
#df5 = pd.DataFrame((df4.stock1.append(df4.stock2).reset_index(drop=True)).unique())
#df5.columns = ['Unique_Stocks']

df.reset_index(level=0, inplace=True)
df6 = df.iloc[:,0].tolist()
df7 = df3['Similar_Stocks'].tolist()

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 
  # Driver Code 
df7 =Diff(df6, df3)
df7 = pd.DataFrame(df7)
df7.columns = ['Unique_Stocks']


'''

df1 = data.iloc[:,0:30]
data.corr( [data.corr > 0.9],2,0)
df1 = data.corr(method = 'spearman')


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import numpy
def plot_corr(df,X=100, Y=100):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(X, Y))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);

plot_corr(data)

df2= df2.drop(['DATE'],axis=1)
df2= df2.drop(['DATE'],axis=0)
df2[df2<0.9] = 0
df2[df2==1] = 0
df2['max_value'] = df2.max(axis=1)
df2['max_value'] = df2.max(axis=0)

#NASDAQ.DISCA, NASDAQ.DISCK
plt.scatter(data['NASDAQ.DISCA'], data['NASDAQ.DISCK'])
plt.show()
data2= data.transform()
data.T.plot(figsize=(15,20), subplots=True,layout=(101,5),axis=1)
'''
#data.plot(figsize=(20,10), linewidth=5, fontsize=20)
#plt.xlabel('Stock', fontsize=20);

data1 = data.iloc[:,1:51]
data1.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data2 = data.iloc[:,51:101]
data2.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data3 = data.iloc[:,101:151]
data3.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data4 = data.iloc[:,151:201]
data4.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data5 = data.iloc[:,201:251]
data5.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data6 = data.iloc[:,251:301]
data6.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data7 = data.iloc[:,301:351]
data7.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data8 = data.iloc[:,351:401]
data8.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data9 = data.iloc[:,401:451]
data9.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);

data10 = data.iloc[:,451:500]
data10.plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.xlabel('Stock', fontsize=20);
'''
