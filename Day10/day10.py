# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:38:31 2020

@author: psycho
"""
#作業1
print("作業1")
# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

data_path = 'data/'
df_train = pd.read_csv(data_path + 'house_train.csv.gz')

train_Y = np.log1p(df_train['SalePrice'])
df = df_train.drop(['Id', 'SalePrice'] , axis=1)
print(df.head())

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
print(df.head())

# 顯示 GrLivArea 與目標值的散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x = df['GrLivArea'], y=train_Y)
plt.show()

# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

print("將 GrLivArea 限制在 800 到 2500 以內, 調整離群值")
# 將 GrLivArea 限制在 800 到 2500 以內, 調整離群值
df['GrLivArea'] = df['GrLivArea'].clip(800, 2500)
sns.regplot(x = df['GrLivArea'], y=train_Y)
plt.show()

# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

print("將 GrLivArea 限制在 800 到 2500 以內, 捨棄離群值")
# 將 GrLivArea 限制在 800 到 2500 以內, 捨棄離群值
keep_indexs = (df['GrLivArea']> 800) & (df['GrLivArea']< 2500)
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]
sns.regplot(x = df['GrLivArea'], y=train_Y)
plt.show()

# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

print("將 1stFlrSF 限制在 800 到 2500 以內, 調整離群值")
# 將 1stFlrSF 限制在 800 到 2500 以內, 調整離群值
df['1stFlrSF'] = df['1stFlrSF'].clip(500, 2200)
sns.regplot(x = df['1stFlrSF'], y=train_Y)
plt.show()

# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

print("將 1stFlrSF 限制在 500 到 2000 以內, 捨棄離群值")
# 將 1stFlrSF 限制在 500 到 2000 以內, 捨棄離群值
keep_indexs = (df['1stFlrSF']> 800) & (df['1stFlrSF']< 2500)
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]
sns.regplot(x = df['1stFlrSF'], y=train_Y)
plt.show()

# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())



#作業2
print("作業2")
print("""捨棄離群值讓訓練時，不會因為離群質影響到回歸線的狀態。
      讓線條更加符合多數資料，使準確率有所提升。""")