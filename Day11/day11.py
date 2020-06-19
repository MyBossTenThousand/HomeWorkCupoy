# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:46:19 2020

@author: psycho
"""

# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 設定 data_path
dir_data = './data/'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
print(app_train.head())


# 1: 計算 AMT_ANNUITY 的 q0 - q100
q_all =[np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in range(101)]

q0_q100=pd.DataFrame({'q': list(range(101)),
              'value': q_all})
    
# 2.1 將 NAs 以 q50 填補
print("Before replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

q_50 = q_all[50]
print(q_50)
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50

print("After replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

# 2.2 Normalize values to -1 to 1
print("== Original data range ==")
print(app_train['AMT_ANNUITY'].describe())




from sklearn import preprocessing

def normalize_value(x):
    n_x=x.copy()
    n_x=n_x.values
    n_x=n_x.reshape(-1, 1)
    print(n_x)
    print(type(n_x))
    maxx=max(n_x)
    minx=min(n_x)
    print(maxx,minx)
    normal=preprocessing.MinMaxScaler()
    n_x=normal.fit_transform(n_x)
    n_x=2*n_x-1
    
    print(n_x)
    return n_x

app_train['AMT_ANNUITY_NORMALIZED'] = normalize_value(app_train['AMT_ANNUITY'])

print("== Normalized data range ==")
print(app_train['AMT_ANNUITY_NORMALIZED'].describe())

# 3
print("Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))

# 列出重複最多的數值
from scipy.stats import mode

mode_get = mode(app_train[~app_train['AMT_GOODS_PRICE'].isnull()]['AMT_GOODS_PRICE'])
print(mode_get)


value_most = mode_get
print(value_most)

mode_goods_price = list(app_train['AMT_GOODS_PRICE'].value_counts().index)
app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(), 'AMT_GOODS_PRICE'] = mode_goods_price[0]

print("After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))
