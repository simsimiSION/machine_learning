#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-7 下午10:58
# @File    : gmm.py
# @Software: PyCharm

# 将相同分布的分成一类

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

data = pd.read_csv ('./data/Fremont.csv', index_col='Date', parse_dates=True)

##########################
# 显示一下
data.groupby(data.index.time).mean().plot()
plt.xticks(rotation=45)
plt.show()

data.columns =['West', 'East']
data ['Total'] =data['West']+data['East']
pivoted = data.pivot_table('Total', index=data.index.time, columns=data.index.date)

pivoted.plot(legend=False, alpha=0.01)
plt.xticks(rotation=45)
plt.show()


##########################
# 通过每天24小时的信息作为特征进行分类

# 用pca试下
X = pivoted.fillna(0).T.values
X2 = PCA(2).fit_transform(X)

plt.scatter(X2[:,0],X2[:,1])
plt.show()

# gmm model
gmm = GaussianMixture(2)
gmm.fit(X)
labels = gmm.predict(X)

plt.scatter(X2[:,0],X2[:,1], c=labels, cmap='rainbow')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
pivoted.T[labels == 0].T.plot(legend=False, alpha=0.1, ax=ax[0])
pivoted.T[labels == 1].T.plot(legend=False, alpha=0.1, ax=ax[1])
ax[0].set_title('Purple Cluster')
ax[1].set_title('Red Cluster')
plt.show()




























