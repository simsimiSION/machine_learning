#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-6 下午12:49
# @File    : kmeans.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

"""
KMEANS
    describe:
        首先给定聚的堆数
        找质心 
        优化距离        
        
 
"""


########################
# 读取数据
beer = pd.read_csv('./data/data.txt', sep=' ')
X = beer[["calories","sodium","alcohol","cost"]]

# 归一化
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

########################
# kmeans运算
km = KMeans(n_clusters=3).fit(X)
beer['cluster'] = km.labels_

cluster_centers = km.cluster_centers_
centers = beer.groupby("cluster").mean().reset_index()

#########################
# 显示
plt.rcParams['font.size'] = 14
colors = np.array(['red', 'green', 'blue', 'yellow'])
plt.scatter(beer["calories"], beer["alcohol"],c=colors[beer["cluster"]])

plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')

plt.xlabel("Calories")
plt.ylabel("Alcohol")
plt.show()


#########################
# 轮廓系数 si->1 ：合理， si->-1 ：不合理
from sklearn import metrics

scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)

plt.plot(list(range(2,20)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")
plt.show()



