#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-6 下午2:38
# @File    : dbascn.py
# @Software: PyCharm

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
DBSCN
    describe:
       核心对象： 某个点的密度达到阈值(minPts)则为核心点
       邻域阈值(r)

       传销算法


"""

colors = np.array(['red', 'green', 'blue', 'yellow'])

# 读取数据
beer = pd.read_csv('./data/data.txt', sep=' ')
X = beer[["calories","sodium","alcohol","cost"]]

# dbscan
db = DBSCAN(eps=10, min_samples=2).fit(X)

beer['cluster_db'] = db.labels_
beer.groupby('cluster_db').mean()

pd.scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10,10), s=100)
plt.show()

