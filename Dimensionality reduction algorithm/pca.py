#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-7 下午2:37
# @File    : pca.py
# @Software: PyCharm

from sklearn.decomposition import PCA


data = [[ 1.  ,  1.  ],
       [ 0.9 ,  0.95],
       [ 1.01,  1.03],
       [ 2.  ,  2.  ],
       [ 2.03,  2.06],
       [ 1.98,  1.89],
       [ 3.  ,  3.  ],
       [ 3.03,  3.05],
       [ 2.89,  3.1 ],
       [ 4.  ,  4.  ],
       [ 4.06,  4.02],
       [ 3.97,  4.01]]

A = data

# 训练降维model
pca=PCA(n_components=1)
newData=pca.fit_transform(data)
print(newData)

# 对新数据进行降维
newA = pca.transform(A)
print(newA)

