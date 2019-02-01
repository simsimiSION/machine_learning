#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-4 下午2:36
# @File    : lr.py
# @Software: PyCharm

import numpy as np
from sklearn import  linear_model, datasets
from sklearn.cross_validation import train_test_split

"""
linear regression
    describe:
        函数： y=k*x+bias
        假设： bias服从高斯分布

        即（y-k*x）f服从高斯分布，
            对（y-k*x）进行最大似然估计，
            得 J = sum(1/2 * (y_real - y)^2), 即使J最大.
        对 J 求偏导数，
            使dJ=0，
            得 k = (X.T * X)^(-1) * X.T * y

"""


"""
logic regression
    describe:
        sigmoid函数： h(x) = 1 / (1 + e^(-k*x))
        似然函数： L() = h(x)^(y) * (1-h(x))^(1-y)
        
        最大似然函数求偏导数（多维的，用对应的维度的进行计算）
        k = k - learning_rate * 1/m * sum(h(x)-y) * x
"""




# 1.加载数据
iris = datasets.load_iris()
X = iris.data[:, :2]  # 使用前两个特征
Y = iris.target
#np.unique(Y)   # out: array([0, 1, 2])


# 2.拆分测试集、训练集。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 设置随机数种子，以便比较结果。


# 3.标准化特征值
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

print(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 4. 训练逻辑回归模型
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train)

# 5. 预测
prepro = logreg.predict_proba(X_test_std)
acc = logreg.score(X_test_std,Y_test)








