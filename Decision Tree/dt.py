#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-4 下午6:12
# @File    : dt.py
# @Software: PyCharm


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

"""
Decision Tree
    describe:
        熵：H(x) = - sum(pi * log(pi))

        id3  ：对每个类进行信息增益计算的，排序，大的优先
        c4.5 : 相对于id3 考虑到自身熵
        CART ： 使用GINI系数进行比较
                Gini(p) = 1 - sum(p^2)

        剪枝：预剪枝：限制深度，叶子节点个数，叶子节点样本数，信息增益量
             后剪枝： C_alpha(T) = C(T) + alpha*abs(T_leaf)
"""

"""
Random Forest
    describe:
        bagging： 并行训练一堆分类器，投票
        boosting：f_m(x) = f_m-1(x) + argmin(sum( L(y_i, f_m-1(x)+h ) ))
            adaboost: 分类错误的获得更大的权重
        stacking： 堆叠分类器 串联堆叠，并联堆叠
                将多个分类的输出结果做输入，进行第二次分类，
                第一层单独训练，第二层用新的数据进行训练。可以永cv进行优化

"""


housing = fetch_california_housing()

#######################################################
# demo 1
#######################################################
#
# dtr = tree.DecisionTreeRegressor(max_depth = 2)
# dtr.fit(housing.data[:, [6, 7]], housing.target)
#
# dot_data = tree.export_graphviz(
#         dtr,
#         out_file = None,
#         feature_names = housing.feature_names[6:8],
#         filled = True,
#         impurity = False,
#         rounded = True)
#
#
# graph = pydotplus.graph_from_dot_data(dot_data)
#
# graph.get_nodes()[7].set_fillcolor("#FFF2DD")
# graph.write_png("dtr_white_background.png")



#######################################################
# demo 2
#######################################################
#
# data_train, data_test, target_train, target_test = \
#     train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)
# dtr = tree.DecisionTreeRegressor(random_state = 42)
# dtr.fit(data_train, target_train)
#
# print(dtr.score(data_test, target_test))


#######################################################
# demo 3 交叉验证
#######################################################

data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)
tree_param_grid = { 'min_samples_split': list((3,6,9)),'n_estimators':list((10,50,100))}
grid = GridSearchCV(RandomForestRegressor(),param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print(grid.grid_scores_, grid.best_params_, grid.best_score_)



#######################################################
# 随机森林
#######################################################
#
# data_train, data_test, target_train, target_test = \
#     train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)
# rfr = RandomForestRegressor( random_state = 42)
# rfr.fit(data_train, target_train)
#
# print(rfr.score(data_test, target_test))