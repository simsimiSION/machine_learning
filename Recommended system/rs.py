#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-7 下午11:26
# @File    : rs.py
# @Software: PyCharm

from surprise import KNNBasic,SVD
from surprise import Dataset
from surprise import evaluate, print_perf
import pandas as pd


"""
Recommended system
    describe:
        相似度计算：
            欧式距离
            皮尔逊相关系数
            余弦距离
        基于物品的协同过滤：算物品之间相关系数， 去掉相关系数较低的，属性×相关系数
        冷启动：物品
               用户 
        隐语义模型：
            p * q = r
            优化： r - p*q
            对 p q 分别求偏导数
            梯度优化 
        评估标准：
            覆盖率
            多样性
               

"""


##################################
# 协同过滤算法
# data = Dataset.load_builtin('ml-100k')
# data.split(n_folds=3)
#
# # 协同过滤算法
# algo = KNNBasic()
#
# # RMAE：Root Mean Square Error
# # MAE：Mean Absolute Error
# perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
#
# print_perf(perf)

##################################
# 隐语义模型
# from surprise import GridSearch
#
# param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
#               'reg_all': [0.4, 0.6]}
# grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])
# data = Dataset.load_builtin('ml-100k')
# data.split(n_folds=3)
#
# grid_search.evaluate(data)
#
# results_df = pd.DataFrame.from_dict(grid_search.cv_results)
# print(results_df)




###############################################
# 协同过滤实践
import os
import io

from surprise import KNNBaseline
from surprise import Dataset


def read_item_names():


    file_name = ('./surprise_data/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid



data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.train(trainset)

rid_to_name, name_to_rid = read_item_names()

toy_story_raw_id = name_to_rid['Now and Then (1995)']

toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in toy_story_neighbors)
toy_story_neighbors = (rid_to_name[rid]
                       for rid in toy_story_neighbors)

print()
print('The 10 nearest neighbors of Toy Story are:')
for movie in toy_story_neighbors:
    print(movie)






