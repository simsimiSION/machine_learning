#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-5 下午2:18
# @File    : ia.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier



SEED = 100
np.random.seed(SEED)

PATH = './data/science_federal_giving.csv'
df = pd.read_csv(PATH)

def get_train_test(test_size=0.95):
    y = 1 * (df.cand_pty_affiliation == "REP")
    X = df.drop(["cand_pty_affiliation"], axis=1)
    X = pd.get_dummies(X, sparse=True)
    X.drop(X.columns[X.std() == 0], axis=1, inplace=True)
    return train_test_split(X, y, test_size=test_size, random_state=True)

# 获得数据集
xtrain, xtest, ytrain, ytest = get_train_test()


def print_graph(clf, feature_name):
    graph = export_graphviz(clf,
                            label="root",
                            proportion=True,
                            impurity=False,
                            out_file=None,
                            feature_names=feature_name,
                            class_names={0:"D", 1:"R"},
                            filled=True,
                            rounded=True)

    graph = pydotplus.graph_from_dot_data(graph)
    return Image(graph.create_png())


###################################################################################
# 创建第一个决策树
t1 = DecisionTreeClassifier(max_depth=1, random_state=SEED)
t1.fit(xtrain, ytrain)
p = t1.predict_proba(xtest)[:,1]

print("score: %.3f" %roc_auc_score(ytest, p))
print_graph(t1, xtrain.columns)

# ###################################################################################
# # 创建第二个决策树
# t2 = DecisionTreeClassifier(max_depth=3, random_state=SEED)
# t2.fit(xtrain, ytrain)
# p = t2.predict_proba(xtest)[:,1]
#
# print("score: %.3f" %roc_auc_score(ytest, p))
# print_graph(t2, xtrain.columns)
#
# ###################################################################################
# # 去掉 一个特征
# drop = ["transaction_amt"]
# xtrain_slim = xtrain.drop(drop, 1)
# xtest_slim = xtest.drop(drop, 1)
#
# t3 = DecisionTreeClassifier(max_depth=3, random_state=SEED)
# t3.fit(xtrain_slim, ytrain)
# p = t3.predict_proba(xtest_slim)[:,1]
#
# print("score: %.3f" %roc_auc_score(ytest, p))
# print_graph(t3, xtrain_slim.columns)
#
# ###################################################################################
# # 整合
# p1 = t2.predict_proba(xtest)[:,1]
# p2 = t3.predict_proba(xtest_slim)[:,1]
# p = np.mean([p1, p2], axis=0)
#
# print("score: %.3f" %roc_auc_score(ytest, p))
#
#
# ###################################################################################
# # 随机森林
# rf = RandomForestClassifier(n_estimators=10,
#                             max_features=3,
#                             random_state=SEED)
# rf.fit(xtrain, ytrain)
# p = rf.predict_proba(xtest)[:,1]
# print("score: %.3f" %roc_auc_score(ytest, p))
#
# ###################################################################################
# # 一起上 多个各种的分类器
# from sklearn.svm import SVC, LinearSVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.kernel_approximation import Nystroem, RBFSampler
# from sklearn.pipeline import  make_pipeline
#
# def get_models():
#     nb = GaussianNB()
#     svc = SVC(C=100, probability=True)
#     knn = KNeighborsClassifier(n_neighbors=3)
#     lr = LogisticRegression(C=100, random_state=SEED)
#     nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
#     gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
#     rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)
#
#     models = {'svm': svc,
#               'knn': knn,
#               'naive bayes': nb,
#               'mlp-nn': nn,
#               'random forest': rf,
#               'gbm': gb,
#               'logistic': lr,}
#
#     return models
#
#
# def train_predict(model_list):
#     P = np.zeros((ytest.shape[0], len(model_list)))
#     P = pd.DataFrame(P)
#
#     cols = []
#     for i, (name, m) in enumerate(model_list.items()):
#         print("%s..." %name, end=" ", flush=False)
#
#         m.fit(xtrain, ytrain)
#         P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
#         cols.append(name)
#
#         print('done!!!')
#
#     P.columns = cols
#     return P
#
# def score_model(P, y):
#     print("Score models")
#     for m in P.columns:
#         score = roc_auc_score(y, P.loc[:, m])
#         print("%-26s: %.3f" %(m, score))
#     print("done！！")
#
#
# models = get_models()
# P = train_predict(models)
# score_model(P, ytest)
#
# print("Ensemble score:{}".format(roc_auc_score(ytest, P.mean(axis=1))))
#
# ####################
# # 显示
# from mlens.visualization import corrmat
# corrmat(P.corr(), inflate=False)
# plt.show()
#
# ###################
# # 绘制roc曲线
# from sklearn.metrics import roc_curve
#
# def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
#     plt.figure(figsize=(10, 8))
#     # 绘制中分线
#     plt.plot([0,1], [0,1], 'k--')
#
#     cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, P_base_learners.shape[1]+1)]
#
#     for i in range(P_base_learners.shape[1]):
#         p = P_base_learners[:, i]
#         fpr, tpr, _ = roc_curve(ytest, p)
#         plt.plot(fpr, tpr, label=labels[i], c=cm[i+1])
#
#     fpr, tpr, _ = roc_curve(ytest, P_ensemble)
#     plt.plot(fpr, tpr, label=ens_label, c=cm[0])
#
#     plt.xlabel("False positive rate")
#     plt.ylabel("True positive rate")
#     plt.title("ROC CURVE")
#     plt.legend(frameon=False)
#     plt.show()
#
#
# plot_roc_curve(ytest, P.values, P.mean(axis=1), list(P.columns), "ensemble")
#
#
#
#
# ###################################################################################
# # stack
# p = P.apply(lambda x: 1*(x >= 0.5).value_counts(normalize=True))
# p.index = ["DEM", "REP"]
# p.loc["REP",:].sort_values().plot(kind="bar")
# plt.axhline(0.25, color="k", linewidth=0.5)
# plt.text(0., 0.23, "True share republicans")
# plt.show()
#
#
# ################
# # stack model（可以使用交叉验证进行优化）
# base_learner = get_models()
#
# meta_learner = GradientBoostingClassifier(n_estimators=1000,
#                                           loss="exponential",
#                                           max_features=4,
#                                           max_depth=3,
#                                           subsample=0.5,
#                                           learning_rate=0.005,
#                                           random_state=SEED)
#
# xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(xtrain, ytrain, test_size=0.5, random_state=SEED)
#
#
# def train_base_learners(base_learners, inp, out, verbose=True):
#     if verbose:
#         print("Fitting models")
#
#     for i, (name, m) in enumerate(base_learners.items()):
#         if verbose:
#             print("%s..." %name, end=" ", flush=False)
#         m.fit(inp, out)
#
#         if verbose:
#             print("done!!!")
#
# # 训练
# train_base_learners(base_learner, xtrain_base, ytrain_base)
#
#
# def predict_base_learners(pred_learners, inp, verbose=True):
#     P = np.zeros((inp.shape[0], len(pred_learners)))
#
#     if verbose:
#         print("Generating base learner predictions")
#
#     for i, (name, m) in enumerate(pred_learners.items()):
#         if verbose:
#             print("%s..." %name, end=" ", flush=False)
#         p = m.predict_proba(inp)
#
#         P[:, i] = p[:, i]
#         if verbose:
#             print("done")
#
#     return P
#
# # 预测,生成第二层的数据
# P_base = predict_base_learners(base_learner, xpred_base)
#
# # 第二层训练
# meta_learner.fit(P_base, ypred_base)
#
# def ensemble_predict(base_learner, meta_learner, inp, verbose=True):
#     P_pred = predict_base_learners((base_learner, inp, verbose))
#     return P_pred, meta_learner.predict_proba(P_pred)[:, 1]
#
# # 生成预测结果
# P_pred, p = ensemble_predict(base_learner, meta_learner, xtest)
# print("\nEnsamble ROC-AUC score: %.3f" % roc_auc_score(ytest, p))
















