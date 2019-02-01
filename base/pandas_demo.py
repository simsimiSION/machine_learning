#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-3 下午9:59
# @File    : pandas_demo.py
# @Software: PyCharm


import pandas as pd
import numpy as np
from matplotlib import  pyplot as plt

########################################################
# Series
########################################################
# # 读取一维数据
# t = pd.Series([1,2,3])
#
# temp_dict = {'1':1, 'a':2, '!':3}
# t1 = pd.Series(temp_dict)
# # 索引
# index = t1.index

# # 读取csv文件
# df = pd.read_csv("./data/dogNames2.csv")
# df = df.sort_values(by='Count_AnimalName', ascending=False)
# print(df)


########################################################
# DataFrame
########################################################
# 读取二维数据
# t = pd.DataFrame(np.arange(12).reshape(3,4), index=list("ABC"), columns=list("QWER"))

# d1 = {"name":["simi", "panda"], "age":[18, 17], "tel":[10086, 10010]} # 列表里面有字典也可以
# t1 = pd.DataFrame(d1)
#
# print(t1)

# # 获取信息
# print(t1.info())
# # 获取描述
# print(t1.describe())
# # 获得值
# print(t1.loc[1])

file_path = "./data/PM2.5/BeijingPM20100101_20151231.csv"

df = pd.read_csv(file_path)

#把分开的时间字符串通过periodIndex的方法转化为pandas的时间类型
period = pd.PeriodIndex(year=df["year"],month=df["month"],day=df["day"],hour=df["hour"],freq="H")
df["datetime"] = period
# print(df.head(10))

#把datetime 设置为索引
df.set_index("datetime",inplace=True)

#进行降采样
df = df.resample("7D").mean()
print(df.head())
#处理缺失数据，删除缺失数据
# print(df["PM_US Post"])

data  =df["PM_US Post"]
data_china = df["PM_Nongzhanguan"]

print(data_china.head(100))
#画图
_x = data.index
_x = [i.strftime("%Y%m%d") for i in _x]
_x_china = [i.strftime("%Y%m%d") for i in data_china.index]
print(len(_x_china),len(_x_china))
_y = data.values
_y_china = data_china.values


plt.figure(figsize=(20,8),dpi=80)

plt.plot(range(len(_x)),_y,label="US_POST",alpha=0.7)
plt.plot(range(len(_x_china)),_y_china,label="CN_POST",alpha=0.7)

plt.xticks(range(0,len(_x_china),10),list(_x_china)[::10],rotation=45)

plt.legend(loc="best")

plt.show()





