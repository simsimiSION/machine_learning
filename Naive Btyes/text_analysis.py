#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-5 下午8:42
# @File    : text_analysis.py
# @Software: PyCharm

import pandas as pd
import jieba
import numpy as np

"""
Text Analysis
    describe:
        去除停顿词
        提取关键词
        
        逆文档频率：
            词频(TF) = 某个词在文章中出现次数 / 该文出现次数最多的词的出现次数
            逆文档频率(IDF) = log(语料库的文档总数 /  包含该词的文档数 + 1)
            TF-IDF = 词频(TF) * 逆文档频率(IDF)
        句子相似度：
            1. 句子清洗
            2. 分词
            3. 构造语料库
            4. 计算词频
            5. 构造词向量
"""

##############################
# 读取数据，去掉缺省项
df_news = pd.read_table('./data/val.txt',names=['category','theme','URL','content'],encoding='utf-8')
df_news = df_news.dropna()


##############################
# 分词
content = df_news.content.values.tolist()
content_S = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n': #换行符
        content_S.append(current_segment)

df_content = pd.DataFrame({'content_S':content_S})


##############################
# 读取停顿词, 去掉停顿词
def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words

stopwords = pd.read_csv("./data/stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')

contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

df_content = pd.DataFrame({'contents_clean':contents_clean})
df_all_words = pd.DataFrame({'all_words':all_words})


##############################
# 计算单词出现次数
words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":np.size})
words_count = words_count.reset_index().sort_values(by=["count"],ascending=False)


##############################
# TF-IDF ：提取关键词
import jieba.analyse
index = 2400
print (df_news['content'][index])
content_S_str = "".join(content_S[index])
print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))


##############################
# LDA 参考ipython文档





























