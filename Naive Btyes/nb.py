#!  anaconda3.6
# -*- coding:utf-8 -*-
# Author   : Simsimi
# @Time    : 19-1-5 下午6:01
# @File    : nb.py
# @Software: PyCharm

import re, collections



"""
Naive Bytes
    describe:
        贝叶斯变换： P(H|D) = P(H) * P(D|H) / P(D)
        奥卡姆剃刀： 越常见的越容易出现
        
        垃圾邮件：
            P(h+|D) = P(h+) * P(D|h+) / P(D)
            P(h-|D) = P(h-) * P(D|h-) / P(D)
            两个相比，取log后进行判断
            
        朴素贝叶斯： 假设特征之间是独立的
            P(d1|h+) * P(d2|d1,h+) * P(d3|d2,d1,h+) = P(d1|h+) * P(d2|h+) * P(d3|h+)
"""



####################################################################
# 拼写检查

# 对单词进行编辑
alphabet = 'abcdefghijklmnopqrstuvwxyz'
PATH = "./data/big.txt"

# 从文档中获取单词
def words(text):
    return re.findall('[a-z]+', text.lower())

# 生成单词库
def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

# 单词经过一次编辑在单词库中
def edits1(word):
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion

# 单词经过两次编辑在单词库中
def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

# 单词在单词库中
def known(words):
    return set(w for w in words if w in NWORDS)

# 检查
def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=lambda w: NWORDS[w])



# 获得词频
NWORDS = train(words(open(PATH).read()))
print(correct('tha'))
