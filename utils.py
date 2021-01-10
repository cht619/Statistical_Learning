# -*- coding: utf-8 -*-
# @Time : 2021/1/10 9:23
# @Author : CHT
# @Site : 
# @File : utils.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

import numpy as np
import heapq
from matplotlib import pyplot as plt
from math import inf, nan, log, sqrt
from collections import Counter


# Functions
def argmax(arr, key=lambda x: x):
    arr = [key(a) for a in arr]
    ans = max(arr)
    return arr.index(ans), ans



# Decision Tree
def entropy(p):
    s = sum(p)
    p = [i/s for i in p]
    ans = sum(-i*log(i) for i in p)
    return ans


def entropy_of_split(X, Y, col):
    # calculate the conditional entropy of splitting data by col
    val_cnt = Counter(x[col] for x in X)
    ans = 0
    for val in val_cnt:
        weight = val_cnt[val] / len(X)
        entropy_ = entropy(Counter(y for x, y in zip(X, Y) if x[col] == val).values())
        ans += weight * entropy_
    return ans


def information_gain(X, Y, col):
    # 信息增益 = 信息熵 - 条件熵
    entropy_of_X = entropy(Counter(Y).values())
    entropy_of_col = entropy_of_split(X, Y, col)
    return entropy_of_X - entropy_of_col

