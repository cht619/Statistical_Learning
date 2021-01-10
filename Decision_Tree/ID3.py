# -*- coding: utf-8 -*-
# @Time : 2021/1/10 9:08
# @Author : CHT
# @Site : 
# @File : ID3.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

from pprint import pprint
from rich.console import Console
from rich.table import Table
from collections import Counter
import sys
import os
from pathlib import Path
from utils import argmax, information_gain
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))

print(str(Path(os.path.abspath(__file__)).parent.parent))


class ID3:
    class Node:
        def __init__(self, col, Y):
            self.col = col
            self.children = {}
            self.prob = Counter(Y)
            s = sum(self.prob.values())
            for y in self.prob:
                # Standardization
                self.prob[y] /= s
            label_ind, self.label_prob = argmax(self.prob.keys(), key=self.prob.__getitem__)
            self.labels = Y[label_ind]

    def __init__(self, information_gain_threshold=0, verbose=False):
        self.information_gain_threshold = information_gain_threshold
        self.verbose = verbose

    def build(self, X, Y, selected):
        cur = self.Node(None, Y)
        if self.verbose:
            print('Current selected columns:{}'.format(selected))
            print('Current data:')
            pprint(X)
            print(Y)
        split = False

        # check if there is no attribute to choose or there is no need for spilt
        if len(selected) != self.column_cnt and len(set(Y)) > 1:
            left_columns = list(set(range(self.column_cnt)) - selected)
            col_ind, best_information_gain = argmax(left_columns,
                                                    key=lambda col: information_gain(X, Y, col))
            col = left_columns[col_ind]
            # if this split is better than not splitting
            if best_information_gain >= self.information_gain_threshold:
                print('Split by {}th column'.format(col))
                split = True
                cur.col = col
                for val in set(x[col] for x in X):
                    ind = [x[col] == val for x in X]
                    child_X = [x for i, x in zip(ind, X) if i]
                    child_Y = [y for i, y in zip(ind, Y) if i]
                    cur.children[val] = self.build(child_X, child_Y, selected | {col})

        if not split:
            print('No split')
        return cur

    def query(self, root, x):
        if root.col is None or x[root.col] not in root.children:
            return root.label

    def fit(self, X, Y):
        self.column_cnt = len(X[0])
        self.root = self.build(X, Y, set())

    def _predict(self, x):
        # can't use for from xxx import *
        return self.query(self.root, x)

    def predict(self, X):
        return [self._predict(x) for x in X]


if __name__ == '__main__':
    console = Console(markup=False)
    id3 = ID3(verbose=True)
    print("Example 1:")
    X = [
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '是', '否', '好'],
        ['老年', '是', '否', '非常好'],
        ['老年', '否', '否', '一般'],
    ]
    Y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    id3.fit(X, Y)
    pred = id3.predict(X)
    print(pred)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)