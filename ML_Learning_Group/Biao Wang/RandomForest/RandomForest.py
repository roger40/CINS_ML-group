#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/5
# @Author  : Wang Biao
# @Site    :
# @File    : CalculateK.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import matplotlib.pyplot as plt


'''
用随机森林对数据进行分类任务
'''

_746Data = '../LogisticRegression/newHIV-1_data/746Data.txt'
_1625Data = '../LogisticRegression/newHIV-1_data/1625Data.txt'
_impensData = '../LogisticRegression/newHIV-1_data/impensData.txt'
_schillingData = '../LogisticRegression/newHIV-1_data/schillingData.txt'
_x_all = 'ARNDCQEGHILKMFPSTWYV'

# 提前对上述字母进行编码
_x_list = []
for i in range(20):
    _x_list.append(_x_all[i])

_x_array = np.array(_x_list).reshape(len(_x_list), 1)
enc = OneHotEncoder()
enc.fit(_x_array)
x_all = enc.transform(_x_array).toarray()
# 每一行对应着每一个编码
# print(x_all)
# print(x_all[0])
# print(x_all[1])
# exit()

class RandomForest(object):
    def loadData(self, data_path):
        # 用那个基因数据集
        data = pd.read_csv(data_path, header=None)
        X_train = data.values[:, 0]
        Y_train = data.values[:, 1]
        X_train_list = []
        # 需要对属性进行数值化
        for i in range(len(X_train)):
            # X_train_list.append([a for a in X_train[i]])
            tempoary_list = [x_all[_x_all.index(a)] for a in X_train[i]]
            # print(tempoary_list)
            other_list = []
            for j in tempoary_list:
                other_list.append(list(j))
            tempoary_list = sum(other_list, [])
            X_train_list.append(tempoary_list)
        X_train = np.array(X_train_list)
        for i in range(len(Y_train)):
            if Y_train[i] == -1:
                Y_train[i] = 0
        Y_train = np.array(Y_train)
        # print(np.hstack((X_train, Y_train.reshape(len(Y_train), 1))))
        return X_train, list(Y_train)

    def randomForest(self, train_data, test_data_1, test_data_2, test_data_3):
        X, Y = self.loadData(train_data)
        X_test_1, Y_test_1 = self.loadData(test_data_1)
        X_test_2, Y_test_2 = self.loadData(test_data_2)
        X_test_3, Y_test_3 = self.loadData(test_data_3)
        X_train, X_validate, Y_train, Y_validate = \
            train_test_split(X, Y, test_size=1 / 3., random_state=1)

        clf = RandomForestClassifier(n_estimators='warn', oob_score=False,
                                     criterion='gini', max_features='auto',
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, min_weight_fraction_leaf=0,
                                     max_leaf_nodes=None, class_weight=None,
                                     min_impurity_decrease=0, min_impurity_split=None,
                                     bootstrap=True)



        clf = clf.fit(X_train, Y_train)
        train_score, validate_score, test_score_1, test_score_2, test_score_3 = \
                                                    clf.score(X_train, Y_train), \
                                                    clf.score(X_validate, Y_validate), \
                                                    clf.score(X_test_1, Y_test_1),\
                                                    clf.score(X_test_2, Y_test_2),\
                                                    clf.score(X_test_3, Y_test_3)
        print('train score:', train_score)
        print('validate score:', validate_score)
        print('test score 1:', test_score_1)
        print('test score 2:', test_score_2)
        print('test score 3:', test_score_3)
        return train_score, validate_score, test_score_1, test_score_2, test_score_3

    # 使用袋外样本来考虑模型好坏
    def randomForest_2(self, train_data, test_data_1, test_data_2, test_data_3):
        X, Y = self.loadData(train_data)
        X_test_1, Y_test_1 = self.loadData(test_data_1)
        X_test_2, Y_test_2 = self.loadData(test_data_2)
        X_test_3, Y_test_3 = self.loadData(test_data_3)


        clf = RandomForestClassifier(oob_score=True, class_weight='balanced')

        clf = clf.fit(X, Y)
        train_score, oob_score, test_score_1, test_score_2, test_score_3 = \
            clf.score(X, Y), \
            clf.oob_score_, \
            clf.score(X_test_1, Y_test_1), \
            clf.score(X_test_2, Y_test_2), \
            clf.score(X_test_3, Y_test_3)
        print('train score:', train_score)
        print('oob score:', oob_score)
        print('test score 1:', test_score_1)
        print('test score 2:', test_score_2)
        print('test score 3:', test_score_3)
        return train_score, oob_score, test_score_1, test_score_2, test_score_3

    def optimalEstimators(self, n, train_data, test_data_1, test_data_2, test_data_3):
        X, Y = self.loadData(train_data)
        X_test_1, Y_test_1 = self.loadData(test_data_1)
        X_test_2, Y_test_2 = self.loadData(test_data_2)
        X_test_3, Y_test_3 = self.loadData(test_data_3)

        clf = RandomForestClassifier(oob_score=True, class_weight='balanced', n_estimators=n, random_state=1)

        clf = clf.fit(X, Y)
        train_score, oob_score, test_score_1, test_score_2, test_score_3 = \
            clf.score(X, Y), \
            clf.oob_score_, \
            clf.score(X_test_1, Y_test_1), \
            clf.score(X_test_2, Y_test_2), \
            clf.score(X_test_3, Y_test_3)
        print('n', n)
        print('train score:', train_score)
        print('oob score:', oob_score)
        print('test score 1:', test_score_1)
        print('test score 2:', test_score_2)
        print('test score 3:', test_score_3)
        print('----------')
        return train_score, oob_score

    def optimalsplit_score(self, score, train_data, test_data_1, test_data_2, test_data_3):
        X, Y = self.loadData(train_data)
        X_test_1, Y_test_1 = self.loadData(test_data_1)
        X_test_2, Y_test_2 = self.loadData(test_data_2)
        X_test_3, Y_test_3 = self.loadData(test_data_3)

        clf = RandomForestClassifier(oob_score=True, class_weight='balanced',
                                     n_estimators=20, random_state=1,
                                     min_impurity_decrease=score)
        clf = clf.fit(X, Y)
        train_score, oob_score, test_score_1, test_score_2, test_score_3 = \
            clf.score(X, Y), \
            clf.oob_score_, \
            clf.score(X_test_1, Y_test_1), \
            clf.score(X_test_2, Y_test_2), \
            clf.score(X_test_3, Y_test_3)
        print('min_impurity_decrease:', score)
        print('train score:', train_score)
        print('oob score:', oob_score)
        print('test score 1:', test_score_1)
        print('test score 2:', test_score_2)
        print('test score 3:', test_score_3)
        print('----------')
        return train_score, oob_score

    def visualization(self, train_data, test_data_1, test_data_2, test_data_3):
        # depths = np.arange(1, 11)
        # scores = [self.maxdepth_score(d, train_data, test_data_1, test_data_2, test_data_3) for d in depths]
        thresholds = np.linspace(0, 0.2, 101)
        scores = [self.optimalsplit_score(threshold, train_data, test_data_1, test_data_2, test_data_3) for threshold in
                  thresholds]

        # estimators_nums = np.arange(2, 51)
        # scores = [self.optimalEstimators(n, train_data, test_data_1, test_data_2, test_data_3) for n in estimators_nums]

        train_scores = [s[0] for s in scores]
        oob_scores = [s[1] for s in scores]
        plt.figure(111)
        # plt.grid()
        # plt.xlabel('max depth of decision tree')
        # plt.xlabel('optimal estimators of decision tree')
        plt.xlabel('optimal min_impurity_decrease of decision tree')
        plt.ylabel('scores')
        plt.plot(thresholds, train_scores, label='train_score', c='red')
        plt.plot(thresholds, oob_scores, label='oob_score', c='black')
        plt.legend()
        plt.show()
        return

if __name__ == '__main__':
    rf = RandomForest()

    rf.visualization(train_data=_746Data, test_data_1=_1625Data, test_data_2=_impensData, test_data_3=_schillingData)
    exit()

    rf.randomForest_2(train_data=_746Data, test_data_1=_1625Data, test_data_2=_impensData, test_data_3=_schillingData)
    print('-----------')
    rf.randomForest_2(train_data=_1625Data, test_data_1=_746Data, test_data_2=_impensData, test_data_3=_schillingData)
    print('-----------')
    rf.randomForest_2(train_data=_impensData, test_data_1=_1625Data, test_data_2=_746Data, test_data_3=_schillingData)
    print('-----------')
    rf.randomForest_2(train_data=_schillingData, test_data_1=_1625Data, test_data_2=_746Data, test_data_3=_impensData)
    print('-----------')