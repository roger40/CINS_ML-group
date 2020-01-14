# -*- coding: utf-8 -*-
# @Time    : 2019/1/6
# @Author  : Wang Biao
# @Site    :
# @File    : EnsembleLearning.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


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

class EnsembleLearning(object):
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
        # print(X_train)
        # print(Y_train)
        return X_train, list(Y_train)

    def adaBoost(self, train_data, test_data_1, test_data_2, test_data_3):
        X, Y = self.loadData(train_data)
        X_test_1, Y_test_1 = self.loadData(test_data_1)
        X_test_2, Y_test_2 = self.loadData(test_data_2)
        X_test_3, Y_test_3 = self.loadData(test_data_3)
        X_train, X_validate, Y_train, Y_validate = \
            train_test_split(X, Y, test_size=1 / 3., random_state=1)
        # clf = AdaBoostClassifier(n_estimators=10, learning_rate=0.01, algorithm='SAMME.R')
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(criterion='gini', max_depth=10,
                                   min_impurity_decrease=0.002,
                                   min_samples_split=4,
                                   min_samples_leaf=8,
                                   max_features=0.8),
            n_estimators=20, learning_rate=0.02, algorithm='SAMME.R')
        # clf = AdaBoostClassifier(
        #     DecisionTreeClassifier(),
        #     n_estimators=10, learning_rate=0.01, algorithm='SAMME.R')


        clf = clf.fit(X_train, Y_train)
        train_score, validate_score, test_score_1, test_score_2, test_score_3\
            = clf.score(X_train, Y_train), \
              clf.score(X_validate, Y_validate), \
              clf.score(X_test_1, Y_test_1), \
              clf.score(X_test_2, Y_test_2), \
              clf.score(X_test_3, Y_test_3)
        Y_train_pre = clf.predict(X_train)
        Y_validate_pre = clf.predict(X_validate)
        Y_test_1_pre = clf.predict(X_test_1)
        Y_test_2_pre = clf.predict(X_test_2)
        Y_test_3_pre = clf.predict(X_test_3)

        print('train score:', train_score)
        # print(classification_report(Y_train, Y_train_pre))
        print('validate score:', validate_score)
        # print(classification_report(Y_validate, Y_validate_pre))
        print('test score 1:', test_score_1)
        # print(classification_report(Y_test_1, Y_test_1_pre))
        print('test score 2:', test_score_2)
        # print(classification_report(Y_test_2, Y_test_2_pre))
        print('test score 3:', test_score_3)
        # print(classification_report(Y_test_3, Y_test_3_pre))
        return

    def optimalEstimators(self, n, train_data, test_data_1, test_data_2, test_data_3):
        X, Y = self.loadData(train_data)
        X_test_1, Y_test_1 = self.loadData(test_data_1)
        X_test_2, Y_test_2 = self.loadData(test_data_2)
        X_test_3, Y_test_3 = self.loadData(test_data_3)
        X_train, X_validate, Y_train, Y_validate = \
            train_test_split(X, Y, test_size=1 / 3., random_state=1)
        '''
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=10, min_samples_split=8,
                                   min_samples_leaf=4, max_features=0.8,
                                   min_impurity_decrease=0.002),
            n_estimators=n, learning_rate=0.02, algorithm='SAMME.R')
        '''

        clf = AdaBoostClassifier(n_estimators=n)
        clf = clf.fit(X_train, Y_train)
        train_score, validate_score, test_score_1, test_score_2, test_score_3 \
            = clf.score(X_train, Y_train), \
              clf.score(X_validate, Y_validate), \
              clf.score(X_test_1, Y_test_1), \
              clf.score(X_test_2, Y_test_2), \
              clf.score(X_test_3, Y_test_3)
        print('n', n)
        print('train score:', train_score)
        print('validate score:', validate_score)
        print('test score 1:', test_score_1)
        print('test score 2:', test_score_2)
        print('test score 3:', test_score_3)
        print('----------')
        return train_score, validate_score

    def optimalLearningrate(self, learning_rate, train_data, test_data_1, test_data_2, test_data_3):
        X, Y = self.loadData(train_data)
        X_test_1, Y_test_1 = self.loadData(test_data_1)
        X_test_2, Y_test_2 = self.loadData(test_data_2)
        X_test_3, Y_test_3 = self.loadData(test_data_3)
        X_train, X_validate, Y_train, Y_validate = \
            train_test_split(X, Y, test_size=1 / 3., random_state=1)
        '''
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=10, min_samples_split=8,
                                   min_samples_leaf=4, max_features=0.8,
                                   min_impurity_decrease=0.002),
            n_estimators=n, learning_rate=0.02, algorithm='SAMME.R')
        '''

        clf = AdaBoostClassifier(n_estimators=30, learning_rate=learning_rate)
        clf = clf.fit(X_train, Y_train)
        train_score, validate_score, test_score_1, test_score_2, test_score_3 \
            = clf.score(X_train, Y_train), \
              clf.score(X_validate, Y_validate), \
              clf.score(X_test_1, Y_test_1), \
              clf.score(X_test_2, Y_test_2), \
              clf.score(X_test_3, Y_test_3)
        print('learning rate', learning_rate)
        print('train score:', train_score)
        print('validate score:', validate_score)
        print('test score 1:', test_score_1)
        print('test score 2:', test_score_2)
        print('test score 3:', test_score_3)
        print('----------')
        return train_score, validate_score

    def visualization(self, train_data, test_data_1, test_data_2, test_data_3):
        # depths = np.arange(1, 11)
        # scores = [self.maxdepth_score(d, train_data, test_data_1, test_data_2, test_data_3) for d in depths]

        learning_rates = np.linspace(0.01, 1.01, 101)
        scores = [self.optimalLearningrate(learning_rate, train_data, test_data_1, test_data_2, test_data_3) for learning_rate in
                  learning_rates]

        # estimators_nums = np.arange(2, 51)
        # scores = [self.optimalEstimators(n, train_data, test_data_1, test_data_2, test_data_3) for n in estimators_nums]

        train_scores = [s[0] for s in scores]
        validate_scores = [s[1] for s in scores]
        plt.figure(111)
        # plt.grid()
        # plt.xlabel('max depth of decision tree')
        # plt.xlabel('optimal estimators of ensemble learning')
        plt.xlabel('optimal learning rate')
        # plt.xlabel('optimal min_impurity_decrease of decision tree')
        plt.ylabel('scores')
        # plt.plot(estimators_nums, train_scores, label='train_score', c='red')
        # plt.plot(estimators_nums, validate_scores, label='validate_score', c='black')
        plt.plot(learning_rates, train_scores, label='train_score', c='red')
        plt.plot(learning_rates, validate_scores, label='validate_score', c='black')

        plt.legend()
        plt.show()
        return

    def xgBoost(self, train_data, test_data_1, test_data_2, test_data_3):
        X, Y = self.loadData(train_data)
        X_test_1, Y_test_1 = self.loadData(test_data_1)
        X_test_2, Y_test_2 = self.loadData(test_data_2)
        X_test_3, Y_test_3 = self.loadData(test_data_3)
        X_train, X_validate, Y_train, Y_validate = \
            train_test_split(X, Y, test_size=1 / 3., random_state=1)
        params = {'booster': 'gbtree',
                  'objective': 'multi:softmax',   # binary:logistic
                  'num_class': 2,
                  'gamma': 0.1,  # 用于控制是否后剪枝
                  'max_depth': 6,  # 构建数的深度，越大越容易过拟合
                  'lambda': 2,   # 控制模型复杂度的权重值L2正则项
                  'subsample': 0.7,  # 随机采样训练样本
                  'colsample_bytree': 1,  # 这个参数默认为1
                  'min_child_weight': 3,
                  'silent': 1, # 设置成1，表示没有运行信息输入
                  'eta': 0.1,  # 学习率
                  'seed': 1000,
                  'nthread': 4,} # CPU线程数
        plst = params.items()
        dtrain = xgb.DMatrix(X_train, Y_train)
        clf = xgb.train(params=plst, dtrain=dtrain, num_boost_round=500)
        X_dtrain = xgb.DMatrix(X_train)
        Y_dtrain_pre = clf.predict(X_dtrain)

        X_dvalidate = xgb.DMatrix(X_validate)
        Y_dvalidate_pre = clf.predict(X_dvalidate)

        X_dtest_1 = xgb.DMatrix(X_test_1)
        Y_dtest_1_pre = clf.predict(X_dtest_1)

        X_dtest_2 = xgb.DMatrix(X_test_2)
        Y_dtest_2_pre = clf.predict(X_dtest_2)

        X_dtest_3 = xgb.DMatrix(X_test_3)
        Y_dtest_3_pre = clf.predict(X_dtest_3)


        train_score = accuracy_score(Y_train, Y_dtrain_pre)
        validate_score = accuracy_score(Y_validate, Y_dvalidate_pre)
        test_score_1 = accuracy_score(Y_test_1, Y_dtest_1_pre)
        test_score_2 = accuracy_score(Y_test_2, Y_dtest_2_pre)
        test_score_3 = accuracy_score(Y_test_3, Y_dtest_3_pre)

        print('train score:', train_score)
        print('validate score:', validate_score)
        print('test score 1:', test_score_1)
        print('test score 2:', test_score_2)
        print('test score 3:', test_score_3)
        return


if __name__ == '__main__':
    el = EnsembleLearning()

    el.adaBoost(train_data=_746Data, test_data_1=_1625Data, test_data_2=_impensData, test_data_3=_schillingData)
    # exit()
    print('-------------')
    el.adaBoost(train_data=_1625Data, test_data_1=_746Data, test_data_2=_impensData, test_data_3=_schillingData)
    print('-------------')
    el.adaBoost(train_data=_impensData, test_data_1=_746Data, test_data_2=_1625Data, test_data_3=_schillingData)
    print('-------------')
    el.adaBoost(train_data=_schillingData, test_data_1=_746Data, test_data_2=_1625Data, test_data_3=_impensData)
    exit()
    el.visualization(train_data=_746Data, test_data_1=_1625Data, test_data_2=_impensData, test_data_3=_schillingData)