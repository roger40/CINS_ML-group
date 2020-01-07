#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/6
# @Author  : Wang Biao
# @Site    :
# @File    : DecisionTree.py
# @Software: PyCharm

import graphviz
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
import os
import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction import DictVectorizer

os.environ['PATH'] += os.pathsep + 'G:\graphviz-2.38\\release\\bin'

'''
criterion:分裂节点时的评价指标。gini, entropy
max_depth:树的最大深度
splitter:指定分裂节点时的策略

'''

_746Data = '../LogisticRegression/newHIV-1_data/746Data.txt'
_1625Data = '../LogisticRegression/newHIV-1_data/1625Data.txt'
_impensData = '../LogisticRegression/newHIV-1_data/impensData.txt'
_schillingData = '../LogisticRegression/newHIV-1_data/schillingData.txt'
_x_all = 'ARNDCQEGHILKMFPSTWYV'

class DecisionTree(object):
    def loadData(self):
        data = pd.read_csv('./ppdai_3_23/LCIS.csv')
        '''
        选择第2、3、6、7、8、9、10、11、12、15、17、18、20、21作为数据属性特征
        借款金额，借款期限，初始评级，借款类型，是否首标（借），年龄，性别，手机认证，
        户口认证，征信认证，历史成功借款次数，历史成功借款金额，历史正常还款期数，历史逾期还款期数
        选择第29列作为标签判断用户是否逾期，标当前逾期天数
        '''
        target_data = data.iloc[:, [1, 2, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 19, 20, 28]]
        print(target_data)
        X = (target_data.iloc[:, 0:14]).values
        print('------------')
        print(X)
        print('------------')
        Y = (target_data.iloc[:, [-1]]).values
        Y_new = []
        for i in range(len(Y)):
            if Y[i] != 0:
                Y_new.append('逾期')
            else:
                Y_new.append('未逾期')
        Y_new = np.array(Y_new).reshape((len(Y_new), 1))
        print('---------------')
        print(np.shape(X))
        print(np.shape(Y_new))
        print('---------------')
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y_new, test_size=1/3., random_state=None)
        return X_train, X_test, Y_train, Y_test

    def loadData_2(self, data_path):
        # 用那个基因数据集
        data = pd.read_csv(data_path, header=None)
        X_train = data.values[:, 0]
        Y_train = data.values[:, 1]
        X_train_list = []
        # 需要对属性进行数值化
        for i in range(len(X_train)):
            X_train_list.append([a for a in X_train[i]])
            # X_train_list.append([_x_all.index(a) + 1 for a in X_train[i]])  # 从1到20进行编号, 字符需要转化为数值,
        X_train = np.array(X_train_list)
        for i in range(len(Y_train)):
            if Y_train[i] == -1:
                Y_train[i] = 0
        # print(np.hstack((X_train, Y_train.reshape(len(Y_train), 1))))
        return X_train, list(Y_train)

    def decisionTree(self, train_data):
        # X_train, X_test, Y_train, Y_test = self.loadData()
        X, Y = self.loadData_2(train_data)
        X_train, X_validate, Y_train, Y_validate = \
            train_test_split(X, Y, test_size=1 / 3., random_state=1)
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=8, class_weight='balanced',
                                          random_state=1, min_impurity_decrease=0.002)
        clf = clf.fit(X_train, Y_train)
        # accuracy = accuracy_score(Y_test, clf.predict(X_test))
        # print('accuracy:', accuracy)
        print('train score:', clf.score(X_train, Y_train))

        print('test score:', clf.score(X_validate, Y_validate))
        # print('----------')
        # print(clf.predict(X_test))
        # print('----------')
        # exit()
        # 绘制决策树图像
        feature_name = ['index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7', 'index_8']
        target_name = ['non-cleaved', 'cleaved']  # 对应于[0, 1]
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_name,
                             class_names=target_name)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf('tree.pdf')
        print('决策树图片已保存')
        return

    def maxdepth_score(self, d, train_data, test_data_1, test_data_2, test_data_3):
        X_test_1, Y_test_1 = self.loadData_2(test_data_1)
        X_test_2, Y_test_2 = self.loadData_2(test_data_2)
        X_test_3, Y_test_3 = self.loadData_2(test_data_3)
        X, Y = self.loadData_2(train_data)
        X_train, X_validate, Y_train, Y_validate = \
            train_test_split(X, Y, test_size=1 / 3., random_state=1)
        clf = tree.DecisionTreeClassifier(max_depth=d, criterion='entropy', class_weight='balanced', random_state=1)
        clf = clf.fit(X_train, Y_train)
        train_score, validate_score = clf.score(X_train, Y_train), clf.score(X_validate, Y_validate)
        print('max_depth:', d)
        print('train score:', train_score)
        print('validate score:', validate_score)
        print('test score 1:', clf.score(X_test_1, Y_test_1))
        print('test score 2:', clf.score(X_test_2, Y_test_2))
        print('test score 3:', clf.score(X_test_3, Y_test_3))
        print('----------------------------')
        return train_score, validate_score
    
    def minsplit_score(self, threshold, train_data, test_data_1, test_data_2, test_data_3):
        X_test_1, Y_test_1 = self.loadData_2(test_data_1)
        X_test_2, Y_test_2 = self.loadData_2(test_data_2)
        X_test_3, Y_test_3 = self.loadData_2(test_data_3)
        X, Y = self.loadData_2(train_data)
        X_train, X_validate, Y_train, Y_validate = \
            train_test_split(X, Y, test_size=1 / 3., random_state=1)
        clf = tree.DecisionTreeClassifier(max_depth=8, criterion='gini',
                                          class_weight='balanced', random_state=1,
                                          min_impurity_decrease=threshold)
        clf = clf.fit(X_train, Y_train)
        train_score, validate_score = clf.score(X_train, Y_train), clf.score(X_validate, Y_validate)
        print('min_impurity_decrease:', threshold)
        print('train score:', train_score)
        print('validate score:', validate_score)
        print('test score 1:', clf.score(X_test_1, Y_test_1))
        print('test score 2:', clf.score(X_test_2, Y_test_2))
        print('test score 3:', clf.score(X_test_3, Y_test_3))
        print('----------------------------')
        return train_score, validate_score


    def visualization(self, train_data, test_data_1, test_data_2, test_data_3):
        # depths = np.arange(1, 11)
        # scores = [self.maxdepth_score(d, train_data, test_data_1, test_data_2, test_data_3) for d in depths]
        thresholds = np.linspace(0, 0.2, 101)
        scores = [self.minsplit_score(threshold, train_data, test_data_1, test_data_2, test_data_3) for threshold in thresholds]
        train_scores = [s[0] for s in scores]
        validate_scores = [s[1] for s in scores]
        plt.figure(111)
        # plt.grid()
        # plt.xlabel('max depth of decision tree')
        plt.xlabel('min impurity decrease of decision tree')
        plt.ylabel('scores')
        plt.plot(thresholds, train_scores, label='train_score', c='red')
        plt.plot(thresholds, validate_scores, label='validate_score', c='black')
        plt.legend()
        plt.show()
        return

    '''
    不调包进行复现
    '''



if __name__ == '__main__':
    decisiontree = DecisionTree()
    '''
    X = [[1, 1, 1, 'yes'],
         [1, 0, 0, 'no'],
         [0, 1, 1, 'no'],
         [0, 0, 1, 'no'],
         [1, 1, 0, 'yes'],
         [0, 1, 1, 'no'],
         [0, 0, 0, 'yes']]
    cha_label = ['态度', '技能', '学费']
    '''
    cha_label = ['index1', 'index2', 'index3', 'index4', 'index5', 'index6', 'index7', 'index8']
    X, Y = decisiontree.loadData_2(data_path=_746Data)
    _X = np.hstack((X, np.array(Y).reshape((len(Y), 1))))
    print('_X:', _X)
    # 将数组转化为datafram
    df = pd.DataFrame(data=_X, columns=['index1', 'index2', 'index3', 'index4', 'index5', 'index6', 'index7', 'index8', 'label'])
    print('df:', df)
    '''
    vec = DictVectorizer(sparse=False)
    feature = df[cha_label]
    X_train = vec.fit_transform(feature.to_dict(orient='record'))
    print('show feature\n', feature)
    print('show vector\n', X_train)
    print('show vector name\n', vec.get_feature_names())
    exit()
    '''

    tree = decisiontree.fit(X, cha_label)
    print(tree)
    exit()

    # decisiontree.visualization(train_data=_1625Data, test_data_1=_746Data,
    #                            test_data_2=_impensData, test_data_3=_schillingData)
    decisiontree.decisionTree(train_data=_746Data)

    decisiontree.visualization(train_data=_746Data, test_data_1=_1625Data,
                               test_data_2=_impensData, test_data_3=_schillingData)
    exit()
