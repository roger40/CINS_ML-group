#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/2
# @Author  : Wang Biao
# @Site    :
# @File    : LogisticRegression.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

_746Data = './newHIV-1_data/746Data.txt'
_1625Data = './newHIV-1_data/1625Data.txt'
_impensData = './newHIV-1_data/impensData.txt'
_schillingData = './newHIV-1_data/schillingData.txt'
_x_all = 'ARNDCQEGHILKMFPSTWYV'  # x的全部取值,从1到20进行编码


class LogisticRegression(object):
    def loadData(self, train_path):
        # 导入数据，并对数据进行预处理
        data_train = pd.read_csv(train_path, header=None)
        X_train = data_train.values[:, 0]
        Y_train = data_train.values[:, 1]
        X_train_list = []
        # 需要对属性进行数值化
        for i in range(len(X_train)):
            X_train_list.append([_x_all.index(a)+1 for a in X_train[i]])
        X_train = np.array(X_train_list)
        for i in range(len(Y_train)):
            if Y_train[i] == -1:
                Y_train[i] = 0
        return X_train, list(Y_train)

    def logReg(self, train_path, test_path_1, test_path_2, test_path_3):
        X_train, Y_train = self.loadData(train_path)
        X_test_1, Y_test_1 = self.loadData(test_path_1)
        X_test_2, Y_test_2 = self.loadData(test_path_2)
        X_test_3, Y_test_3 = self.loadData(test_path_3)
        print('X_train:', X_train)
        print('Y_train:', Y_train)
        logreg = linear_model.LogisticRegression(class_weight='balanced')  #
        # logreg = linear_model.LogisticRegression(solver='lbfgs', penalty='l2', C=1.0, class_weight='balanced')
        logreg.fit(X_train, Y_train)
        print('-----------')
        # print(logreg.predict(X_train))
        # print(logreg.predict_proba(X_train))
        print('训练误差：', logreg.score(X_train, Y_train))
        # print(logreg.predict_proba(X_test))
        print('测试误差1：', logreg.score(X_test_1, Y_test_1))
        print('-----------')
        print('测试误差2：', logreg.score(X_test_2, Y_test_2))
        print('-----------')
        print('测试误差3：', logreg.score(X_test_3, Y_test_3))
        print('-----------')
        w, b = logreg.coef_, logreg.intercept_
        print(w, b)
        return w, b

    def train(self, train_path, epoch, error, learning_rate):
        X_train, Y_train = self.loadData(train_path)
        X_train_one = np.hstack((X_train, np.ones((len(X_train), 1))))
        n, d = np.shape(X_train_one)
        print('X_train_one:', X_train_one)
        print('Y_train:', Y_train)
        loss_list = []
        w_list = []
        j = 0
        w = np.random.randn(d, 1)
        for i in range(epoch):
            w_last = w
            j += 1
            loss_last = (self.negloglikelihoodWhole(X_train_one, w_last, Y_train))[0]
            loss_list.append(loss_last)
            w_list.append(w_last)
            print('epoch:', j, 'loss:', loss_last, 'w:', w_last)
            '''牛顿法'''
            # w = w_last-np.dot(np.linalg.inv(self.secondeDerWhole(X_train_one, w_last, Y_train)), self.firstDerWhole(X_train_one, w_last, Y_train))
            '''梯度下降'''
            w = w_last-learning_rate*self.firstDerWhole(X_train_one, w_last, Y_train)
            loss = (self.negloglikelihoodWhole(X_train_one, w, Y_train))[0]
            if abs(loss-loss_last) <= error or np.linalg.norm(w-w_last) <= error:
                print('w', w)
                break
        fig = plt.subplot(111)
        fig.set_title('loss')
        fig.plot([i+1 for i in range(len(loss_list))], loss_list, c='red', marker='.', label='True')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        return w, w_list[loss_list.index(min(loss_list))]

    def test(self, w, train_path, test_path_1, test_path_2, test_path_3):
        X_train, Y_train = self.loadData(train_path)
        X_test_1, Y_test_1 = self.loadData(test_path_1)
        X_test_2, Y_test_2 = self.loadData(test_path_2)
        X_test_3, Y_test_3 = self.loadData(test_path_3)
        X_train_one = np.hstack((X_train, np.ones((len(X_train), 1))))
        X_test_1_one = np.hstack((X_test_1, np.ones((len(X_test_1), 1))))
        X_test_2_one = np.hstack((X_test_2, np.ones((len(X_test_2), 1))))
        X_test_3_one = np.hstack((X_test_3, np.ones((len(X_test_3), 1))))

        print('-----训练准确率-----')
        Y_train_pro = np.dot(X_train_one, w)
        right = 0
        for y_train_pro, y_train in zip(Y_train_pro, Y_train):
            if (y_train_pro >= 0.5 and y_train == 1) or (y_train_pro < 0.5 and y_train == 0):
                right += 1
        print(right/len(Y_train))
        # print('Y_train_pre:', Y_train_pre)
        print('-----测试准确率1-----')
        Y_test_pro = np.dot(X_test_1_one, w)
        right_test = 0
        for y_test_pro, y_test in zip(Y_test_pro, Y_test_1):
            if (y_test_pro >= 0.5 and y_test == 1) or (y_test_pro < 0.5 and y_test == 0):
                right_test += 1
        print(right_test/len(Y_test_1))

        print('-----测试准确率2-----')
        Y_test_pro = np.dot(X_test_2_one, w)
        right_test = 0
        for y_test_pro, y_test in zip(Y_test_pro, Y_test_2):
            if (y_test_pro >= 0.5 and y_test == 1) or (y_test_pro < 0.5 and y_test == 0):
                right_test += 1
        print(right_test / len(Y_test_2))

        print('-----测试准确率3-----')
        Y_test_pro = np.dot(X_test_3_one, w)
        right_test = 0
        for y_test_pro, y_test in zip(Y_test_pro, Y_test_3):
            if (y_test_pro >= 0.5 and y_test == 1) or (y_test_pro < 0.5 and y_test == 0):
                right_test += 1
        print(right_test/len(Y_test_3))

        return

    '''
    def logisticFun(self, x_one, w):
        return 1/(1+math.exp(-np.dot(x_one.T, w)))

    def likelihood(self, x_one, w, y):
        return y*(np.exp(np.dot(w.T, x_one))/(1+np.exp(np.dot(w.T, x_one))))\
               +(1-y)/(1/(1+np.exp(np.dot(w.T, x_one))))
    '''

    '''最小化该目标函数, 负对数似然函数'''
    def negloglikelihood(self, x_one, w, y):
        return -y*np.dot(w.T, x_one)+np.log(1+np.exp(np.dot(w.T, x_one)))

    def negloglikelihoodWhole(self, X_one, w, Y):
        loss = 0
        for x_one, y in zip(X_one, Y):
            loss += self.negloglikelihood(x_one.T, w, y)
        return loss


    def firstDer(self, x_one, w, y):
        return -x_one*(y-np.exp(np.dot(w.T, x_one))/(1+np.exp(np.dot(w.T, x_one))))

    def firstDerWhole(self, X_one, w, Y):
        n, d = np.shape(X_one)
        der_1th = np.zeros((d, 1))
        for x_one, y in zip(X_one, Y):
            der_1th += self.firstDer(x_one.reshape((d, 1)), w, y)
        return der_1th

    def secondeDer(self, x_one, w):
        return np.dot(x_one, x_one.T)*\
               (np.exp(np.dot(w.T, x_one))/(1+np.exp(np.dot(w.T, x_one))))*\
               (1/1+np.exp(np.dot(w.T, x_one)))

    def secondeDerWhole(self, X_one, w, Y):
        n, d = np.shape(X_one)
        der_2nd = np.zeros((d, d))
        for x_one, y in zip(X_one, Y):
            der_2nd += self.secondeDer(x_one.reshape((d, 1)), w)
        return der_2nd

if __name__ == '__main__':
    lr_model = LogisticRegression()
    # lr_model.logReg(train_path=_schillingData, test_path_1=_746Data,
    #                 test_path_2=_1625Data, test_path_3=_impensData)

    # lr_model.logReg(train_path=_impensData, test_path_1=_746Data,
    #                 test_path_2=_1625Data, test_path_3=_schillingData)

    # lr_model.logReg(train_path=_746Data, test_path_1=_1625Data,
    #                 test_path_2=_impensData, test_path_3=_schillingData)

    # exit()
    w, w_min = lr_model.train(train_path=_impensData, epoch=200, error=0.01, learning_rate=0.00001)

    print('------w-----')
    print('w:', w)
    # lr_model.test(w=w, train_path=_746Data, test_path_1=_1625Data,
    #                 test_path_2=_impensData, test_path_3=_schillingData)

    lr_model.test(w=w, train_path=_impensData, test_path_1=_746Data,
                  test_path_2=_1625Data, test_path_3=_schillingData)

    # lr_model.test(w=w, train_path=_schillingData, test_path_1=_746Data,
    #               test_path_2=_1625Data, test_path_3=_impensData)

    # lr_model.test(w=w, train_path=_1625Data, test_path_1=_746Data,
    #               test_path_2=_impensData, test_path_3=_schillingData)

    print('----w_min-----')
    print('w_min:', w_min)
    lr_model.test(w=w_min,  train_path=_impensData, test_path_1=_746Data,
                  test_path_2=_1625Data, test_path_3=_schillingData)