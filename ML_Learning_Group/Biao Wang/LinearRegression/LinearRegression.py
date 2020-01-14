#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/24
# @Author  : 王标
# @Site    :
# @File    : LinearRegression.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def loadData(self):
        data = pd.read_csv(self.data_path)
        data_used = (data.iloc[:, 1:27]).values  # 取了第1到26列, 第1列为预测值
        print(np.shape(data_used))
        Y = data_used[:, 0]
        Y = Y.reshape(len(Y), 1)
        X = data_used[:, 1:26]
        print(np.shape(X))
        print('-------------')
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y, test_size=1 / 3., random_state=None)
        return X_train, X_test, Y_train, Y_test

    def processData(self):
        data = pd.read_csv(self.data_path)
        data_used = (data.iloc[:, 1:27]).values  # 取了第1到26列, 第1列为预测值
        # data_used = preprocessing.MinMaxScaler().fit_transform(data_used)
        data_used = preprocessing.StandardScaler().fit_transform(data_used)
        # data_used = PolynomialFeatures(degree=3)

        Y = data_used[:, 0]
        Y = Y.reshape(len(Y), 1)
        X = data_used[:, 1:26]
        # X = PolynomialFeatures(degree=2).fit_transform(X)
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y, test_size=1 / 3., random_state=None)
        return X_train, X_test, Y_train, Y_test


    def MultiGaussian(self, x, u, C):
        d = len(x)
        coefficient = (1/((2*math.pi)**(d/2)))*(1/((np.linalg.det(C))**0.5))
        exponent = math.exp((-0.5*np.dot((x-u).T, np.dot(np.linalg.inv(C), (x-u))))[0][0])
        return coefficient*exponent

    def diagonalMat(self, d):
        X = np.random.randn(d, d)
        X = np.triu(X)
        X += X.T - np.diag(X.diagonal())
        return X

    def designMat_MG(self, m):
        data = pd.read_csv(self.data_path)
        data_used = (data.iloc[:, 1:27]).values  # 取了第1到26列, 第1列为预测值
        Y = data_used[:, 0]
        Y = Y.reshape(len(Y), 1)
        X = data_used[:, 1:27]
        n, d = np.shape(X)
        design_mat = np.empty(shape=[len(X), m])
        u_list = [np.ones((d, 1))*i for i in range(m)]
        C = np.diag([1 for i in range(d)])
        print('C:', C)
        # C_list = [self.diagonalMat(d) for i in range(m)]
        for i in range(m):
            if i == m-1:
                design_mat[:, i] = [1 for x in design_mat]
            else:
                design_mat[:, i] = [self.MultiGaussian(x, u_list[i], C) for x in design_mat]
        X_train, X_test, Y_train, Y_test = \
            train_test_split(design_mat, Y, test_size=1 / 3., random_state=None)
        return X_train, X_test, Y_train, Y_test


    def powerFunction(self, x, i):
        return x**i

    def designMatrix(self, X_train, X_test):  # 拼接一列全1向量
        ones = np.ones((len(X_train), 1))
        ones_ = np.ones((len(X_test), 1))
        # X_train_d = np.hstack((X_train, ones))
        # X_test_d = np.hstack((X_test, ones_))
        X_train_d = np.concatenate((X_train, ones), axis=1)
        X_test_d = np.concatenate((X_test, ones_), axis=1)
        return X_train_d, X_test_d

    def calculateW(self, X_train_d, Y_train):
        left = np.linalg.inv(np.dot(X_train_d.T, X_train_d))
        right = np.dot(X_train_d.T, Y_train)
        return np.dot(left, right)

    def calculateW_regular_terms(self, X_train_d, Y_train, alpha):
        n, d = np.shape(X_train_d)
        left = np.linalg.inv(np.dot(X_train_d.T, X_train_d)+alpha*np.identity(d))
        right = np.dot(X_train_d.T, Y_train)
        return np.dot(left, right)

    def visualization(self, w, X_train_d, X_test_d, Y_train, Y_test):
        Y_train_pre = np.dot(X_train_d, w)
        fig = plt.subplot(211)
        fig.set_title('lr_train')
        fig.plot(range(len(X_train)), Y_train, c='black', marker='.', label='True')
        fig.plot(range(len(X_train)), Y_train_pre, c='red', marker='.', label='Pre')
        plt.legend()
        print('----test----')
        Y_test_pre = np.dot(X_test_d, w)
        test_loss = (np.linalg.norm(Y_test_pre - Y_test)) ** 2
        print('test_loss:', test_loss)
        fig = plt.subplot(212)
        fig.set_title('lr_test')
        fig.plot(range(len(X_test)), Y_test, c='black', marker='.', label='True')
        fig.plot(range(len(X_test)), Y_test_pre, c='red', marker='.', label='Pre')
        plt.legend()
        plt.show()
        return

    # 均方误差/平方损失，其实可以看做等同于平方和误差函数，就是损失函数前面加了个系数
    def loss(self, w, X_train_d, Y_train): # w为列向量
        Y_predict = np.dot(X_train_d, w)
        # return (np.linalg.norm(Y_predict-Y_train))**2
        return (np.linalg.norm(Y_predict-Y_train))**2

    # 均方根误差函数
    def loss_rmse(self, w, X_train_d, Y_train):
        Y_predict = np.dot(X_train_d, w)
        return (np.linalg.norm(Y_predict-Y_train)/(len(Y_train)**0.5))

    def loss_regular_terms(self, w, X_train_d, Y_train, alpha): # 加正则项
        Y_predict = np.dot(X_train_d, w)
        return (np.linalg.norm(Y_predict-Y_train))**2+alpha*(np.linalg.norm(w))**2
        # 对应的梯度就是增加了2*alpha*w

    def loss_rmse_regular_terms(self, w, X_train_d, Y_train, alpha):
        Y_predict = np.dot(X_train_d, w)
        return (np.linalg.norm(Y_predict-Y_train)/(len(Y_train)**0.5))+alpha*(np.linalg.norm(w))**2

    def train(self, X_train_d, Y_train, alpha, epoch, learning_rate, error):
        n, d= np.shape(X_train_d)
        i = 0
        w = np.random.randn(d, 1)
        print('初始w:', w)
        for i in range(epoch):
            w_last = w
            i += 1
            loss_last = self.loss_regular_terms(w_last, X_train_d, Y_train, alpha)
            print('epoch:', i, 'w: ', w, 'loss: ', loss_last)
            # print('epoch:', i, 'loss: ', loss_last)
            # gradient = 2*np.dot(X_train_d.T, (np.dot(X_train_d, w_last)-Y_train))
            gradient = 2*np.dot(X_train_d.T, (np.dot(X_train_d, w_last)-Y_train))+2*alpha*w
            w = w_last-learning_rate*gradient
            loss = self.loss_regular_terms(w, X_train_d, Y_train, alpha)
            if abs(loss-loss_last) <= error or np.linalg.norm(w-w_last) <= error:
                print('w', w)
                break
        return w

    # 随机梯度下降
    def train_sgd(self, X_train_d, Y_train, learning_rate, error_1, error_2, epoch):
        n, d = np.shape(X_train_d)
        w = np.random.randn(d, 1)
        print('初始w:', w)
        for j in range(epoch):  # 遍历完所有样本为一个epoch
            # 进行相同的打乱操作
            state = np.random.get_state()
            np.random.shuffle(X_train_d)
            np.random.set_state(state)
            np.random.shuffle(Y_train)
            for i in range(len(X_train_d)):
                w_last = w
                loss_last = self.loss(w_last, X_train_d, Y_train)
                print('epoch:', j, '第%d个样本' %(i), 'loss: ', loss_last)
                # gradient = 2 * np.dot(X_train_d.T, (np.dot(X_train_d, w_last) - Y_train))
                gradient = 2 * ((np.dot(X_train_d[i], w)-Y_train[i])[0]) * (X_train_d[i]).T
                w = w_last - learning_rate * gradient
                loss = self.loss(w, X_train_d, Y_train)
                if abs(loss - loss_last) <= error_1 or np.linalg.norm(w - w_last) <= error_2:
                    print('以求得最优w：', w)
                    break
        return w

    # 批梯度下降
    def train_bgd(self, X_train_d, Y_train, learning_rate, error_1, error_2, epoch, batch_size):
        n, d = np.shape(X_train_d)
        w = np.random.randn(d, 1)   # 基于标准高斯分布
        batch_num = math.ceil(len(X_train_d)/batch_size)
        print('初始w:', w)
        print('batch_size:', batch_size, 'batch_num:', batch_num)
        for j in range(epoch):  # 遍历完所有样本为一个epoch
            # 进行相同的打乱操作
            state = np.random.get_state()
            np.random.shuffle(X_train_d)
            np.random.set_state(state)
            np.random.shuffle(Y_train)
            k = 0
            for i in range(batch_num):
                if i < batch_num-1:
                    X_batch = X_train_d[k:(k+batch_size)]
                    Y_batch = Y_train[k:(k+batch_size)]
                else:
                    X_batch = X_train_d[k:len(X_train_d)]
                    Y_batch = Y_train[k:len(X_train_d)]
                w_last = w
                loss_last = self.loss(w_last, X_train_d, Y_train)
                print('epoch:', j, '第%d个batch' % (i), 'loss: ', loss_last)
                # gradient = 2 * np.dot(X_train_d.T, (np.dot(X_train_d, w_last) - Y_train))
                # gradient = 2 * (np.dot(X_train_d[i], w)[0][0]) * (X_train_d[i]).T
                gradient = 2 * np.dot(X_batch.T, (np.dot(X_batch, w_last) - Y_batch))
                w = w_last - learning_rate * gradient
                loss = self.loss(w, X_train_d, Y_train)
                if abs(loss - loss_last) <= error_1 or np.linalg.norm(w - w_last) <= error_2:
                    print('以求得最优w：', w)
                    break
                k = k+batch_size
        return w


    def test(self, X_test, Y_test, w):
        Y_pre = np.dot(X_test, w)
        test_loss = self.loss(X_test, Y_test, w)
        return Y_pre, test_loss

    def LR(self, X_train, Y_train, X_test, Y_test):  # 进行调包
        lr = linear_model.LinearRegression()
        lr.fit(X_train, Y_train)
        w,b = lr.coef_, lr.intercept_
        print('w', w)
        print('b', b)
        Y_train_pre = lr.predict(X_train)
        train_loss = (np.linalg.norm(Y_train_pre-Y_train))**2
        print('train_loss:', train_loss)
        fig = plt.subplot(211)
        fig.set_title('lr_train')
        fig.plot(range(len(X_train)), Y_train, c='black', marker='.', label='True')
        fig.plot(range(len(X_train)), Y_train_pre, c='red', marker='.', label='Pre')
        plt.legend()
        print('----test----')
        Y_test_pre = lr.predict(X_test)
        test_loss = (np.linalg.norm(Y_test_pre-Y_test))**2
        print('test_loss:', test_loss)
        fig = plt.subplot(212)
        fig.set_title('lr_test')
        fig.plot(range(len(X_test)), Y_test, c='black', marker='.', label='True')
        fig.plot(range(len(X_test)), Y_test_pre, c='red', marker='.', label='Pre')
        plt.legend()
        plt.show()
        return

    def LR_ridge(self, X_train, Y_train, X_test, Y_test):
        lr = linear_model.Ridge(alpha=10.0)  # 加了一个l2正则项
        lr.fit(X_train, Y_train)
        w, b = lr.coef_, lr.intercept_
        print('w', w)
        print('b', b)
        # Y_train_pre = lr.predict(X_train)
        Y_train_pre = np.dot(X_train, w.T)+b*np.ones((len(X_train), 1))
        train_loss = (np.linalg.norm(Y_train_pre - Y_train)) ** 2
        print('train_loss:', train_loss)
        fig = plt.subplot(211)
        fig.set_title('ridge_train')
        fig.plot(range(len(X_train)), Y_train, c='black', marker='.', label='True')
        fig.plot(range(len(X_train)), Y_train_pre, c='red', marker='.', label='Pre')
        plt.legend()
        print('----test----')
        Y_test_pre = lr.predict(X_test)
        test_loss = (np.linalg.norm(Y_test_pre - Y_test)) ** 2
        print('test_loss:', test_loss)
        fig = plt.subplot(212)
        fig.set_title('ridge_test')
        fig.plot(range(len(X_test)), Y_test, c='black', marker='.', label='True')
        fig.plot(range(len(X_test)), Y_test_pre, c='red', marker='.', label='Pre')
        plt.legend()
        plt.show()
        return

    def LR_lasso(self, X_train, Y_train, X_test, Y_test):
        lr = linear_model.Lasso(alpha=10.0)  # 加了一个l1正则项
        lr.fit(X_train, Y_train)
        w, b = lr.coef_, lr.intercept_
        print('w', w)
        print('b', b)
        Y_train_pre = lr.predict(X_train)
        train_loss = (np.linalg.norm(Y_train_pre - Y_train)) ** 2
        print('train_loss:', train_loss)
        fig = plt.subplot(211)
        fig.set_title('lasso_train')
        fig.plot(range(len(X_train)), Y_train, c='black', marker='.', label='True')
        fig.plot(range(len(X_train)), Y_train_pre, c='red', marker='.', label='Pre')
        plt.legend()
        print('----test----')
        Y_test_pre = lr.predict(X_test)
        test_loss = (np.linalg.norm(Y_test_pre - Y_test)) ** 2
        print('test_loss:', test_loss)
        fig = plt.subplot(212)
        fig.set_title('lasso_test')
        fig.plot(range(len(X_test)), Y_test, c='black', marker='.', label='True')
        fig.plot(range(len(X_test)), Y_test_pre, c='red', marker='.', label='Pre')
        plt.legend()
        plt.show()
        return

    def LR_elasticNet(self, X_train, Y_train, X_test, Y_test):
        lr = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)  # 加了l1,l2正则项
        lr.fit(X_train, Y_train)
        w, b = lr.coef_, lr.intercept_
        print('w', w)
        print('b', b)
        Y_train_pre = lr.predict(X_train)
        train_loss = (np.linalg.norm(Y_train_pre - Y_train)) ** 2
        print('train_loss:', train_loss)
        fig = plt.subplot(211)
        fig.set_title('elasticNet')
        fig.plot(range(len(X_train)), Y_train, c='black', marker='.', label='True')
        fig.plot(range(len(X_train)), Y_train_pre, c='red', marker='.', label='Pre')
        plt.legend()
        print('----test----')
        Y_test_pre = lr.predict(X_test)
        test_loss = (np.linalg.norm(Y_test_pre - Y_test)) ** 2
        print('test_loss:', test_loss)
        fig = plt.subplot(212)
        fig.set_title('elasticNet')
        fig.plot(range(len(X_test)), Y_test, c='black', marker='.', label='True')
        fig.plot(range(len(X_test)), Y_test_pre, c='red', marker='.', label='Pre')
        plt.legend()
        plt.show()
        return

if __name__ == '__main__':
    lr_model = LinearRegression(data_path='data.csv')
    '''
    X_train, X_test, Y_train, Y_test = lr_model.designMat_MG(5)
    print(len(Y_train))
    print(len(Y_test))
    exit()
    '''
    # X_train, X_test, Y_train, Y_test = lr_model.processData() # 进行归一化处理

    X_train, X_test, Y_train, Y_test = lr_model.loadData()
    X_train_d, X_test_d = lr_model.designMatrix(X_train, X_test)
    w = lr_model.train(X_train_d, Y_train, epoch=100, alpha=1, learning_rate=0.001, error=1)
    # w = lr_model.train_sgd(X_train_d=X_train_d, Y_train=Y_train, learning_rate=0.01, error_1=1, error_2=1, epoch=5)
    # lr_model.train_bgd(X_train_d=X_train_d, Y_train=Y_train, learning_rate=0.01, error_1=10, error_2=1, epoch=5, batch_size=100)

    # w = lr_model.calculateW(X_train_d, Y_train)
    # w = lr_model.calculateW_regular_terms(X_train_d, Y_train, alpha=1.0)
    # print(w)
    lr_model.visualization(w, X_train_d, X_test_d, Y_train, Y_test)
    exit()
    '''
    print('----------------')
    print('lr')
    lr_model.LR(X_train, Y_train, X_test, Y_test)
    exit()
    '''
    '''
    print('----------------')
    print('ridge')
    lr_model.LR_ridge(X_train, Y_train, X_test, Y_test)
    '''

    print('----------------')
    print('lasso')
    lr_model.LR_lasso(X_train, Y_train, X_test, Y_test)
    exit()
    print('----------------')
    print('elasticNet')
    lr_model.LR_elasticNet(X_train, Y_train, X_test, Y_test)
    exit()

    lr_model.train(X_train_d, Y_train, learning_rate=0.01, error=1)