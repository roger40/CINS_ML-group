# -*- coding=utf-8 -*-

import numpy as np
"""PCA"""
def getEigVect(covmat, num=2):
    eigvalue, eigvector = np.linalg.eig(covmat)
    sumeigvalue = sum(eigvalue)
    eigvalue_list = list(eigvalue)
    eigvaluesort = sorted(eigvalue_list, reverse=True)
    eigvalues = []
    eigvectors = []
    t = 0
    for obj in eigvaluesort:
        eigvalues.append(obj)
        eigvectors.append(np.array(eigvector)[:, eigvalue_list.index(obj)])
        t += 1
        if sum(eigvalues) / sumeigvalue > 0.9:
            break
        elif t > num:
            break
    vector = np.matrix(eigvectors[0:num])
    return vector

def getMean(evs):
    r, c = evs.shape
    # print(row, clum)
    row = np.zeros(c)
    # print(evs)
    for j in range(c):
        k = 0
        for i in range(r):
            row[j] += evs[i, j]
            if evs[i, j] != 0:
                k += 1
        if k != 0:
            row[j] = row[j]/k
    # print(Row)
    return row

def pca2(evs, eigvect, n=None):
    if n == None:
        size = int(np.sqrt(evs.size))
        evs.shape = size, size  # 传进来的是1*n^2的矩阵，需要转维度
    else:
        evs.shape = n, -1
    #
    # 均值
    # #
    a_mat = evs
    evs = np.mat(evs - np.mean(evs, axis=0))

    f = evs * eigvect.T    # 保持计算与pca1_6的一致，排除其他影响
    m2_mat = f * eigvect

    return m2_mat, f

# ADD 5
def getNorm1_6(evs, n=None):    #将事件格式转化data[x]=w
    meanrow = np.mean(evs, axis=0)  # 按列求每个字段的平均
    if n == None:
        size = int(np.sqrt(meanrow.size))
        meanrow.shape = size, size
    else:
        meanrow.shape = n, -1
    # print(meanrow)
    row = np.mean(meanrow, axis=0)  # 每列的平均值

    return np.mat(meanrow-row), np.mat(meanrow), np.mat(row)

def pca1_6(evs, num=2, n=None):
    norm, meanrow, row = getNorm1_6(evs, n=n)
    cov_mat = (1/norm.shape[0])*norm.T*norm
    eigvect = getEigVect(cov_mat, num=num)  # 1*n
    # print(eigvect)

    #
    # 均值
    # #
    a_mat = meanrow
    meanrow = norm

    f = meanrow * eigvect.T  # n*1
    m_mat = f * eigvect
    return m_mat, eigvect, a_mat #, meanrow