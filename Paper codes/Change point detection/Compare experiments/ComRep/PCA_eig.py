# -*- coding=utf-8 -*-

import numpy as np
"""PCA"""
def getEigVect(covmat, num=2):
    # try:
    eigvalue, eigvector = np.linalg.eig(covmat)
    sumeigvalue = sum(eigvalue)
    # print(eigvalue)
    eigvalue_list = list(eigvalue)
    # c = np.argsort(-np.array(a))
    # print(c[0])
    eigvaluesort = sorted(eigvalue_list, reverse=True)
    # print((eigvector.shape))
    # print(eigvaluesort)
    # print(np.array(eigvector).shape)
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
    # 取一个主特征
    # print(eigvectors[0])
    # EigVector = np.matrix(EigVectors[0])
    # 去两个主特征
    # print(eigvector[eigvalue_list.index(obj)][0].shape)
    vector = np.matrix(eigvectors[0:num])
    # EigVector = np.matrix(EigVectors)
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

def pca2(evs, eigvect):
    size = int(np.sqrt(evs.size))
    evs.shape = size, size  # 传进来的是1*n^2的矩阵，需要转维度
    #
    # 均值
    # #
    a_mat = evs
    evs = np.mat(evs - np.mean(evs, axis=0))

    # print(type(eigvect), type(evs))
    f = evs * eigvect.T    # 保持计算与pca1_6的一致，排除其他影响
    m2_mat = f * eigvect
    # #
    # print("next matrix: " + str(a_mat))
    # print("after next matrix: " + str(m2_mat))

    return m2_mat, f

# ADD 5
def getNorm1_6(evs):    #将事件格式转化data[x]=w
    meanrow = np.mean(evs, axis=0)  # 按列求每个字段的平均
    size = int(np.sqrt(meanrow.size))
    meanrow.shape = size, size
    # print(meanrow)
    row = np.mean(meanrow, axis=0)  # 每列的平均值
    # 正常均值
    # row = getMean(meanrow)     # 稀疏矩阵的PCA均值计算方法
    # print("eves:" + str(evs))
    # print("meanRow:" + str(meanrow))
    # print("row:" + str(row))
    # print("Norm:" + str(meanrow-row))
    return np.mat(meanrow-row), np.mat(meanrow), np.mat(row)

def ti(m):
    # import copy
    x, y = m.shape
    s = np.zeros_like(m)
    for i in range(x):
        for j in range(y):
            s[i, j] = round(m[i, j], 2)
    return s

def pca1_6(evs, num=2):
    norm, meanrow, row = getNorm1_6(evs)  # Norm 为归零矩阵，meanRow为原数据，mm为数据均值
    print(ti(norm.T))
    # covMat = np.cov(Norm, rowvar=0)
    # print(type(norm), type(meanrow), type(row))
    cov_mat = (1/norm.shape[0])*norm.T*norm
    print((1/norm.shape[0]))
    print(ti(cov_mat))
    # print(type(cov_mat))
    # mean_Norm = Norm - np.mean(Norm, axis=0)
    # covMat = (1 / mean_Norm.shape[0]) * mean_Norm.T * mean_Norm
    eigvect = getEigVect(cov_mat, num=num)  # 1*n
    print(ti(eigvect))

    #
    # 均值
    # #
    a_mat = meanrow
    meanrow = norm

    # print(type(meanrow), type(eigvect))
    f = meanrow * eigvect.T  # n*1
    # print("feature", f)
    m_mat = f * eigvect
    # 相似度极低的原因
    # f1 = Norm * eigVect.T  # n*1
    # M1 = f1 * eigVect + mm
    # drawGraph(ti(A), "origin")
    # drawGraph(ti(M), "after")
    # print("before: " + str(a_mat))
    # print("after itself:" + str(m_mat))
    print(ti(m_mat.T))
    return m_mat, eigvect, a_mat #, meanrow
    # return M1, eigVect, meanrow
# ADD 5
