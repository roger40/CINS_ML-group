#!/usr/local/bin/python
# -*-coding:utf-8 -*-

import numpy as np
import scipy.io as scio
import networkx as nx
from scipy.stats import ks_2samp
from optparse import OptionParser
import sys
#from evaluationPandR import Evaluation

# from evaluation import Evaluation
import random
random.seed(5)

class ChangePointDetection():
    def __init__(self, window,alpha,boots):
        '''

        :param window: the size of window
        :param alpha: the significant level
        '''
        self.window = window
        self.alpha = alpha
        self.boots = boots
        self.cpL = []

    def detectAnomaliesInSequence(self,networkSequence,sequenceLen,nodeSize):
        '''

        :param networkSequence:
        :return:
        '''
        self.networkSequence = networkSequence
        self.sequenceLen = sequenceLen
        self.nodeSize = nodeSize


        t = 0
        while t <= self.sequenceLen - self.window:
            startindx = t
            endindx = t + self.window

            self.currentNorNetworks =[self.networkSequence[gi,:] for gi in range(startindx,endindx)]
            changeDetected = self.testNetwork(self.currentNorNetworks)


            if changeDetected:
                tau = t + self.window
                t += np.argmin(self.test_model)
                self.cpL.append(t+1)
            t += 1


    def testNetwork(self, currentNorNetworks):
        '''
        计算每个切割点对应pvalue
        :param currentNorNetworks:
        :return:
        '''
        # print currentNorNetworks
        ng = len(currentNorNetworks)
        pvalueS = np.empty(ng - 1)#ng-1个可能的变点

        for c in range(1, ng):#对每个可能的变点
            adjmat1 = currentNorNetworks[:c]
            adjmat2 = currentNorNetworks[c:]

            # print adjmat1
            # print adjmat2
            deg1 = get_deg(adjmat1, self.nodeSize)
            deg2 = get_deg(adjmat2, self.nodeSize)
            observationD = ks(deg1, deg2)

            # test bootstrap 1000 times
            sum = 0.0
            for j in np.arange(self.boots):
                bde = bootstrap(deg1, self.nodeSize*c)#
                bs = ks(deg1, bde)
                if bs > observationD:
                    sum += 1
            p = sum / self.boots

            pvalueS[c-1]=p
            # pvalueS[c - 1] = observationD

        self.test_model = pvalueS
        return min(pvalueS)<self.alpha



def get_deg(adjS,nodeSize):
    # 存储采样————观测值为节点的度
    deg = []
    for adj in adjS:#对每个snapshot
        adjmat = np.reshape(adj, (nodeSize, nodeSize))
        G = nx.from_numpy_matrix(adjmat)
        deg_dict = list(G.degree().values())  # 返回节点、度二元组 迭代器
        for pairs in deg_dict:
            # deg.append(pairs[1])
            if pairs != 0:
                deg.append(pairs)
    return deg

def ks(deg1, deg2):
    result = ks_2samp(deg1, deg2)
    d = result.statistic
    return d
    # p = result.pvalue
    # return p

def ksDistance(adjmat1,adjmat2,nodeSize):
    fdegree = get_deg(adjmat1, nodeSize)
    ndegree = get_deg(adjmat2, nodeSize)
    # print(ndegree)
    d = ks(fdegree, ndegree)
    return d


def bootstrap(degree, node_size):
    # dg = []
    # for d in degree:
    #     dg.append(d)
    # index = np.random.randint(len(degree), size=len(degree))
    # deg = []
    # for i in index:
    #     deg.append(degree[i])
    # return deg
    bs_sample = np.random.choice(degree,len(degree))##1000
    return bs_sample

def Evaluation(anomaly, schedule, n):
    temp = anomaly
    tp = 0
    fp = 0
    for item in temp:
        if item in schedule:
            tp += 1
        else:
            fp += 1  # 检测错误
    fn = len(schedule) - tp  # 未检测出的异常点数
    tn = (n - fn - tp) - fp  # 本来是正常的，也确实没检测为异常
    accuracy = (tp + tn) / n
    recall = tp / len(schedule)
    if tp == 0:
        precision = 0
        f1 = 0
    else:
        precision = tp / len(temp)
        f1 = 2 * recall * precision / (recall + precision)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    # self.Z.append([FPR, TPR])
    # print("recall: %d/%d = %f    precision: %d/%d = %f  假阳性:%f   F1:%f"
    #       % (tp, len(schedule), recall,
    #          tp, (tp + fp), precision, fpr, f1))
    return recall, precision, f1, fpr

def detect(networkData, window,type1):
    alpha = 0.05
    boots = 1000
    networkData = scio.mmread(networkData) #.todense()
    sequenceLen, node2 = networkData.shape
    nodeSize = int(np.sqrt(node2))

    if type1 ==1:#1 for MIT
        MIT_Ground_month = set([1, 7, 8, 10, 11, 12, 14, 15, 16, 18, 20, 22, 26, 27, 29, 30, 36, 37, 38, 39])
        groundEvent = MIT_Ground_month

    elif type1 ==2:# for enron
        Enron_Ground_month = set([12, 14, 20, 21, 23, 24, 25, 25, 26, 28, 29, 30, 32, 32, 33, 36, 36, 36, 36, 37, 38, 39, 39, 39, 40])
        groundEvent = Enron_Ground_month
    elif type1 == 4:# for cosponsor
        groundEvent = set([3, 7])
    elif type1 == 5:
        groundEvent = set([3,4,5])
    else:
        toynetwork0406_Ground = set([1, 3, 4, 6, 7, 9, 10, 12])
        groundEvent = toynetwork0406_Ground

    cpDetector = ChangePointDetection(window, alpha,boots)
    cpDetector.detectAnomaliesInSequence(networkData,sequenceLen,nodeSize)
    print(cpDetector.cpL)
    # pass
    p_c = set(cpDetector.cpL)

    precision, recall, fvalue, fpr = Evaluation(p_c, groundEvent, sequenceLen)
    return precision, recall, fvalue, fpr


def main(**kw):
    # networkData = kw['data']
    # type1 = kw['type1']
    # window = kw['window']
    # print(type1)
    # print(window)
    # type1 = int(type1)
    # window = int(window)
    # networkData = 'G:\CodeSet\workspace\HGCN\simulate_100n_a3a4a5\simulate_100n_a3a4a5.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\simulate_24g_100n_a3a4a5\simulant_24g_100n_a3a4a5.mtx'
    # US senate
    networkData = 'E:\Source\Python Learning\Weekly\Compare\Congress\sample.mtx'
    # 安然邮件数据 按月   0.0正常PCA
    # networkData = "E:\Source\Python Learning\source2\Sample_Mat.mtx"  # 所有样本的矩阵
    # MIT蓝牙数据以周   0.01正常PCA
    # networkData = "E:\Source\Python Learning\Weekly\\buletooth\Sample_MIT.mtx"


    # 10
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_10g_50n\SBM_10g_50n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_10g_100n\SBM_10g_100n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_10g_500n\SBM_10g_500n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_10g_1000n\SBM_10g_1000n.mtx'
    # 30
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_30g_50n\SBM_30g_50n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_30g_100n\SBM_30g_100n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_30g_500n\SBM_30g_500n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_30g_1000n\SBM_30g_1000n.mtx'
    # 50
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_50g_50n\SBM_50g_50n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_50g_100n\SBM_50g_100n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_50g_500n\SBM_50g_500n.mtx'
    # networkData = 'G:\CodeSet\workspace\HGCN\GenNet\SBM_50g_1000n\SBM_50g_1000n.mtx'


    window = 5
    type1 = 5
    print("data", networkData)
    print("type: ", type1)
    print("window size: ", window)

    tempI = -1
    tempFL = []

    for i in range(10):
        precision, recall, fvalue, fpr = detect(networkData, window, type1)
        tempFL.append(fvalue)
    print(precision, recall, fvalue)
    # if len(tempFL) == 10:
    #     print(np.mean(tempFL), np.median(tempFL))
        # print tempFL.index(np.median(tempFL))

if __name__ == "__main__":

    # parser = OptionParser(usage="%prog -d data -o out -w window")
    #
    # parser.add_option(
    #     "-d", "--data",
    #     help=u"The file name of the data(networkData) to be extracted(includes the full path)"
    # )
    #
    # parser.add_option(
    #     "-t", "--type",
    #     help=u"The network ,0(Toy),1(MIT),2(Enron)"
    # )
    #
    # parser.add_option(
    #     "-w", "--window",
    #     help=u"The sliding window, default to be 2"
    # )
    # if not sys.argv[1:]:
    #     parser.print_help()
    #     exit(1)
    #
    # (opts, args) = parser.parse_args()
    # main(data=opts.data, type1=opts.type,window=opts.window)
    main()


