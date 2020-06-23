# -*- coding=utf-8 -*-

"""
    input: the txn^2 matrix of snapshots, where t is the number of snapshots and n is the number of nodes
        nodes is the total nodes in each snapshots. the threshold, a fixed variable restrict the interval [0.01, 0.2]
    goal: to find the border of events, anomaly
        (use threshold to judge whether the new snapshot network belong to the sample of  events)
    sample: xs, which is the sequence about the set of events
        (the similarity values of snapshots of one event)
    output: the sequence of anomaly

"""

import numpy as np
import scipy.io as si
# np.linalg.eig 函数返回值为特征值，和特征向量对应的列矩阵
# import PCA  # PCA.py
# import Detection.Weekly.PCA_eig as PCA
import PCA_eig as PCA
import time

"""
    compute the similarity between events event1 and next snapshot event2
"""
class Conpute(object):
    event1 = None
    event2 = None
    edge_list = None
    def __init__(self, event1, event2, edge_list):
        self.event1 = event1
        self.event2 = event2
        self.edge_list = edge_list

    def getFeature(self, flage):
        th = 0
        # th = 1e-15
        feature_dict_pos = {}
        feature_dict_neg = {}
        if flage == 0:
            m_mat = self.event1
        elif flage == 1:
            m_mat = self.event2
        row, col = m_mat.shape
        for i in range(row):
            feature_dict_pos[self.edge_list[i]] = []

            feature_dict_neg[self.edge_list[i]] = []

            for j in range(col):
                if m_mat[i, j] <= th:
                    feature_dict_neg[self.edge_list[i]].append(self.edge_list[j])

                if m_mat[i, j] > th:
                    feature_dict_pos[self.edge_list[i]].append(self.edge_list[j])
        return feature_dict_pos, feature_dict_neg

    def jaccSimalary(self, feature_dict_pos1, feature_dict_pos2):
        # print "the len of dict"
        diclen = len(feature_dict_pos2)

        # print diclen
        jaccard = 0
        for index in feature_dict_pos1:
            same_edge = 0
            all_edge = 0
            all_edge = len(feature_dict_pos2[index]) + len(feature_dict_pos1[index])
            if all_edge != 0:
                for obj in feature_dict_pos1[index]:
                    if obj in feature_dict_pos2[index]:
                        same_edge = same_edge + 1
                jaccard = jaccard + (1.0 * same_edge / (all_edge - same_edge))

            # else:
            #     jaccard = jaccard + 1

        return jaccard / diclen

    def getPJacc(self):
        # print(self.edge_list)
        feature_dict_pos1, feature_dict_neg1 = self.getFeature(0)
        feature_dict_pos2, feature_dict_neg2 = self.getFeature(1)
        # print(feature_dict_pos1, feature_dict_pos2)
        Jacarrd_1_2 = self.jaccSimalary(feature_dict_pos1, feature_dict_pos2)
        return Jacarrd_1_2

    def getNJacc(self):
        # print(self.edge_list)
        feature_dict_pos1, feature_dict_neg1 = self.getFeature(0)
        feature_dict_pos2, feature_dict_neg2 = self.getFeature(1)
        # print(feature_dict_pos1, feature_dict_pos2)
        Jacarrd_1_2 = self.jaccSimalary(feature_dict_neg1, feature_dict_neg2)
        return Jacarrd_1_2

    def getPNsim(self):  # concert postive and negative edges because there are both of them in primal graph
        s_n = self.getNJacc()
        s_p = self.getPJacc()
        # print(s_p, s_n)
        return (s_n+s_p)*0.5

    def positionMat(self, sample):
        tem = np.zeros_like(sample)
        x, y = sample.shape
        for i in range(x):
            for j in range(y):
                if sample[i, j] > 0:
                    tem[i, j] = 1
                    # tem[i, j] = sample[i, j]
                else:
                    tem[i, j] = 0
        return tem

    def getSimilarity(self):  # only concert positive edges and zeros edges as primal graph is positive
        I = np.ones_like(self.event1)
        E1 = self.positionMat(self.event1)
        E2 = self.positionMat(self.event2)
        l = len(I)
        xnor = np.multiply(E1, E2) + np.multiply((I - E1), (I - E2))
        # print(np.sum(xnor))
        jacc = np.sum(xnor).real/(l*l)
        return jacc




"""
    goal: find the borders of events, the anomalies
"""
class Sharding(object):
    samp_mat = None    # the matrix of sample
    sampCount = 0     # the number of row of matrix
    # schedule = None    # the real anomaly
    edge_list = None    # the label
    # test
    threshold = 0    # the judgement

    st = [0]  # sequence of the start times of events, and the first element is zero
    dt = []  # the sequence of number of each events
    et = []  # sequence of the end time of events, we assume st[i+1] == et[i]
    # array = []
    jacc = []  # the sequence of similarity during we comput them
    js = [1] # the jacc without PCA

    xs = []    # the similarity set of snapshots of one event
    anomaly = None    # the prediction of anomaly we tested
    # 2018/1/6
    # 修改threshold，event 收敛
    dyna = 0

    def __init__(self, samp_mat, threshold, edge_list):
        self.samp_mat = samp_mat
        self.threshold = threshold
        # self.schedule = schedule
        self.edge_list = edge_list
        self.sampcount = len(self.samp_mat)
        self.st = [0]  # 时间窗样本开始时间 从0开始
        self.dt = []  # 第i开始的时间窗大小
        self.et = []  # 第i个样本的结束时间 st[i+1] == et[i]
        self.jacc = [1]  # 当前开始的dt[i]时间窗的 jaccard值
        self.xs = [1]

    # 更新XS里面的累计 jaccard值
    def updateXS(self, events):  # 原时间样本
        event1, eigVect, meanRow = PCA.Pca1_6(events)  # 邻接矩阵
        temp = []
        for i in range(len(events)):
            events2 = events[i]
            event2 = PCA.pca2(events2, eigVect)
            Jacarrd_1_2 = Conpute(event1, event2, self.edge_list).getJacc()
            temp.append(Jacarrd_1_2)
        return temp

    # add 4
    def judJacc(self, events1, events2):

        # print("there is a new Search:")

        event1, eigvect, meanrow = PCA.pca1_6(events1)  # 邻接矩阵
        event2, _ = PCA.pca2(events2, eigvect)   # 前面的 主特征 处理后面的矩阵

        # jacarrd_1_2 = Conpute(event1, event2, self.edge_list).getPJacc()
        # 同或
        jacarrd_1_2 = Conpute(event1, event2, self.edge_list).getSimilarity()

        self.jacc.append(jacarrd_1_2)  # 前段和异常的相似度

        if len(self.xs) >= 1:

            temp = self.xs
            if np.mean(temp) - jacarrd_1_2 <= self.threshold:  # add 8
                self.xs.append(jacarrd_1_2)
                return True

            else:
                self.xs.append(jacarrd_1_2)
                # temat = events2
                # temat.shape = event2.shape
                # PCA_network(label, [meanRow, event1])
                return False
        else:
            self.xs.append(jacarrd_1_2)
            return True

    def eveSearch(self, result):
        time = self.st[-1]  # ==et[-1]前一个的结束时间
        # Result 的第一个样本
        odt = [1]  # eve1 的时间窗大小 time + odt[i] = xs[i] = eve1,eve2[i]
        self.xs = [1]  # ADD 4  xs=[] --> xs=[1]
        while odt[-1] < len(result):
            eve1 = result[0:odt[-1]]
            eve2 = result[odt[-1]]
            boo = self.judJacc(eve1, eve2)

            if boo is True:
                odt.append(odt[-1] + 1)
                continue
            else:
                break
        # print st[-1]
        # print(odt)
        # print("事件的搜索: "+str(self.xs))
        if len(self.xs) == 1:
            self.dt.append(0)  # 相应的dt   分片，不取出现异常的那个样本，所以  odt[-1]-1
            self.et.append(time)  # 事件结束时间
            self.st.append(time)  # 下一个事件开始的时间
            return np.mat([])
        else:

            """
            用来检测的样本，异常的话，要把它截断，不要他
            time=st[-1]=0, [0,1,2,3,4] 若odt[-1] = 2 则eve1=[0,1] eve2=[2]
            若没有异常 odt[-1] = 3 eve1=[0,1,2] eve2=[3]
            若有异常 odt[-1] = 2 eve2摒弃，et=[1]=[time+1] dt=1=odt[-1]-1
            """
            self.dt.append(odt[-1])  # 相应的dt   分片，不取出现异常的那个样本，所以  odt[-1]-1
            self.et.append(time + self.dt[-1]-1)  # 事件结束时间
            self.st.append(time + self.dt[-1])  # 下一个事件开始的时间
            # print("%d \t %d \t %f" % (st[-2], st[-1], jacc[-1]))
            return result[self.dt[-1]:]  # 事件 和 剩下的事件集


    def getEveDict(self):
        # print "Jacarrd---> start:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st = [0]
        result = self.samp_mat
        while result.size != 0:
            # print Result.size
            result = self.eveSearch(result)
            # Event[(st[-2], st[-1])] = eve
        temp = self.st[1:-1]
        self.anomaly = temp
        print(self.anomaly)
        print("with PCA", self.jacc)

        return temp, self.jacc

def verfication(anomaly, schedule, n):
        temp = anomaly
        tp = 0
        fp = 0
        # print("index:%f, splitting:(anomal is end+1)" % self.threshold)
        # print("start\t\tend \t\tJanomaly")
        # for i in range(len(self.jacc)):
        #     print("%d\t\t\t%d\t\t\t%f" % (self.st[i], self.et[i], self.jacc[i]))

        for item in temp:
            if item in schedule:
                tp += 1
            else:
                fp += 1         # 检测错误

        fn = len(schedule)-tp    # 未检测出的异常点数
        tn = (n - fn - tp) - fp  # 本来是正常的，也确实没检测为异常
        accuracy = (tp + tn) / n
        recall = tp / len(schedule)
        if tp == 0:
            precision = 0
            f1 = 0
        else:
            precision = tp / len(temp)
            f1 = 2*recall * precision / (recall + precision)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        # self.Z.append([FPR, TPR])
        print("recall: %d/%d = %f    precision: %d/%d = %f  假阳性:%f   F1:%f"
              % (tp, len(schedule), recall,
                 tp, (tp + fp), precision, fpr, f1))

        return recall, precision, fpr, f1


if __name__ == "__main__":

    # # 模拟网络数据
    # # # kqpath = "E:\Source\\time_network\Sample_Mat.mtx"
    # # kqpath = "E:\Source\\time_network\Sample_Manualt.mtx"
    # # sampmat = si.mmread(kqpath).todense()  # 所有样本的矩阵
    # # # print(SampMat)
    # # # # edge_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']  # 所有点的列表，矩阵邻接矩阵的行标签
    # # edge_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # # newanomal = oldanomal = set([1, 2, 5])
    #

    # # 切割数据
    # gopath = "E:\Source\\test\\\Sample_Manualt.mtx"
    # sampmat = si.mmread(gopath).todense()  # 所有样本的矩阵
    # edge_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    # anomal = oldanomal = set([1, 2, 5])


    # #脑功能网络
    #
    # sampmat = si.mmread('E:\\DATASet\\ping\\Nor67_Pat53.mtx').todense()
    # edge_list = [str(item) for item in range(90)]
    # anomal = set([67])


    # 0.16正常PCA  11个网络，20个节点
    # sampmat = si.mmread("E:\Source\Ting\\toynetwork0406.mtx").todense()  # 所有样本的矩阵
    # edge_list = [str(item) for item in range(9)]  # 所有点的列表，矩阵邻接矩阵的行标签
    # anomal = set([1,3,4,6,7,9,10,12])



    # 安然邮件数据以周 0.01正常PCA
    # sampmat = si.mmread("E:\Source\Python Learning\Weekly\\mail\Sample_Enron_one_week.mtx").todense()  # 所有样本的矩阵
    # edge_list = [str(item) for item in range(147)]  # 所有点的列表，矩阵邻接矩阵的行标签
    # # # # newanomal = set([30, 37, 63, 68, 74, 78, 84, 86, 89, 98, 102, 106, 110, 115, 120, 131, 133, 134, 134, 135, 143, 144, 145, 145, 149])
    # # # # dyn
    # # # oldanomal = set([30, 37, 60, 68, 74, 78, 84, 84, 89, 98, 102, 106, 114, 116, 120, 131, 133, 134, 134, 135, 142, 144, 144, 144, 149])
    # # # # ghrg
    # anomal = set(
    #     [30, 37, 63, 68, 74, 78, 84, 86, 89, 98, 102, 106, 114, 116, 120, 131, 133, 134, 134, 135, 142, 144, 144, 144,
    #      149])



    # 安然邮件数据 以2周    0.06正常PCA
    # sampmat = si.mmread("E:\Source\Python Learning\Weekly\\mail\Sample_Enron_two_week.mtx").todense()  # 所有样本的矩阵
    # edge_list = [str(item) for item in range(147)]  # 所有点的列表，矩阵邻接矩阵的行标签
    # oldanomal = set([30, 36, 60, 68, 74, 78, 84, 85, 89, 98, 102, 106, 114, 116, 120, 131, 133, 134, 134, 135, 142, 144, 144, 144, 149])
    # newanomal = set([30, 37, 63, 68, 74, 78, 84, 86, 89, 98, 102, 106, 110, 115, 120, 131, 133, 134, 134, 135, 143, 144, 145, 145, 149])
    # temp = []
    # # for i in oldanomal:
    # for i in newanomal:
    #     temp.append(int(i/2))
    # # oldanomal = set(temp)
    # newanomal = set(temp)


    # MIT蓝牙数据以 2周   0.05正常PCA
    # sampmat = si.mmread("E:\Source\Python Learning\Weekly\\buletooth\Sample_MIT_two_week").todense()  # 所有样本的矩阵
    # edge_list = [str(item) for item in range(94)]  # 所有点的列表，矩阵邻接矩阵的行标签
    # oldanomal = set([1, 7, 8, 10, 10, 12, 14, 15, 15, 18, 22, 28, 29, 36, 37, 37, 39])
    # newanomal = set([1, 7, 8, 10, 11, 12, 14, 15, 16, 18, 20, 22, 26, 27, 29, 30, 36, 37, 37, 39])
    # temp = []
    # # for i in oldanomal:
    # for i in newanomal:
    #     temp.append(int(i/2))
    # # oldanomal = set(temp)
    # newanomal = set([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 18, 19])

    #
    # # # MIT蓝牙数据以周   0.01正常PCA
    sampmat = si.mmread("E:\Source\Python Learning\Weekly\\buletooth\Sample_MIT.mtx").todense()  # 所有样本的矩阵
    edge_list = [str(item) for item in range(94)]  # 所有点的列表，矩阵邻接矩阵的行标签
    anomal = set([1, 7, 8, 10, 11, 12, 14, 15, 16, 18, 20, 22, 26, 27, 29, 30, 36, 37, 38, 39])

    # directed weight networks
    # sampmat = si.mmread("E:\RealityMining\RealityMining\Week_Sample_MIT.mtx").todense()  # 所有样本的矩阵

    # 安然邮件数据 按月   0.0正常PCA
    # sampmat = si.mmread("E:\Source\Python Learning\source2\Sample_Mat.mtx").todense()  # 所有样本的矩阵
    # edge_list = [str(item) for item in range(147)]  # 所有点的列表，矩阵邻接矩阵的行标签
    # anomal = set(np.array([12, 14, 20, 21, 23, 24, 25, 25, 26, 28, 29, 30, 32, 32, 33, 36, 36, 36, 36, 37, 38, 39, 39, 39, 40]))

    # timeline
    # anomal = [6, 7, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    # directed weight networks
    # sampmat = si.mmread("E:\enron_mail_20150507\Month\directedSampleMat.mtx").todense()  # 所有样本的矩阵

    # US senate
    # sampmat = si.mmread("E:\Source\Python Learning\Weekly\Compare\Congress\sample.mtx")
    # edge_list = [str(item) for item in range(222)]
    # anomal = set([3, 7])

    # SBM模拟
    # sampmat = si.mmread('G:\CodeSet\workspace\HGCN\SBM_node_100\sample.mtx')
    # edge_list = [str(item) for item in range(100)]
    # anomal = set([3, 5])

    # SBM 6个网络3个异常
    # sampmat = si.mmread('G:\CodeSet\workspace\HGCN\simulate_100n_a3a4a5\simulate_100n_a3a4a5.mtx').todense()
    # edge_list = [str(item) for item in range(100)]
    # anomal = set([3, 4, 5])

    # SBM 24个网络7个异常
    # sampmat = si.mmread('G:\CodeSet\workspace\HGCN\simulate_24g_100n_a3a4a5\simulant_24g_100n_a3a4a5.mtx').todense()
    # edge_list = [str(item) for item in range(100)]
    # anomal = set([7,12,15,16,17,20,21])

    # Ei = 0.26
    # Mi = 0.14
    # ii = 0.27
    # print("threshold    recall   percision   accurancy")
    # index = 100
    # 模拟 0.0    Enron  0.1   MIT  0.16
    #
    # n = len(sampmat)
    # test_anomaly, jacc = Sharding(samp_mat=sampmat, threshold=0.01, schedule=anomal, edge_list=edge_list).getEveDict()
    # print(test_anomaly)
    # recall, precision, fpr, f1 = verfication(test_anomaly, anomal, n)

    # ROC曲线绘制
    # # FPR = []; TPR = []
    # T
    # sampmat = si.mmread("E:\Source\Ting\\toynetwork0304.mtx").todense()  # 所有样本的矩阵
    # edge_list = [str(item) for item in range(9)]  # 所有点的列表，矩阵邻接矩阵的行标签
    #
    # anomal = [1, 3, 4, 6, 7, 9, 10, 12]
	# sampmat = T*(n^2)	schedule = 1*{No. of groudtruth}	edge_list = 1*{number of nodes}

    n = len(sampmat)
    Recall = []; Precision = []; Fpr = []; F1 = []
    for index in range(0, 20):
        i = index/100.0
      # if i == 0.02:
        t1 = time.clock()
        print("index:%f, splitting:" % i)
        test_anomaly, jacc = Sharding(samp_mat=sampmat, threshold=i, edge_list=edge_list).getEveDict()
        #print(test_anomaly)
        t2 = time.clock()
        recall, precision, fpr, f1 = verfication(test_anomaly, anomal, n)
        print('############Time##########')
        print(t2-t1)
    #     Recall.append(recall)
    #     Precision.append(precision)
    #     Fpr.append(fpr)
    #     F1.append(f1)
    # print(Recall, "\n", Precision, '\n', Fpr, '\n', F1)
