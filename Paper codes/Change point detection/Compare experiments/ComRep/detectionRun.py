import sys
import numpy as np
import scipy.io as si
import time

from ComRep import Sharding
import PCA_eig as PCA

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run network changepoint detection on a sequence of networks")

    parser.add_argument("dynamicNetworks",
                        help='Input sequence of network files: e.g. "network1.pairs network2.pairs network3.pairs"')

    args = parser.parse_args()
    file = args.dynamicNetworks

    thresholds = np.array(range(0, 10))/100.0

    sampmat = si.mmread(file).todense()  # 所有样本的矩阵
    edge_list = [str(item) for item in range(94)]  # 所有点的列表，矩阵邻接矩阵的行标签
    Recall = []; Precision = []; Fpr = []; F1 = []
    for index in thresholds:
        # if index != 0.02:
        i = index
        # t1 = time.clock()
        print('---------------------------------------')
        print("threshold:%f, splitting:" % i)
        test_anomaly, jacc = Sharding(samp_mat=sampmat, threshold=i, edge_list=edge_list).getEveDict()
        # print(test_anomaly)
        # print('############Time##########')
        # print(time.clock()-t1)