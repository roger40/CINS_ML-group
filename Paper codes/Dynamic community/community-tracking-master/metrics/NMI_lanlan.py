# from __future__ import division
from sklearn import metrics
import numpy as np

class NMI:
    def __init__(self, comms1, comms2):
        '''
        comms1: ground truth
        comms2: pred
        '''
        self.nodes1 = self.get_node_assignment(comms1)
        self.nodes2 = self.get_node_assignment(comms2)
        self.nodes = list(set().union([node for i, com in comms2.items() for node in com],
                                      [node for i, com in comms1.items() for node in com]))
        # print(len(self.nodes))
        # labels_true, labels_pred = self.observed()
        # self.results = metrics.normalized_mutual_info_score(labels_true, labels_pred)
        self.results = self.get_nmi(comms1, comms2)


    def get_node_assignment(self, comms):
        """
        returns a dictionary with node-cluster assignments of the form {node_id :[cluster1, cluster_3]}
        :param comms:
        :return:
        """
        nodes = {}
        for i, com in comms.items():
            for node in com:
                try:
                    nodes[node].append(i)
                except KeyError:
                    nodes[node] = [i]
        return nodes

    def observed(self):
        label_gt = []
        label_pr = []
        for u in list(self.nodes):
            try:
                label_gt.append(self.nodes1[u][0])
            except:
                # print('1')
                label_gt.append(-11)
            try:
                label_pr.append(self.nodes2[u][0])
            except:
                # print('-1')
                label_pr.append(-11)
        return label_gt, label_pr

    def get_matrix(self, comms1, comms2):
        """
        compute matrix C(i,j) is the number of overlapping elements in cluster i and cluster j
        comms1: ground truth
        comms2: predicted cluster
        :return:
        """
        gt = [k for k in comms1.keys()]
        gt_l = len(gt)
        pc = [k for k in comms2.keys()]
        pc_l = len(pc)
        C = np.zeros((gt_l, pc_l))
        for i in range(gt_l):
            c1 = set(comms1[gt[i]])
            # print(c1)
            for j in range(pc_l):
                c2 = set(comms2[pc[j]])
                # print(c2)
                C[i, j] = len(c1.intersection(c2))
        return C

    def get_mi_score(self, inter_mat):
        """
        get mutual information
        :return: score
        """
        inter_mat = 1.0 * np.array(inter_mat)
        x, y = inter_mat.shape
        # print(x, y)
        n = len(self.nodes)*1.0
        x_sum = np.sum(inter_mat, axis=1)  # sum of columns
        y_sum = np.sum(inter_mat, axis=0)  # sum of rows
        mi = 0
        for i in range(x):
            n_i = x_sum[i]
            # print(n_i)
            for j in range(y):
                n_j = y_sum[j]
                # print(n_j)
                if inter_mat[i, j] <= 0:
                    mi += 0
                else:
                    mi += (inter_mat[i, j]/n)*np.log((n*inter_mat[i, j])/(n_i*n_j))
                    # print((n*inter_mat[i, j]), (n_i*n_j))
        return mi

    def get_h_score(self, inter_mat):
        """
        get entropy
        :return: score
        """
        inter_mat = 1.0 * np.array(inter_mat)
        x, y = inter_mat.shape
        n = len(self.nodes)*1.0
        x_sum = np.sum(inter_mat, axis=1)  # sum of columns
        y_sum = np.sum(inter_mat, axis=0)  # sum of rows
        h_x = 0
        for i in range(x):
            n_i = x_sum[i]
            if n_i <= 0:
                h_x += 0
            else:
                h_x += (n_i/n)*np.log(n_i/n)
        h_y = 0
        for i in range(y):
            n_i = y_sum[i]
            if n_i <= 0:
                h_y += 0
            else:
                h_y += (n_i/n)*np.log(n_i/n)
        return h_x*h_y

    def get_nmi(self, comms1, comms2):
        inter_mat = self.get_matrix(comms1, comms2)
        # print(inter_mat)
        mi = self.get_mi_score(inter_mat)
        h_xy = self.get_h_score(inter_mat)
        if h_xy <= 0:
            return 0
        return mi/np.sqrt(h_xy)

# if __name__ == '__main__':
    # comms1 = {1: [5, 6, 7], 2: [3, 4, 5], 3: [6, 7, 8]}
    # comms2 = {1: [5, 6, 7], 2: [3, 4, 6], 3: [6, 7, 8]}
#     comms3 = {1: [5, 6, 7], 2: [6, 7, 8], 3: [3, 4, 5]}
#     comms4 = {0: ['1-t0', '2-t0', '3-t0', '4-t0', '1-t1', '2-t1',  '3-t1','4-t1', '1-t2','2-t2','3-t2','4-t2'],
#               1: ['11-t1', '12-t1', '13-t1'],
#               2: ['5-t2', '6-t2', '7-t2', '5-t0', '6-t0', '7-t0']}
#     comms5 = {1: ['11-t1', '12-t1', '13-t1'],
#               2: ['1-t0', '2-t0', '3-t0', '4-t0', '1-t1', '2-t1',  '3-t1','4-t1', '1-t2','2-t2','3-t2','4-t2'],
#               3: ['5-t2', '6-t2', '7-t2', '5-t0', '6-t0', '7-t0']}
#     comms6 ={0: ["1-t0", "2-t0", "3-t0", "4-t0","1-t1", "2-t1", "3-t1", "4-t1", "1-t2", "2-t2", "3-t2",  "4-t2"],
# 			 1: ["5-t0", "6-t0", "7-t0", "8-t0", "5-t2", "6-t2", "7-t2", "8-t2"]}
#     comms7 = {0: ["1-t0", "2-t0", "3-t0", "4-t0", "1-t1", "2-t1", "3-t1", "4-t1", "1-t2", "2-t2", "3-t2", "4-t2"],
#               1: ["5-t0", "6-t0", "7-t0", "8-t0", "5-t2", "6-t2", "7-t2", "8-t2"]}
#     comms1 = {1: [5, 6, 7]}
#     comms2 = {1: [5, 6, 7]}
#     nmi = NMI(comms1, comms2)
#     print(nmi.results)
