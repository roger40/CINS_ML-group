#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/27
# @Author  : Wang Biao
# @Site    :
# @File    : LoadData.py
# @Software: PyCharm

import torch
import pickle as pkl
import sys
import numpy as np
import scipy.sparse as sp
import networkx as nx
from community import best_partition
from networkx.algorithms import community


# from keras.utils import to_categorical
# from pylouvain import PyLouvain


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def loadData(dataset_str, precent, method):
    """
        Loads input data from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    '''保存真实label'''
    # np.savetxt('./labels/'+dataset_str+'_labels.txt', labels)
    # exit()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))

    # idx_train = range(int(len(y)*1))  #这个是之前的做法，有点粗暴
    '''在这里开始进行修改idx_train'''
    # 若需要各类别相同减少标签样本，对idx_train进行修改就好了
    labels_ = list(range(labels.shape[1]))
    idx_train = list(idx_train)
    # print('idx_train:', idx_train)
    nums = len(idx_train)
    # 每个类别的数据需要减少的数目, 20是每个类别原有的数目
    reduce_num = 20 * (10 - precent * 10) / 10
    # print('reduce_num:', reduce_num)
    dic = {i: 0 for i in labels_}
    # print(dic)
    for i in range(len(idx_train)):
        label = list(labels[nums - 1 - i]).index(1)  # 分析该点的标签,从尾部开始分析
        if dic[label] == reduce_num:
            continue
        else:
            dic[label] = dic[label] + 1
            idx_train.remove(nums - 1 - i)  # 删除指定数据点
        if sum(dic.values()) == len(labels) * reduce_num:
            break
    # print('idx_train:', idx_train)
    # print(len(idx_train))
    '''idx_train修改结束'''
    # exit()
    # 从后面进行删除

    idx_val = range(len(y), len(y) + 500)
    '''7月17日改增加一千个测试点'''
    idx_test_add = range(len(y)+500, len(y)+1500)
    # print(type(idx_test_add))
    idx_test = list(idx_test)+list(idx_test_add)
    '''7月17日改增加一千个测试点'''

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    '''
    自行进行社团划分的，得到各自的社团标签
    '''
    G = nx.from_numpy_matrix(adj.A)
    nodes = G.nodes
    if method == 'CNM':
        '''基于模块度最大化'''
        communities = community.greedy_modularity_communities(G)
        communities1 = sorted(map(sorted, communities))
    elif method == 'Louvain':
        communities = best_partition(G, random_state=2020)
        keys = communities.keys()
        sorted(keys)
        label_list = []
        for key in keys:
            label_list.append(communities[key])
        y_class = label_list
        return adj, features, labels, train_mask, val_mask, test_mask, y_class
    elif method == 'ALPA':
        communities = community.asyn_lpa_communities(G, seed=2020)
        communities1 = sorted(map(sorted, communities))
    # method == 'SLPA'
    else:
        communities = community.label_propagation_communities(G)
        communities1 = sorted(map(sorted, communities))

    '''
    自行进行社团划分的，得到各自的社团标签
    '''
    node_label_dic = {}
    for comm_label in range(len(communities1)):
        for node in communities1[comm_label]:
            node_label_dic[node] = comm_label
    '''每个节点对应的value值为社团标签'''

    y_label_list = []
    for node in range(len(nodes)):
        y_label_list.append(node_label_dic[node])
    # y_class = np.array(one_hot(torch.tensor(y_label_list), len(communities1)))
    y_class = y_label_list
    # print(y_class)
    # np.savetxt('./labels_comm/'+dataset_str+'_gm.txt', np.array(y_class).reshape(len(y_class), 1))

    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_class
    return adj, features, labels, train_mask, val_mask, test_mask, y_class


if __name__ == '__main__':
    adj, features, labels, train_mask, val_mask, test_mask, y_class = \
        loadData(dataset_str='pubmed', precent=1, method='CNM')
    # print(len(set(list(y_class))))
    # exit()
    labels = torch.tensor(labels)
    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)
    train_labels = labels[train_mask]
    val_labels = labels[val_mask]
    test_labels = labels[test_mask]


    def main(train_labels):
        a = torch.argmax(train_labels, dim=1)
        # print('a:', a)
        b = list(np.array(a))
        # print('b:', b)
        print('总数：', len(train_labels))
        for i in list(set(b)):
            print(str(i) + ':', b.count(i))
        return


    print('cora')
    print('train:')
    main(train_labels)
    print('val:')
    main(val_labels)
    print('test:')
    main(test_labels)
    exit()
    print(train_mask)
    print((np.array(train_mask) > 0).sum())  # 统计布尔值
    print((np.array(val_mask) > 0).sum())
    print((np.array(test_mask) > 0).sum())