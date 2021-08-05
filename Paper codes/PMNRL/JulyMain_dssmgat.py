#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/19
# @Author  : Wang Biao
# @Site    :
# @File    : JulyMain_dssmgat.py
# @Software: PyCharm

'''
搭建基于mutiltaske的gat模型
改成稀疏版本计算的模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from GAT import GraphAttentionLayer, SpGraphAttentionLayer
# from GAT_gridesearch import GraphAttentionLayer, SpGraphAttentionLayer

import scipy.sparse as sp
import numpy as np
import matplotlib as mpl
# mpl.use('Agg') #服务器上保存图片
import matplotlib.pyplot as plt
from JulyLoadData import loadData
import time
# import torchsnooper

from kmeans import kmeans, l2_distance
from sklearn.cluster import KMeans

import random

class MutiltaskSpGatNet(nn.Module):
    '''
    定义所使用的基于多任务的图卷积模型, gcn2_1和gcn2_2属于并行关系
    '''
    def __init__(self, input_dim, hidden_dim, output_dim2_1, output_dim2_2, dropout, alpha, nheads, nout_heads):
        '''
        :param input_dim: 输入维度
        :param hidden_dim: 共享层输出维度
        :param output_dim2_1: 任务1输出维度，即任务1类别数
        :param output_dim2_2: 任务2输出维度，即任务2类别数
        :param dropout: 计算注意力时用到的dropout，以及搭建GAT网络时也会用到
        :param alpha: leakyrelu的参数，这个是在计算注意力时用到的，所有gat层都共用先
        :param nheads: 注意力组数
        '''
        super(MutiltaskSpGatNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.alpha = alpha
        self.output_dim2_1 = output_dim2_1
        self.output_dim2_2 = output_dim2_2
        self.nheads = nheads
        self.nout_heads = nout_heads
        '''第一层gat'''
        self.attentions = [SpGraphAttentionLayer(in_features=self.input_dim, out_features=self.hidden_dim, dropout=self.dropout, alpha=alpha, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        '''第二层gat'''

        self.gat2_1 = SpGraphAttentionLayer(in_features=self.hidden_dim*self.nheads, out_features=self.output_dim2_1, dropout=self.dropout, alpha=alpha, concat=False)

        '''只对我们的主任务设置多个头'''
        # self.gat2_2 = SpGraphAttentionLayer(in_features=self.hidden_dim*self.nheads, out_features=self.output_dim2_2, dropout=self.dropout, alpha=alpha, concat=False)
        self.out_attentions = [SpGraphAttentionLayer(in_features=self.hidden_dim * self.nheads, out_features=self.output_dim2_2, dropout=self.dropout, alpha=alpha, concat=False) for _ in range(self.nout_heads)]

        for i, out_attention in enumerate(self.out_attentions):
            self.add_module('attention_{}'.format(i+self.nheads), out_attention)


    '''
    logits2_1:这个表示用于自监督，无监督节点分类
    logits2_2:这个表示用于半监督学习
    '''
    def forward(self, feature_x, adj_norm):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        h = F.dropout(feature_x, self.dropout, training=self.training)
        h = torch.cat([att(h, adj_norm) for att in self.attentions], dim=1)
        logits_share = h
        h = F.dropout(logits_share, self.dropout, training=self.training)
        logits2_1 = F.elu(self.gat2_1(h, adj_norm))
        logits2_2 = F.elu(sum([out_att(h, adj_norm) for out_att in self.out_attentions])/self.nout_heads)
        return logits2_1, logits2_2, logits_share

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def normalization(A):
    # A = A.to(device) # 9.11
    # A = A+I
    A = torch.tensor(A, dtype=torch.float)
    A_trans = A + torch.eye(A.size(0), dtype=torch.float)
    rowsum = A_trans.sum(1)  # 按行加
    rowsum_trans = [d ** (-0.5) for d in rowsum]
    # D = t.diag(t.Tensor(rowsum))  # 度矩阵
    D_trans = torch.diag(torch.tensor(rowsum_trans))  # 对角矩阵
    return D_trans.mm(A_trans).mm(D_trans)


def test(model, adj_norm, feature_X, test_x_part, train_y, criterion):
    '''
    :param model:
    :param adj_norm:
    :param feature_X:
    :param test_x: 测试的节点的编号
    :param train_y: 全部节点的标签
    :return:
    '''
    model.eval()
    with torch.no_grad():
        logits2_1, logits2_2, logits_share = model(feature_X, adj_norm)
        test_x_logits = logits2_2[test_x_part]
        loss = criterion(logits2_2[test_x_part], torch.argmax(train_y[test_x_part], dim=1))
        # 这个错了，没有进行softmax，好像并不影响
        predict_y = test_x_logits.max(1)[1]
        accuracy = torch.eq(predict_y, torch.argmax(train_y[test_x_part], dim=1)).float().mean()
    return accuracy, loss

# @torchsnooper.snoop()
def train(model, optimizer, epochs, adj_norm, criterion, weight_loss2_1, weight_loss2_2,
          feature_X, cluster_num, train_y2_2, train_x_part, test_x_part, val_x_part,
          result_path, dataset_name):
    '''
    :param model:
    :param epochs:
    :param adj_norm:
    :param criterion:
    :param weight_loss2_1:
    :param weight_loss2_2:
    :param feature_X: 节点的初始特征
    :param train_y2_1: 自行进行社团划分获得的标签信息
    :param train_y2_2: 数据的真实标签信息
    :param train_x_part:
    :param test_x_part:
    :param val_x_part:
    :return:
    '''
    loss_history = []
    loss2_1_history = []
    loss2_2_history = []
    val_loss_history = []
    loss_BNM_history = []
    loss_reg_l1_history = []
    loss_reg_l2_history = []
    train_acc_history = []
    test_acc_history = []
    val_acc_history = []
    params_list = []
    best_loss = 1e9
    best_val_loss = 1e9
    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0
    # model.train() 之前写错了
    for epoch in range(epochs):
        model.train()
        logits2_1, logits2_2, logits_share = model(feature_X, adj_norm)
        # l2_reg = torch.tensor(0, dtype=torch.float)
        # l1_reg = torch.tensor(0, dtype=torch.float)
        # for param in model.parameters():
        #     l2_reg += torch.norm(param, 2, dtype=torch.float)**2
        #     l1_reg += torch.norm(param, 1, dtype=torch.float)
        # l2_reg = l2_reg.to(device)
        # l1_reg = l1_reg.to(device)

        '''
        #准备对共享层参数进行一个稀疏化
        paras = model.named_parameters()

        for name, param in paras:
            params_list.append(param)
            break
        weight_gcn1 = params_list[-1]
        # print('weight_gcn1:', weight_gcn1)
        gcn1_l1_reg = torch.norm(weight_gcn1, 1, dtype=torch.float)
        gcn1_l1_reg = gcn1_l1_reg.to(device)
        #共享层参数l1正则计算完毕
        '''

        '''
        kmeans = KMeans(n_clusters=cluster_num, n_init=20)
        y_pred = kmeans.fit_predict(logits2_1.detach().cpu().numpy())
        cluster_centers = torch.tensor(kmeans.cluster_centers_).to(device)
        '''
        # 7月19日
        pt_whitened = logits2_1
        codebook, distortion = kmeans(obs=pt_whitened, k=cluster_num, distance_function=l2_distance,
                                      batch_size=6400000, iter=20)
        cluster_centers = codebook

        dis_tesnsor = l2_distance(pt_whitened, codebook)  # [n, k]
        cluster_label = dis_tesnsor.argmin(1)
        y_pred = cluster_label.detach().cpu().numpy()
        # 7月19日

        v = 1
        q = 1.0 / (1.0 + torch.sum(torch.pow(logits2_1.unsqueeze(1) - cluster_centers, 2), 2) / v)
        q = q.pow((v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        p = target_distribution(q)


        # loss2_1 = criterion(logits2_1, train_y2_1)
        loss2_1 = F.kl_div(q.log(), p, reduction='batchmean')

        loss2_2 = criterion(logits2_2[train_x_part], torch.argmax(train_y2_2[train_x_part], dim=1))
        # loss_BNM = -torch.norm(F.softmax(logits2_2, dim=1), 'nuc')*0.5

        '''加l1正则，l2正则'''
        # loss_total = weight_loss2_1*loss2_1+weight_loss2_2*loss2_2+weight_reg_l1*l1_reg+weight_reg_l2*l2_reg
        '''不加正则'''
        loss_total = weight_loss2_1*loss2_1+weight_loss2_2*loss2_2
        '''加共享层参数的l1正则'''
        # loss_total = weight_loss2_1*loss2_1+weight_loss2_2*loss2_2+weight_reg_l1*gcn1_l1_reg
        loss_total = loss_total.to(device)
        # loss_total = weight_loss2_1*loss2_1+weight_loss2_2*loss2_2+weight_reg*l2_reg
        # print(loss_total)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        train_acc, _=test(model=model, adj_norm=adj_norm, feature_X=feature_X, test_x_part=train_x_part, train_y=train_y2_2, criterion=criterion)
        test_acc, _= test(model=model, adj_norm=adj_norm, feature_X=feature_X, test_x_part=test_x_part, train_y=train_y2_2, criterion=criterion)
        val_acc, val_loss= test(model=model, adj_norm=adj_norm, feature_X=feature_X, test_x_part=val_x_part, train_y=train_y2_2, criterion=criterion)

        loss2_1_history.append((weight_loss2_1*loss2_1).item())
        loss2_2_history.append((weight_loss2_2*loss2_2).item())
        # loss_BNM_history.append(loss_BNM.item())
        # loss_reg_history.append(l2_reg.item())
        # loss_reg_l1_history.append((weight_reg_l1*gcn1_l1_reg).item())
        # loss_reg_l2_history.append((weight_reg_l2*l2_reg).item())
        loss_history.append(loss_total.item())
        train_acc_history.append(train_acc.item())
        test_acc_history.append(test_acc.item())
        val_acc_history.append(val_acc.item())
        val_loss_history.append(val_loss.item())
        # print('part_1:', weight_loss2_1*loss2_1, 'part_2:', weight_loss2_2*loss2_2, 'reg:', weight_reg*l2_reg/((feature_X[train_x_part].shape)[0]))

        # print('epoch:', epoch+1, 'Train Loss:', loss_total.item(),
        #       'Train acc:', train_acc.item(), 'Test acc:', test_acc.item(), 'Val acc:', val_acc.item(), 'Val Loss:', val_loss.item())

        print('epoch:', epoch, 'Train Loss:', loss_total.item(), 'loss_dssm:', loss2_1_history[-1], 'loss_classify:',
              loss2_2_history[-1], 'Train acc:', train_acc.item(), 'Test acc:', test_acc.item(), 'Val acc:', val_acc.item())



        # np.save(result_path+'rep/'+dataset_name+'/'+str(epoch+1)+dataset_name+'_comm_best.npy', logits2_1.detach().cpu().numpy())
        # np.save(result_path+'rep/'+dataset_name+'/'+str(epoch+1)+dataset_name+'_node_best.npy', logits2_2.detach().cpu().numpy())

        '''
        # 1、直接找最优结果
        if test_acc_history[-1] >= best_test_acc:
            best_epoch = epoch
            best_test_acc = test_acc_history[-1]
            # torch.save(model.state_dict(), best_model_name)
            # 保存表示
            np.save(result_path+'rep/'+dataset_name+'_comm_best.npy', logits2_1.detach().cpu().numpy())
            np.save(result_path+'rep/'+dataset_name+'_node_best.npy', logits2_2.detach().cpu().numpy())

            # np.save(best_rep_name_comm, logits2_1.detach().cpu().numpy())
            # np.save(best_rep_name_node, logits2_2.detach().cpu().numpy())
        '''

        '''
        # 2、直接找最优的val_acc, 同时要满足此时的test_acc要大于val_acc
        if val_acc_history[-1] > best_val_acc and test_acc_history[-1] > val_acc_history[-1]:
            best_epoch = epoch
            best_val_acc = val_acc_history[-1]
            best_test_acc = test_acc_history[-1]
        '''

        '''
        # 3、验证集的损失连续大于最优验证集损失若干次停止
        #    这种早停止方式可能不适合
        if val_loss_history[-1] < best_loss:
            best_epoch = epoch
            best_test_acc = test_acc_history[-1]
            best_loss = val_loss_history[-1]
            cnt_wait = 0
            # torch.save(model.state_dict(), best_model_name)
            # np.save(best_rep_name, logits.detach().numpy())
        else:
            cnt_wait += 1

        if cnt_wait == 50:
            print('early stop!!!')
            # torch.save(model.state_dict(), final_model_name)
            # np.save(final_rep_name, logits.detach().numpy())
            break
        # '''


        # 4、同时根据验证集的损失和验证集的准确率来找最优解，这种方法结果还行
        if val_acc_history[-1] >= best_val_acc or val_loss_history[-1] <= best_val_loss:
            # if val_acc_history[-1] >= best_val_acc and val_loss_history[-1] <= best_val_loss and test_acc_history[-1] >= val_acc_history[-1]:
            # if val_acc_history[-1] >= best_val_acc and val_loss_history[-1] <= best_val_loss and loss_history[-1] <= best_loss:
            if val_acc_history[-1] >= best_val_acc and val_loss_history[-1] <= best_val_loss:
            # if val_acc_history[-1] >= best_val_acc and loss_history[-1] <= best_loss:
                best_epoch = epoch
                best_test_acc = test_acc_history[-1]
                best_val_loss = val_loss_history[-1]
                best_val_acc = val_acc_history[-1]
                best_loss = loss_history[-1]
                best_y_pred = y_pred
                # 保存模型参数及优化器参数
                saved_dict = {
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict()
                }
                # torch.save(saved_dict, result_path+'model/'+dataset_name+'.pt')
                # np.savetxt(result_path + 'kmeanslabel/' + dataset_name + '.txt', best_y_pred.reshape(-1, 1))
                # np.save(result_path + 'rep/' + dataset_name + '_share.npy', logits_share.detach().cpu().numpy())
                # np.save(result_path+'rep/'+dataset_name+'_comm_best.npy', logits2_1.detach().cpu().numpy())
                np.save(result_path+'rep/'+dataset_name+'_node_best.npy', logits2_2.detach().cpu().numpy())


            cnt_wait = 0
        else:

            cnt_wait += 1
        if cnt_wait == 100:
            print('early stop!!!')
            break


        '''
        #5、规定至少运行400个epoch，并且val_loss_history > np.mean(val_loss_history[-(early_stop+1):-1])
        if epoch > 399 and val_loss_history[-1] > np.mean(val_loss_history[-400:-1]):
            best_epoch = epoch
            best_test_acc = test_acc_history[-1]
            print('early stop!!!')
            break
        '''

        '''
        #6、ping criteria：GL = 100*(Eval(t)/Eopt(t)-1), 其中Eopt(t)为到t之前(包含t)的最小Eval
        # 这个不适合？
        # GL = 100*(val_loss_history[-1]/best_val_acc-1)
        if val_loss_history[-1] < best_val_loss:
            best_epoch = epoch
            best_test_acc = test_acc_history[-1]
            best_val_loss = val_loss_history[-1]
        GL = 100*(val_loss_history[-1]/best_val_loss-1)
        print('GL:', GL)
        if GL > 20:   # 0.5相当于是一个阈值
            print('early stop!!!')
            break
        '''

        # 7、UP criteria，前面以尝试过，验证集损失连续增加若干次。

        '''
        # 8、PQ criteria
        # 设置一个滑动窗口
        if val_loss_history[-1] < best_val_loss:
            best_epoch = epoch
            best_test_acc = test_acc_history[-1]
            best_val_loss = val_loss_history[-1]
        k=10
        if (epoch+1)%10 == 0:
            GL_t = 100 * (val_loss_history[-1] / best_val_loss - 1)
            Pk_t = 1000 * (sum(loss_history[-k:-1])/(k*min(loss_history[-k:-1])) - 1)
            PQ = GL_t/Pk_t
            # 算出来的Pk_t小于0？？？？？？？？？
            print('GL_t:', GL_t, 'PQ:', PQ, 'Pk_t:', Pk_t)
            if PQ > 8:
                print('early stop!!!')
                break
        '''

    '''保留最后一层参数'''
    # paras = model.named_parameters()
    # for name, param in paras:
    #     print(name, param)
    #     # np.savetxt(result_path+dataset_name+'sparse_'+name+'.txt', param.detach().cpu().numpy(), fmt='%0.8f', delimiter=',')
    '''保留最后一层参数'''
    print('best_epoch:', best_epoch)
    print('best_test_acc:', best_test_acc)


    '''
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, label='train_loss', color='red')
    plt.legend()
    plt.savefig(result_path + 'picture/' + dataset_name + '_train_loss.png')

    plt.figure()
    plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label='val_loss', color='red')
    plt.legend()
    plt.savefig(result_path+'picture/'+dataset_name+'_val_loss.png')

    # plt.figure()
    # plt.plot(range(1, len(train_acc_history)+1), train_acc_history, label='Train acc', color='green')
    # plt.plot(range(1, len(val_acc_history)+1), val_acc_history, label='Val acc', color='blue')
    # plt.plot(range(1, len(test_acc_history)+1), test_acc_history, label='Test acc', color='yellow')
    # plt.legend()
    # plt.savefig(result_path+'picture/'+dataset_name+'_acc.png')
    # plt.figure()
    # plt.plot(range(1, len(loss2_1_history)+1), loss2_1_history, label='loss2_1_history', color='red')
    # plt.plot(range(1, len(loss2_2_history)+1), loss2_2_history, label='loss2_2_history', color='black')
    # plt.plot(range(1, len(loss_BNM_history)+1), loss_BNM_history, label='loss_BNM_history', color='blue')
    # plt.plot(range(1, len(loss_reg_l1_history)+1), loss_reg_l1_history, label='loss_reg_gcn1_l1_history', color='green')
    # plt.plot(range(1, len(loss_reg_l1_history)+1), loss_reg_l1_history, label='loss_reg_l1_history', color='green')
    # plt.plot(range(1, len(loss_reg_l2_history)+1), loss_reg_l2_history, label='loss_reg_l2_history', color='blue')
    # plt.legend()
    # plt.savefig(result_path+'picture/'+dataset_name+'_all_loss.png')
    # plt.show()
    # '''
    return best_test_acc




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda:1'
    # device = 'cpu'
    print('-------使用{}-------'.format(device))
    dataset_name = 'cora'
    # dataset_name = 'citeseer'
    # dataset_name = 'pubmed'
    precent = 1
    print('-------加载{}数据--------'.format(dataset_name))
    start = time.time()

    '''基于louvain算法'''
    adj, features, y_labels, train_mask, val_mask, test_mask, y_class = loadData(dataset_str=dataset_name, precent=precent, method=None)

    end = time.time()
    print('------数据加载完毕--------')
    load_time = end - start
    print('数据加载和社团检测时间:', load_time)
    n, d = features.shape
    input_dim = d
    dropout = 0.4
    weight_decay = 0.0005
    learning_rate = 0.03
    weight_loss2_1 = 1
    weight_loss2_2 = 0.5
    alpha = 0.2
    nheads = 2
    nout_heads = 2
    hidden_dim = 8
    output_dim2_1 = 8
    output_dim2_2 = y_labels.shape[1]
    cluster_num = y_labels.shape[1]
    print('dataset_name:', dataset_name)
    print('precent:', precent)
    print('input_dim:', input_dim)
    print('dropout:', dropout)
    print('weight_decay:', weight_decay)
    print('learning_rate:', learning_rate)
    print('weight_loss2_1:', weight_loss2_1)
    print('weight_loss2_2:', weight_loss2_2)
    print('alpha:', alpha)
    print('nheads:', nheads)
    print('nout_heads:', nout_heads)
    print('hidden_dim:', hidden_dim)
    print('output_dim2_1:', output_dim2_1)
    print('output_dim2_2:', output_dim2_2)
    print('cluster_num:', cluster_num)

    def main(num, dataset_name, adj, features, y_labels, train_mask, val_mask, test_mask, input_dim,
             dropout, weight_decay, learning_rate, weight_loss2_1, weight_loss2_2,
             alpha, nheads, nout_heads, hidden_dim, output_dim2_1, output_dim2_2, cluster_num, precent):

        # n, d = features.shape
        output_dim2_2 = output_dim2_2
        output_dim2_1 = output_dim2_1
        print('-------第{}次---------'.format(str(num)))
        '''
        有如下参数
        '''
        dropout = dropout
        weight_decay = weight_decay
        # weight_decay = 5e-4  #pubmed
        # weight_decay = 0
        learning_rate = learning_rate
        # learning_rate = 0.008
        '''共享层参数的权重'''
        weight_reg_l1 = 10e-4
        '''共享层参数的权重'''
        weight_reg_l2 = 10e-4
        epochs = 1000
        input_dim = d
        hidden_dim = hidden_dim
        # weight_loss2_1 = 0.5
        weight_loss2_1 = weight_loss2_1  #citeseer，pubmed
        # weight_loss2_1 = 0.4
        weight_loss2_2 = weight_loss2_2
        # print('input_dim:', input_dim)
        # output_dim2_1 = len(set(y_class))
        # print('output_dim2_1:', output_dim2_1)
        # output_dim2_2 = y_labels.shape[1]
        # print('output_dim2_2:', output_dim2_2)
        alpha = alpha
        nheads = nheads
        # adj_norm = normalization(adj.A).to(device, dtype=torch.float32)
        adj = torch.tensor(adj.A).to(device, dtype=torch.float32)
        adj_norm = adj + torch.eye(adj.size(0)).to(device, dtype=torch.float32)
        feature_X = torch.tensor(features.A).to(device, dtype=torch.float32)
        # adj = torch.tensor(adj.A).to(device, dtype=torch.float32)
        # feature_X = torch.cat((feature_X, adj), 1)
        # input_dim = feature_X.shape[1]

        '''进行社团划分得到的标签'''
        train_y2_1 = torch.tensor(y_class).to(device, torch.long) #不是one-hot编码
        '''进行训练的节点'''
        train_x_part = torch.tensor(train_mask).to(device)
        '''进行测试的节点'''
        test_x_part = torch.tensor(test_mask).to(device)
        '''进行验证的节点'''
        val_x_part = torch.tensor(val_mask).to(device)
        '''节点的真实标签'''
        train_y2_2 = torch.tensor(y_labels).to(device)
        dataset_name = dataset_name+'_'+str(precent)+'_'+'717'+'第'+str(num)+'次'
        # result_path = './asynlpa_result/'
        # result_path = './lpa_result/'
        # result_path = './gm_result/'
        # result_path = './gm_result_gat/'
        # result_path = './louvain_result/'
        # result_path = './louvain_result_gat/'
        # result_path = './spectral_result/'
        result_path = './test_result/'
        # result_path = './dssm_gcn_result/'
        # result_path = './dssm_gat_result/'
        # result_path = './不加正则/'
        # result_path = './只加共享层正则/'
        # result_path = './考察矩阵范数/'
        # result_path = './加入BNM损失/'
        '''下面两个变量没啥用'''
        # best_rep_name_node = result_path+'rep/'+dataset_name+'_node_best.npy'
        # best_rep_name_comm = result_path+'rep/'+dataset_name+'_comm_best.npy'

        '''
        有如上参数
        '''
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print('------使用{}------'.format(device))
        mutiltask_model = MutiltaskSpGatNet(input_dim=input_dim, hidden_dim=hidden_dim,
                                          output_dim2_1=output_dim2_1, output_dim2_2=output_dim2_2,
                                          dropout=dropout, alpha=alpha, nheads=nheads, nout_heads=nout_heads).to(device)
        # mutiltask_model = torch.nn.DataParallel(mutiltask_model, device_ids=[0,1])
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(mutiltask_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimizer = optim.SGD(mutiltask_model.parameters(), lr=learning_rate)
        best_test_acc = train(model=mutiltask_model, optimizer=optimizer, epochs=epochs, adj_norm=adj_norm, criterion=criterion, weight_loss2_1=weight_loss2_1, weight_loss2_2=weight_loss2_2,
          feature_X=feature_X, cluster_num=cluster_num, train_y2_2=train_y2_2, train_x_part=train_x_part, test_x_part=test_x_part, val_x_part=val_x_part,
          result_path=result_path, dataset_name=dataset_name)

        return best_test_acc


    result_list = []
    time_list = []
    num = 0
    for i in range(50):
        start = time.time()

        best_test_acc = main(num, dataset_name, adj, features, y_labels, train_mask, val_mask, test_mask, input_dim,
             dropout, weight_decay, learning_rate, weight_loss2_1, weight_loss2_2,
             alpha, nheads, nout_heads, hidden_dim, output_dim2_1, output_dim2_2, cluster_num, precent)

        end = time.time()
        run_time = end - start
        print('第{}次运行时间:'.format(str(i)), run_time)
        num = num + 1
        time_list.append(run_time)
        result_list.append(best_test_acc)

    print('average acc:', np.average(result_list))
    print('std:', np.std(result_list))
    print('数据加载和社团检测时间:', load_time)
    print('average run time:', sum(time_list) / len(time_list))


    exit()




    input_dim = d
    output_dim2_2 = y_labels.shape[1]


    dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    random.shuffle(dropout_list)
    # dropout_list = [0.5]
    weight_decay_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    random.shuffle(weight_decay_list)
    # weight_decay_list = [0.005, 0.004, 0.003, 0.002]
    learning_rate_list = [0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
    random.shuffle(learning_rate_list)
    # learning_rate_list = [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
    '''无监督节点分类'''
    weight_loss2_1_list = [1]
    '''半监督节点分类任务'''
    weight_loss2_2_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    random.shuffle(weight_loss2_2_list)
    alpha_list = [0.2]
    nheads_list = [2]
    nout_heads_list = [2]
    hidden_dim_list = [8]
    output_dim2_1_list = [8]

    cluster_num_list = [y_labels.shape[1]]
    # cora=7, citeseer=6, pubmed=3

    num = 0

    for dropout in dropout_list:
        for weight_decay in weight_decay_list:
            for learning_rate in learning_rate_list:
                for weight_loss2_1 in weight_loss2_1_list:
                    for weight_loss2_2 in weight_loss2_2_list:
                        for alpha in alpha_list:
                            for nheads in nheads_list:
                                for nout_heads in nout_heads_list:
                                    for hidden_dim in hidden_dim_list:
                                        for output_dim2_1 in output_dim2_1_list:
                                            for cluster_num in cluster_num_list:
                                                num = num + 1
                                                main(num, dataset_name, adj, features, y_labels, train_mask, val_mask,
                                                     test_mask, input_dim, dropout, weight_decay, learning_rate, weight_loss2_1,
                                                     weight_loss2_2, alpha, nheads, nout_heads, hidden_dim, output_dim2_1,
                                                     output_dim2_2, cluster_num)
