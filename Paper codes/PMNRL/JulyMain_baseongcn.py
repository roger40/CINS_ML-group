#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/27
# @Author  : Wang Biao
# @Site    :
# @File    : Main.py
# @Software: PyCharm

'''
搭建基于mutiltaske的gcn模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from GCN import GCNlayer
from GAT import GraphAttentionLayer
import scipy.sparse as sp
import numpy as np
import matplotlib as mpl
# mpl.use('Agg') #服务器上保存图片
import matplotlib.pyplot as plt
from JulyLoadData import loadData
import time
# import torchsnooper

class MutiltaskGcnNet(nn.Module):
    '''
    定义所使用的基于多任务的图卷积模型, gcn2_1和gcn2_2属于并行关系
    '''
    def __init__(self, input_dim, hidden_dim, output_dim2_1, output_dim2_2, dropout):
        super(MutiltaskGcnNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim2_1 = output_dim2_1
        self.output_dim2_2 = output_dim2_2
        self.dropout = dropout
        self.gcn1 = GCNlayer(self.input_dim, self.hidden_dim)
        self.gcn2_1 = GCNlayer(self.hidden_dim, self.output_dim2_1)
        self.gcn2_2 = GCNlayer(self.hidden_dim, self.output_dim2_2)
    '''
    logits2_1:这个表示用于自监督，自行进行社团划分获得的
    logits2_2:这个表示用于半监督学习
    '''
    def forward(self, adj_norm, feature):
        h = F.relu(self.gcn1(adj_norm, feature))
        # h = F.leaky_relu(self.gcn1(adj_norm, feature), negative_slope=0.01)
        logits_share = h
        h = F.dropout(h, p=self.dropout)
        logits2_1 = self.gcn2_1(adj_norm, h)
        logits2_2 = self.gcn2_2(adj_norm, h)
        return logits2_1, logits2_2, logits_share


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
        logits2_1, logits2_2, logits_share = model(adj_norm, feature_X)
        test_x_logits = logits2_2[test_x_part]
        loss = criterion(logits2_2[test_x_part], torch.argmax(train_y[test_x_part], dim=1))
        # 这个错了，没有进行softmax，好像并不影响
        predict_y = test_x_logits.max(1)[1]
        accuracy = torch.eq(predict_y, torch.argmax(train_y[test_x_part], dim=1)).float().mean()
    return accuracy, loss

# @torchsnooper.snoop()
def train(model, optimizer, epochs, adj_norm, criterion, weight_loss2_1, weight_loss2_2,
          weight_reg_l1, weight_reg_l2,
          feature_X, train_y2_1, train_y2_2, train_x_part, test_x_part, val_x_part,
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
        logits2_1, logits2_2, logits_share = model(adj_norm, feature_X)
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

        loss2_1 = criterion(logits2_1, train_y2_1)

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

        print('epoch:', epoch, 'Train Loss:', loss_total.item(),
              'Train acc:', train_acc.item(), 'Test acc:', test_acc.item(), 'Val acc:', val_acc.item())

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
            if val_acc_history[-1] >= best_val_acc and val_loss_history[-1] <= best_val_loss:
            # if val_acc_history[-1] >= best_val_acc and val_loss_history[-1] <= best_val_loss and loss_history[-1] <= best_loss:
                best_epoch = epoch
                best_test_acc = test_acc_history[-1]
                best_val_loss = val_loss_history[-1]
                best_val_acc = val_acc_history[-1]
                best_loss = loss_history[-1]

                # 保存模型参数及优化器参数
                saved_dict = {
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict()
                }
                # torch.save(saved_dict, result_path+'model/'+dataset_name+'.pt')
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
        # np.savetxt(result_path+dataset_name+'sparse_'+name+'.txt', param.detach().cpu().numpy(), fmt='%0.8f', delimiter=',')
    '''保留最后一层参数'''
    print('best_epoch:', best_epoch)
    print('best_test_acc:', best_test_acc)


    '''
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, label='train_loss', color='red')
    plt.legend()
    plt.figure()
    plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label='val_loss', color='red')
    plt.legend()
    # plt.savefig(result_path+'picture/'+dataset_name+'_train_loss.png')
    plt.figure()
    plt.plot(range(1, len(train_acc_history)+1), train_acc_history, label='Train acc', color='green')
    plt.plot(range(1, len(val_acc_history)+1), val_acc_history, label='Val acc', color='blue')
    plt.plot(range(1, len(test_acc_history)+1), test_acc_history, label='Test acc', color='yellow')
    plt.legend()
    # plt.savefig(result_path+'picture/'+dataset_name+'_acc.png')
    plt.figure()
    plt.plot(range(1, len(loss2_1_history)+1), loss2_1_history, label='loss2_1_history', color='red')
    plt.plot(range(1, len(loss2_2_history)+1), loss2_2_history, label='loss2_2_history', color='black')
    # plt.plot(range(1, len(loss_BNM_history)+1), loss_BNM_history, label='loss_BNM_history', color='blue')
    # plt.plot(range(1, len(loss_reg_l1_history)+1), loss_reg_l1_history, label='loss_reg_gcn1_l1_history', color='green')
    # plt.plot(range(1, len(loss_reg_l1_history)+1), loss_reg_l1_history, label='loss_reg_l1_history', color='green')
    # plt.plot(range(1, len(loss_reg_l2_history)+1), loss_reg_l2_history, label='loss_reg_l2_history', color='blue')
    plt.legend()
    # plt.savefig(result_path+'picture/'+dataset_name+'_all_loss.png')
    plt.show()
    '''

    return best_test_acc


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda:0'
    device = 'cuda:1'
    # device = 'cpu'
    # 梯度下降

    print('-------使用{}-------'.format(device))
    dataset_name = 'cora'
    # dataset_name = 'citeseer'
    # dataset_name = 'pubmed'
    precent = 1
    # CNM, Louvain, ALPA, SLPA
    method = 'CNM'
    print('-------加载{}数据--------'.format(dataset_name))
    start = time.time()
    '''基于模块度最大化'''
    adj, features, y_labels, train_mask, val_mask, test_mask, y_class = loadData(dataset_str=dataset_name, precent=precent, method=method)
    # adj, features, y_labels, train_mask, val_mask, test_mask, y_class = loadData_old(dataset_str=dataset_name, precent=1)


    end = time.time()
    print('------数据加载完毕--------')
    load_time = end - start
    print('数据加载和社团检测时间:', load_time)

    dropout = 0.4
    weight_decay = 0.005
    learning_rate = 0.009
    weight_loss2_1 = 0.4
    hidden_num = 512
    print('dataset_name:', dataset_name)
    print('precent:', precent)
    print('dropout:', dropout)
    print('weight_decay:', weight_decay)
    print('learning_rate:', learning_rate)
    print('weight_loss2_1:', weight_loss2_1)
    print('hidden_num:', hidden_num)

    def main(num, dataset_name, adj, features, y_labels, train_mask, val_mask, test_mask, y_class,
             dropout, weight_decay, learning_rate, weight_loss2_1, hidden_num):
        print('-------第{}次---------'.format(str(num)))
        n, d=features.shape
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
        # epochs = 5
        input_dim = d
        hidden_dim = hidden_num
        # hidden_dim = 256

        # weight_loss2_1 = 0.5
        weight_loss2_1 = weight_loss2_1  #citeseer，pubmed
        # weight_loss2_1 = 0.4
        weight_loss2_2 = 1

        print('input_dim:', input_dim)
        output_dim2_1 = len(set(y_class))
        print('output_dim2_1:', output_dim2_1)
        output_dim2_2 = y_labels.shape[1]
        print('output_dim2_2:', output_dim2_2)


        adj_norm = normalization(adj.A).to(device, dtype=torch.float32)
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
        # result_path = './louvain_result/'
        # result_path = './spectral_result/'
        result_path = './test_result/'
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
        mutiltask_model = MutiltaskGcnNet(input_dim=input_dim, hidden_dim=hidden_dim,
                                          output_dim2_1=output_dim2_1, output_dim2_2=output_dim2_2, dropout=dropout).to(device)
        # mutiltask_model = torch.nn.DataParallel(mutiltask_model, device_ids=[0,1])
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(mutiltask_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimizer = optim.SGD(mutiltask_model.parameters(), lr=learning_rate)
        best_test_acc = train(model=mutiltask_model, optimizer=optimizer, epochs=epochs, adj_norm=adj_norm,
              criterion=criterion, weight_loss2_1=weight_loss2_1,
              weight_loss2_2=weight_loss2_2, weight_reg_l1=weight_reg_l1, weight_reg_l2=weight_reg_l2,
              feature_X=feature_X, train_y2_1=train_y2_1, train_y2_2=train_y2_2,
              train_x_part=train_x_part,test_x_part=test_x_part, val_x_part=val_x_part,
              result_path=result_path, dataset_name=dataset_name)

        return best_test_acc

    result_list = []
    time_list = []
    for i in range(50):
        # i = i+1
        start = time.time()
        best_test_acc = main(i, dataset_name, adj, features, y_labels, train_mask, val_mask, test_mask, y_class,
                             dropout, weight_decay, learning_rate, weight_loss2_1, hidden_num)
        end = time.time()
        run_time = end - start
        print('第{}次运行时间:'.format(str(i)), run_time)
        # exit()
        time_list.append(run_time)
        result_list.append(best_test_acc)
    print('average acc:', np.average(result_list))
    print('std:', np.std(result_list))
    print('数据加载和社团检测时间:', load_time)
    print('average run time:', sum(time_list) / len(time_list))
