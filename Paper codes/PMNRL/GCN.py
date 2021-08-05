#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/27
# @Author  : Wang Biao
# @Site    :
# @File    : GCN.py
# @Software: PyCharm

'''
定义GCN层
X‘ = f(LXW), f为激活函数，L为D^(-1/2)*A'*D^(-1/2)
A' = A+I
'''

import torch
import torch.nn as nn
import torch.nn.init as init

class GCNlayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=None):
        '''
        :param input_dim:
        :param output_dim:
        :param use_bias:
        '''
        super(GCNlayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        # torch.manual_seed(2020) 固定参数
        self.weight = nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        if self.use_bias:
            self.use_bias = nn.Parameter(torch.randn(self.output_dim, 1))
        # else:
        #     self.reset_parameter('bias', None)

    # def reset_parameter(self):
    #     init.kaiming_uniform(self.weight)
    #     if self.use_bias:
    #         init.zeros_(self.use_bias)

    def forward(self, adj_norm, input_feature):
        '''

        :param adj_norm:
        :param input_feature:
        :return:
        '''
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adj_norm, support)
        if self.use_bias:
            output += self.use_bias
        return output

'''
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGCNlayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=None):
        super(SpGCNlayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        # torch.manual_seed(2020) 固定参数
        self.weight = nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        self.special_spmm = SpecialSpmm()
        if self.use_bias:
            self.use_bias = nn.Parameter(torch.randn(self.output_dim, 1))

    def forward(self, adj_norm, input_feature):
        device = 'cuda' if input.is_cuda else 'cpu'

        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adj_norm, support)


        h = torch.where(torch.isnan(output), torch.full_like(output, 1e-6), output)
        assert not torch.isnan(h).any()

        edge = adj_norm.nonzero().t()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E




        # 使有nan的地方等于1e-6
        edge_e = torch.where(torch.isnan(edge_e), torch.full_like(edge_e, 1e-6), edge_e)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=device))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)

        # h: N x out
        assert not torch.isnan(h).any()

        if self.use_bias:
            output += self.use_bias
        return output

'''





