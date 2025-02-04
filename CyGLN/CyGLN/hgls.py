#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/14 4:28
# @Author : ZM7
# @File : hgls
# @Software: PyCharm
import random
import torch
import torch.nn as nn
from rrgcn import RecurrentRGCN
from hrgnn import HRGNN
from rgcn.utils import build_sub_graph
import torch.nn.functional as F
from Time_decoder import TimeConvTransE ,TimeConvTransR
from rrgcn import RGCNCell
from hrgnn import GNN
import numpy as np

class HGLS(nn.Module):
    def __init__(self, graph, num_nodes, num_rels,num_times,time_interval, h_dim, task, relation_prediction, short=True, long=True, fuse='con',
                 r_fuse='re', short_con=None, long_con=None,static_graph=None,):
        super(HGLS, self).__init__()
        self.g = graph
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.h_dim = h_dim
        self.sequence_len = short_con['sequence_len']
        self.task = task
        self.relation_prediction = relation_prediction
        self.short = short
        self.long = long
        self.fuse = fuse
        self.r_fuse = r_fuse
        self.en_embedding = nn.Embedding(self.num_nodes, self.h_dim)
        self.rel_embedding = nn.Embedding(self.num_rels * 2 + 1, self.h_dim)
        torch.nn.init.normal_(self.en_embedding.weight)
        torch.nn.init.xavier_normal_(self.rel_embedding.weight)
        self.gnn = long_con['encoder']
        self.g1=static_graph#静态参数
        #------------------全局历史和时间向量
        self.num_times=num_times
        self.time_interval=time_interval
        self.sin=torch.sin
        self.cos=torch.cos
        self.tan=torch.tan
        self.Linear_0=nn.Linear(num_times,1)
        self.Linear_1=nn.Linear(num_times,self.h_dim - 1)
        self.tanh=nn.Tanh()
        self.use_cuda =None

        self.history_rate=0.3
        #------------------全局历史和时间向量
        # GNN 初始化

        if self.gnn == 'regcn':
            self.rgcn = RGCNCell(num_nodes,
                                 h_dim,
                                 h_dim,
                                 num_rels * 2,
                                 short_con['num_bases'],
                                 short_con['num_basis'],
                                 long_con['a_layer_num'],
                                 short_con['dropout'],
                                 short_con['self_loop'],
                                 short_con['skip_connect'],
                                 short_con['encoder'],
                                 short_con['opn'])
        elif self.gnn == 'rgat':
            self.rgcn = GNN(self.h_dim, self.h_dim, layer_num=long_con['a_layer_num'], gnn=self.gnn, attn_drop=0.0, feat_drop=0.2)
        if self.short:
            #self.model_r = RecurrentRGCN(num_ents=num_nodes, num_rels=num_rels, gnn=self.gnn, **short_con)原始的
            self.model_r = RecurrentRGCN(num_ents=num_nodes, num_rels=num_rels,num_times=num_times,time_interval=time_interval, gnn=self.gnn,stat_graph=self.g1, **short_con)#这个是加入静态图的
            # self.model_r = RecurrentRGCN(num_ents=num_nodes, num_rels=num_rels,num_times=num_times,time_interval=time_interval,\
            #                              one_hot_rel_seq=self.one_hot_rel_seq,one_hot_tail_seq=self.one_hot_tail_seq,gnn=self.gnn,stat_graph=self.g1, **short_con)#这个是加入静态图的以及全局历史和时间向量的

            self.model_r.rgcn = self.rgcn
            self.model_r.dynamic_emb = self.en_embedding.weight
            self.model_r.emb_rel = self.rel_embedding.weight


        if self.long:
            self.model_t = HRGNN(graph=graph, num_nodes=num_nodes, num_rels=num_rels, **long_con)
            self.model_t.aggregator = self.rgcn
            self.model_t.en_embedding = self.en_embedding
            self.model_t.rel_embedding = self.rel_embedding
        if self.short and self.long:
            if self.fuse == 'con':
                self.linear_fuse = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            elif self.fuse == 'att':
                self.linear_l = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s = nn.Linear(self.h_dim, self.h_dim, bias=True)
            elif self.fuse == 'att1':
                self.linear_l = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.fuse_f = nn.Linear(self.h_dim, 1, bias=True)
            elif self.fuse == 'gate':
                self.gate = GatingMechanism(self.num_nodes, self.h_dim)
            else:
                print('no fuse function')
            if self.r_fuse == 'con':
                self.linear_fuse_r = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            elif self.r_fuse == 'att1':
                self.linear_l_r = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s_r = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.fuse_f_r = nn.Linear(self.h_dim, 1, bias=True)
            elif self.r_fuse == 'gate':
                self.gate_r = GatingMechanism(self.num_rels *2 , self.h_dim)

            else:
                print('no fuse_r function')
        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()


        self.decoder_ob = TimeConvTransE(num_nodes, h_dim, short_con['input_dropout'], short_con['hidden_dropout'], short_con['feat_dropout'])
        self.rdecoder = TimeConvTransR(num_rels, h_dim, short_con['input_dropout'], short_con['hidden_dropout'], short_con['feat_dropout'])

        # -----------------------------------这也是自己加入的全局历史和时间向量
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1 ,h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1 ,h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1 ,h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1 ,h_dim))
        # -----------------------------------这也是自己加入的全局历史和时间向量


#-----------------------------------------------------------这是原来的为修改的代码。只是在有些部分加入了静态图，以及在解码器中使用了时间向量编码器
    def forward(self, total_list, data_list,  node_id_new=None, time_gap=None,device=None, mode='test',static_graph=None):
        # RE-GCN的更新
        t = data_list['t'][0].to(device)
        all_triples = data_list['triple'].to(device)
        if self.short:
            if t - self.sequence_len < 0:
                input_list = total_list[0:t]
            else:
                input_list = total_list[t-self.sequence_len: t]
            history_glist = [build_sub_graph(self.num_nodes, self.num_rels, snap, device) for snap in input_list]
            evolve_embs, static_emb, r_emb, _, _ = self.model_r(history_glist, static_graph, device=device)#这是修改的代码 加入静态图
            #evolve_embs, static_emb, r_emb, _, _ = self.model_r(history_glist, device=device)  这是源代码
            pre_emb = F.normalize(evolve_embs[-1])
        if self.long:
            new_embedding = F.normalize(self.model_t(data_list, node_id_new, time_gap, device, mode))
            new_r_embedding = self.model_t.rel_embedding.weight[0:self.num_rels*2]

        if self.long and self.short:
            # entity embedding fusion
            if self.fuse == 'con':
                pre_emb = self.linear_fuse(torch.cat((pre_emb, new_embedding), 1))
            elif self.fuse == 'att':
                pre_emb, e_cof = self.fuse_attention(pre_emb, new_embedding, self.en_embedding.weight)
            elif self.fuse == 'att1':
                pre_emb, e_cof = self.fuse_attention1(pre_emb, new_embedding)
            elif self.fuse == 'gate':
                pre_emb, e_cof = self.gate(pre_emb, new_embedding)
            # relation embedding fusion
            if self.r_fuse == 'short':
                r_emb = r_emb
            elif self.r_fuse == 'long':
                r_emb = new_r_embedding
            elif self.r_fuse == 'con':
                r_emb = self.linear_fuse_r(torch.cat((r_emb, new_r_embedding), 1))
            elif self.r_fuse == 'att1':
                r_emb, r_cof = self.fuse_attention_r(r_emb, new_r_embedding)
            elif self.r_fuse == 'gate':
                r_emb, r_cof = self.gate_r(r_emb, new_r_embedding)
        elif self.long and not self.short:
            pre_emb = new_embedding
            r_emb = new_r_embedding

        # 构造loss
        loss_ent = torch.zeros(1).to(device)
        loss_rel = torch.zeros(1).to(device)

        time_embs=self.get_init_time(all_triples) #自己加入的（时间周期性和全局历史）
        #scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples, mode).view(-1, self.num_nodes)
        scores_ob = self.decoder_ob.forward(pre_emb, r_emb, time_embs,all_triples, mode).view(-1, self.num_nodes)

        # loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
        loss_ent += self.loss_e(scores_ob, all_triples[:, 2].long())
        if self.relation_prediction:
            # score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode).view(-1, self.num_rels *2)
            score_rel = self.rdecoder.forward(pre_emb, r_emb, time_embs,all_triples, mode).view(-1, self.num_rels *2)
            # loss_rel += self.loss_r(score_rel, all_triples[:, 1])
            loss_rel += self.loss_r(score_rel, all_triples[:, 1].long())
        loss = self.task * loss_ent + (1 - self.task) * loss_rel
        if mode == 'test':
            return scores_ob, 0, loss
        else:
            return loss_ent, 0, loss

# -----------------------------------------------------------这是原来的为修改的代码。只是在有些部分加入了静态图，
    def fuse_attention(self, s_embedding, l_embedding, o_embedding):
        w1 = (o_embedding * torch.tanh(self.linear_s(s_embedding))).sum(1)
        w2 = (o_embedding * torch.tanh(self.linear_l(l_embedding))).sum(1)
        aff = F.softmax(torch.cat((w1.unsqueeze(1),w2.unsqueeze(1)),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def fuse_attention1(self, s_embedding, l_embedding):
        w1 = self.fuse_f(torch.tanh(self.linear_s(s_embedding)))
        w2 = self.fuse_f(torch.tanh(self.linear_l(l_embedding)))
        aff = F.softmax(torch.cat((w1,w2),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def fuse_attention_r(self, s_embedding, l_embedding):
        w1 = self.fuse_f_r(torch.tanh(self.linear_s_r(s_embedding)))
        w2 = self.fuse_f_r(torch.tanh(self.linear_l_r(l_embedding)))
        aff = F.softmax(torch.cat((w1,w2),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff
#----------------------这是自己加入的时间向量编码器--------------
    def get_init_time(self,quadrupleList):
        T_idx=quadrupleList[:,3]//self.time_interval
        T_idx=T_idx.unsqueeze(1).float()
        t1=self.weight_t1 * T_idx +self.bias_t1
        t2=self.sin(self.weight_t2 * T_idx +self.bias_t2)
        return t1,t2

#----------------------这是自己加入的时间向量编码器---------------


# class GatingMechanism(nn.Module):
#     def __init__(self, entity_num, hidden_dim):
#         super(GatingMechanism, self).__init__()
#         self.gate_theta = nn.Parameter(torch.empty(entity_num, hidden_dim))
#         # 添加一个共享层来生成调整参数的权重
#         self.conditioning_layer = nn.Linear(hidden_dim, hidden_dim)
#         nn.init.xavier_uniform_(self.gate_theta)
#         nn.init.xavier_uniform_(self.conditioning_layer.weight)
#
#     def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
#         # 通过共享层获得条件权重
#         condition = torch.sigmoid(self.conditioning_layer(X))
#         # 使用条件权重来调整门控参数
#         gate = torch.sigmoid(self.gate_theta * condition)
#
#         output = torch.mul(gate, X) + torch.mul(1 - gate, Y)
#         return output, gate

#------------------------这是源代码中的门控集成
class GatingMechanism(nn.Module):
    def __init__(self, entity_num, hidden_dim):
        super(GatingMechanism, self).__init__()
        # gating 的参数
        self.gate_theta = nn.Parameter(torch.empty(entity_num, hidden_dim))
        nn.init.xavier_uniform_(self.gate_theta)
        # self.dropout = nn.Dropout(self.params.dropout)

    def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        gate = torch.sigmoid(self.gate_theta)
        output = torch.mul(gate, X) + torch.mul(-gate + 1, Y)
        return output, gate

#------------------这是修改后的门控集成-----------------------

# class GatingMechanism(nn.Module):
#     def __init__(self, entity_num, hidden_dim, dropout=0.2):
#         super(GatingMechanism, self).__init__()
#         self.gate_theta = nn.Parameter(torch.empty(2, entity_num, hidden_dim))
#         nn.init.xavier_uniform_(self.gate_theta)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
#         # 应用 dropout
#         X = self.dropout(X)
#         Y = self.dropout(Y)
#
#         # 使用 softmax 函数
#         gate = F.softmax(self.gate_theta, dim=0)
#         output = gate[0] * X + gate[1] * Y
#
#         return output, gate

# class GatingMechanism(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GatingMechanism, self).__init__()
#         # GRU的参数，这里定义了一个GRU层
#         self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
#
#     def forward(self, X, Y):
#         '''
#         :param X: 序列输入, 形状为(batch, seq_len, input_dim), LSTM的输出
#         :param Y: 可能是初始隐藏状态或者另一个形状为(batch, seq_len, input_dim)的序列
#         :return: GRU的输出结果和最新的隐藏状态
#         '''
#         # GRU期望的输入形状是(batch, seq_len, input_dim)，输出形状(batch, seq_len, hidden_dim)
#         # 我们使用Y作为初始隐藏状态，这是一个假设；具体情况可能有所不同
#         # Y的形状应该是(1, batch, hidden_dim)，我们可能需要调整维度或者处理方式
#         # 为简化，这里假设Y已经是正确的形状
#         output, hidden = self.gru(X, Y)
#
#         # 输出结果可以是最后一个时间步的输出，也可以是全部时间步的输出
#         # 这里我们返回最后一个时间步的输出
#         return output[:, -1, :], hidden

#--------------------------------------------------------
# class GatingMechanism(nn.Module):
#     def __init__(self, entity_num, hidden_dim):
#         super(GatingMechanism, self).__init__()
#         self.gate_theta = nn.Parameter(torch.empty(entity_num, hidden_dim))
#         self.weights_x = nn.Linear(hidden_dim, 1)
#         self.weights_y = nn.Linear(hidden_dim, 1)
#         nn.init.xavier_uniform_(self.gate_theta)
#         nn.init.xavier_uniform_(self.weights_x.weight)
#         nn.init.xavier_uniform_(self.weights_y.weight)
#
#     def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
#         gate = torch.sigmoid(self.gate_theta)
#         weights_x = torch.sigmoid(self.weights_x(X))
#         weights_y = torch.sigmoid(self.weights_y(Y))
#         output = weights_x * X + weights_y * Y
#         return output, gate

# class GatingMechanism(nn.Module):
#     def __init__(self, entity_num, hidden_dim):
#         super(GatingMechanism, self).__init__()
#         # gating 的参数
#         self.gate_theta = nn.Parameter(torch.empty(entity_num, hidden_dim))
#         nn.init.xavier_uniform_(self.gate_theta)
#         # self.dropout = nn.Dropout(self.params.dropout)
#
#     def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
#         '''
#         :param X:   LSTM 的输出tensor   |E| * H
#         :param Y:   Entity 的索引 id    |E|,
#         :return:    Gating后的结果      |E| * H
#         '''
#         gate = torch.sigmoid(self.gate_theta)
#         output = gate[0] * X + gate[1] * Y
#         return output, gate
