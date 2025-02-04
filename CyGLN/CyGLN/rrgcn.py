import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from model import BaseRGCN
from decoder import ConvTransE, ConvTransR
from rgcn.layers import CompGCNLayer



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
            # return CompGCNLayer(self.h_dim,self.h_dim,self.num_rels)
        else:
            raise NotImplementedError

    # def forward(self, init_ent_emb, init_rel_emb, edges, mode='add'):
    #     if self.encoder_name == "uvrgcn":
    #         # 初始化节点和关系的嵌入
    #         h_v, h_r = init_ent_emb, init_rel_emb
    #
    #         # 遍历每一层
    #         for i, layer in enumerate(self.layers):
    #             # 适配CompGCNLayer的输入和调用
    #             h_v, h_r = layer(h_v, h_r, edges, mode)
    #
    #         # 返回更新后的实体嵌入和关系嵌入
    #         return h_v, h_r



    def forward(self, g, init_ent_emb, init_rel_emb, method=0):
        if self.encoder_name == "uvrgcn":
            if method == 0:
                node_id = g.ndata['id'].squeeze()
                g.ndata['h'] = init_ent_emb[node_id]
            elif method == 1:
                g.ndata['h'] = init_ent_emb
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')
#这是原来的之加入了静态图---------------------------------------------------------------------------------------------------
class RecurrentRGCN(nn.Module):
    def __init__(self, decoder, encoder, gnn, num_ents, num_rels, num_times,time_interval,num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=True, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, sequence='rnn', use_cuda=False, history_rate=0.3,analysis=False,stat_graph=None):
        super(RecurrentRGCN, self).__init__()
        self.static_graph=stat_graph
        self.decoder_name = decoder
        self.encoder_name = encoder
        self.gnn = gnn
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.sequence = sequence
        self.emb_rel = None
        # self.gpu=gpu
        self.num_times=num_times
        self.time_interval=time_interval
        self.history_rate=history_rate
        self.sin=torch.sin
        self.Linear_0=nn.Linear(num_times,1)
        self.Linear_1=nn.Linear(num_times,self.h_dim - 1)
        self.tanh=nn.Tanh()
        self.use_cuda =None
# ---------------------------------------------这也是自己加入的全局历史和时间向量

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)
        #这两个矩阵是处理静态信息的
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        # 这两个矩阵是处理静态信息的
        #self.dynamic_emb = None
        #其中97行到101行发生了改变

        # -----------------------------------这也是自己加入的全局历史和时间向量
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1 ,h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1 ,h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1 ,h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1 ,h_dim))
        # -----------------------------------这也是自己加入的全局历史和时间向量

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
       # self.rgcn = None

        # self.rgcn = RGCNCell(num_ents,
        #                      h_dim,
        #                      h_dim,
        #                      num_rels * 2,
        #                      num_bases,
        #                      num_basis,
        #                      num_hidden_layers,
        #                      dropout,
        #                      self_loop,
        #                      skip_connect,
        #                      encoder,
        #                      self.opn,
        #                      self.emb_rel,
        #                      use_cuda,
        #                      analysis)

#-----------------------------------这也是自己加入的全局历史和时间向量
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.global_weight=nn.Parameter(torch.Tensor(self.num_ents,1))
        nn.init.xavier_uniform_(self.global_weight, gain=nn.init.calculate_gain('relu'))
        self.global_bias=nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)
# -----------------------------------这也是自己加入的全局历史和时间向量
        if self.sequence == 'regcn':
            self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
            nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
            self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
            nn.init.zeros_(self.time_gate_bias)
        elif self.sequence == 'rnn':
            self.rnn_cell = nn.GRUCell(self.h_dim, self.h_dim)
        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
#-----------------------------------这也是自己加入的全局历史和时间向量
        # if decoder_name == "timeconvtranse":
        #     self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        #     self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        #     self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        #     self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        # else:
        #     raise NotImplementedError
# --------------------------这也是自己加入的全局历史和时间向量
    def forward(self, g_list, static_graph,device):
        gate_list = []
        degree_list = []

        if self.use_static:
            self.static_graph = self.static_graph.to(device)
            self.static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(self.static_graph, [])
            static_emb = self.static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(device)
            # temp_e = self.h[g.r_to_e]
            temp_e = self.h[g.r_to_e.long()]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().to(device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel[0:self.num_rels *2], x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel[0:self.num_rels *2])    # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel[0:self.num_rels *2], x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            if self.gnn == 'regcn':
                current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
                current_h=F.normalize(current_h) if self.layer_norm else self.h###################这是自己添加的部分提高了1%
                # current_h = F.normalize(current_h)
            elif self.gnn == 'rgat':
                g.edata['r_h'] = self.h_0[g.edata['etype']]
                #g.edata['r_h'] = self.emb_rel[g.edata['etype']]
                current_h = self.rgcn(g, self.h)
            if self.sequence == 'regcn':
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                time_weight = torch.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
                self.h = time_weight * current_h + (1-time_weight) * self.h
            elif self.sequence == 'rnn':
                self.h = self.rnn_cell(current_h, self.h)
                self.h=F.normalize(self.h) if self.layer_norm else self.h #########################这是自己添加的部分提高了1%
                # self.h = F.normalize(self.h)
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    # def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
    #     with torch.no_grad():
    #         inverse_test_triplets = test_triplets[:, [2, 1, 0]]
    #         inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
    #         all_triples = torch.cat((test_triplets, inverse_test_triplets))
    #
    #         evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
    #         embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
    #
    #         score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
    #         score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
    #         return all_triples, score, score_rel

#--------------------------------------------这是修改后的 加入全局历史和时间向量编码器
    # def predict(self, test_graph, num_rels, static_graph, test_triplets, entity_history_vocabulary,rel_history_vocabulary,use_cuda):
    #     self.use_cuda=use_cuda
    #     with torch.no_grad():
    #         inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
    #         inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
    #         all_triples = torch.cat((test_triplets, inverse_test_triplets))
    #
    #         evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
    #         embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
    #         time_embs=self.get_init_time(all_triples)
    #
    #         score_rel_r =self.rel_raw_mode(embedding,r_emb,time_embs,all_triples)
    #         score_rel_h=self.rel_history_mode(embedding,r_emb,time_embs,all_triples,rel_history_vocabulary)
    #         score_r =self.raw_mode(embedding,r_emb,time_embs,all_triples)
    #         score_h=self.history_mode(embedding,r_emb,time_embs,all_triples,entity_history_vocabulary)
    #
    #         score_rel=self.history_rate* score_rel_h+(1-self.history_rate)*score_rel_r
    #         score_rel=torch.log(score_rel)
    #         score=self.history_rate * score_h +(1-self.history_rate) *score_r
    #         score=torch.log(score)
    #
    #         return all_triples, score, score_rel
# --------------------------------------------这是修改后的 加入全局历史和时间向量编码器

    # def get_loss(self, glist, triples, static_graph, device):
    #     """
    #     :param glist:
    #     :param triplets:
    #     :param static_graph:
    #     :param use_cuda:
    #     :return:
    #     """
    #     loss_ent = torch.zeros(1).to(device)
    #     loss_rel = torch.zeros(1).to(device)
    #     loss_static = torch.zeros(1).to(device)
    #
    #     inverse_triples = triples[:, [2, 1, 0]]
    #     inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
    #     all_triples = torch.cat([triples, inverse_triples])
    #     all_triples = all_triples.to(device)
    #
    #     evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, device)
    #     pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
    #     if self.entity_prediction:
    #         scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
    #         loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
    #
    #     if self.relation_prediction:
    #         score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
    #         loss_rel += self.loss_r(score_rel, all_triples[:, 1])


# --------------------------------------------------------------这是修改后的，加入了时间向量和全局历史编码器
#     def get_loss(self, glist, triples, static_graph,entity_history_vocabulary,rel_history_vocabulary, use_cuda):
#         """
#                 :param glist:
#                 :param triplets:
#                 :param static_graph:
#                 :param use_cuda:
#                 :return:
#                 """
#         self.use_cuda = use_cuda
#         loss_ent = torch.zeros(1).cuda.to(self.gpu) if use_cuda else torch.zeros(1)
#         loss_rel = torch.zeros(1).cuda.to(self.gpu) if use_cuda else torch.zeros(1)
#         loss_static = torch.zeros(1).cuda.to(self.gpu) if use_cuda else torch.zeros(1)
#
#         inverse_triples = triples[:, [2, 1, 0 , 3]] ###这样是修改
#         inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
#         all_triples = torch.cat([triples, inverse_triples])
#         all_triples = all_triples.to(self.gpu)
#
#         evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
#         pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
#         time_embs=self.get_init_time(all_triples) #自己加入的（时间周期性和全局历史）
#
#         if self.entity_prediction:
#             score_r =self.raw_mode(pre_emb,r_emb,time_embs,all_triples)
#             score_h=self.history_mode(pre_emb,r_emb,time_embs,all_triples,entity_history_vocabulary)
#             score_en=self.history_rate *score_h+(1-self.history_rate) *score_r
#             scores_en=torch.log(score_en)
#             loss_ent += F.nll_loss(scores_en,all_triples[:,2])
#
#         if self.relation_prediction:
#             score_rel_r =self.rel_raw_mode(pre_emb,r_emb,time_embs,all_triples)
#             score_rel_h=self.rel_history_mode(pre_emb,r_emb,time_embs,all_triples,rel_history_vocabulary)
#             score_re=self.rel_history_rate *score_rel_h+(1-self.history_rate) *score_rel_r
#             scores_re=torch.log(score_re)
#             loss_rel += F.nll_loss(scores_re,all_triples[:,1])
# # --------------------------------------------------------------这是修改后的，加入了时间向量和全局历史编码器
#
#         if self.use_static:
#             if self.discount == 1:
#                 for time_step, evolve_emb in enumerate(evolve_embs):
#                     step = (self.angle * math.pi / 180) * (time_step + 1)
#                     if self.layer_norm:
#                         sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
#                     else:
#                         sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
#                         c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
#                         sim_matrix = sim_matrix / c
#                     mask = (math.cos(step) - sim_matrix) > 0
#                     loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
#             elif self.discount == 0:
#                 for time_step, evolve_emb in enumerate(evolve_embs):
#                     step = (self.angle * math.pi / 180)
#                     if self.layer_norm:
#                         sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
#                     else:
#                         sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
#                         c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
#                         sim_matrix = sim_matrix / c
#                     mask = (math.cos(step) - sim_matrix) > 0
#                     loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
#         return loss_ent, loss_rel, loss_static
#
#
#     #--------------------------------这里是加入全局历史编码器和时间周期向量
#     def get_init_time(self,quadrupleList):
#         T_idx=quadrupleList[:,3]//self.time_interval
#         T_idx=T_idx.unsqueeze(1).float()
#         t1=self.weight_t1 * T_idx +self.bias_t1
#         t2=self.sin(self.weight_t2 * T_idx +self.bias_t2)
#         return t1,t2
#
#     def raw_mode(self,pre_emb,r_emb,time_embs,all_triples):
#         scores_ob =self.decoder_ob1.forward(pre_emb,r_emb,time_embs,all_triples).view(-1,self.num_ents)
#         score=F.softmax(scores_ob,dim=1)
#         return score
#
#     def history_mode(self,pre_emb,r_emb,time_embs,all_triples,history_vocabulary):
#         if self.use_cuda:
#             global_index=torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
#             global_index=global_index.to('cuda')
#         else:
#             global_index=torch.Tensor(np.array(history_vocabulary.cpu() ,dtype=float))
#         score_global=self.decoder_ob2.forward(pre_emb,r_emb,time_embs,all_triples,partial_embeding=global_index)
#         score_h=score_global
#         score_h=F.softmax(score_h,dim=1)
#         return score_h
#
#     def rel_raw_mode(self,pre_emb,r_emb,time_embs,all_triples):
#         scores_re =self.rdecoder_re1.forward(pre_emb,r_emb,time_embs,all_triples).view(-1,2*self.num_rels)
#         score=F.softmax(scores_re,dim=1)
#         return score
#
#     def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
#         if self.use_cuda:
#             global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
#             global_index = global_index.to('cuda')
#         else:
#             global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
#         score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples,partial_embeding=global_index)
#         score_h = score_global
#         score_h = F.softmax(score_h, dim=1)
#         return score_h
#     # --------------------------------这里是加入全局历史编码器和时间周期向量

