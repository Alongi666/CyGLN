import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial

class RGCNLayer(nn.Module):
    # 参数说明：
    # in_feat 输入维度
    # out_feat 输出维度
    # num_rels 边类型数量
    # num_bases W_r分解的数量，对应原文公式3的B（求和符号的上界）
    # bias 偏置
    # activation 激活函数
    # is_input_layer 是否是输入层（第一层）
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        # 矩阵分解的参数校验条件：不能小于0，不能比现有维度大（复杂度会变高，参数反而增加）
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        # 这里是根据公式3把W_r算出来，用V_b（weight）表示，共有num_bases个V_b累加得到
        # 得到的结果是Tensor，因此用 nn.Parameter将一个不可训练的类型Tensor
        # 转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            # 这里的w_comp是公式3里面的a_{rb}
            # 一个边类型对应一个W_r（那么就一共有num_rels种W_r），每个W_r分解为num_bases个组合
            # 因此w_comp这里的维度就是num_rels×num_bases
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        # 偏置要进行加法运算其维度要和输出维度大小一样
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        # 这里用的是xavier初始化，可以查下为什么
        # 因为我们用的是线性激活函数，网上有说用ReLU和Leaky ReLU
        # 可以考虑用别的初始化方法：He init 。
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:#分解就走公式3
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:#不分解就直接用weight算
            weight = self.weight

        if self.is_input_layer:#如果是第一层
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                # 对于第一层，输入可以直接用独热编码进行aggregate
                # 信息的汇聚就可以直接写成矩阵相乘的形式
                embed = weight.view(-1, self.out_feat)# embed维度整成out_feat维度一样
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
            	#根据边类型'rel_type'获取对应的w
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()#消息汇聚，就是w乘以src['h']（输入节点特征）
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:#有偏置加偏置
                h = h + self.bias
            if self.activation:#经过激活函数
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        # 先初始化参数
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        # 具体看下面函数
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        #torch.arange(start=0, end=5)的结果并不包含end，start默认是0
        features = torch.arange(self.num_nodes)
        return features

    # 输入num_nodes的独热编号
    # 输出h_dim
    # 激活函数是relu
    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    # 输入h_dim
    # 输出h_dim
    # 激活函数是relu
    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    # 输入h_dim
    # 输出out_dim
    # 激活函数是softmax后归一化
    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')
