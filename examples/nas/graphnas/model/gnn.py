import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNet(torch.nn.Module):

    def __init__(self, actions, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=True, residual=True,
                 state_num=5):
        '''
        :param actions:
        :param multi_label:
        '''
        super(GraphNet, self).__init__()
        # args

        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_label = num_label
        self.dropout = drop_out
        self.residual = residual
        # check structure of GNN
        self.layer_nums = self.evalate_actions(actions, state_num)

        # layer module
        self.build_model(actions, batch_normal, drop_out, num_feat, num_label, state_num)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        self.layers = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.prediction = None
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)

    def evalate_actions(self, actions, state_num):
        state_length = len(actions)
        if state_length % state_num != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        layer_nums = state_length // state_num
        if self.evaluate_structure(actions, layer_nums, state_num=state_num):
            pass
        else:
            raise RuntimeError("wrong structure")
        return layer_nums

    def evaluate_structure(self, actions, layer_nums, state_num=5):
        hidden_units_list = []
        out_channels_list = []
        for i in range(layer_nums):
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            hidden_units_list.append(head_num * out_channels)
            out_channels_list.append(out_channels)

        return out_channels_list[-1] == self.num_label

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=6):
        out_channels = None
        head_num = None

        # build hidden layer
        for i in range(layer_nums):

            # compute input
            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num

            # extract operator types from action
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            # Multi-head used in GAT.
            # "concat" is True, concat output of each head;
            # "concat" is False, get average of each head output;
            concat = True
            if i == layer_nums - 1:
                concat = False  # The last layer get average
            else:
                pass

            if i == 0:
                residual = False and self.residual  # special setting of dgl
            else:
                residual = True and self.residual
            self.layers.append(
                NASLayer(attention_type, aggregator_type, act, head_num, in_channels, out_channels, dropout=drop_out,
                         concat=concat, residual=residual, batch_normal=batch_normal))

    def forward(self, feat, g):

        output = feat
        for i, layer in enumerate(self.layers):
            output = layer(output, g)

        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    # map GNN's parameters into dict
    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = NASLayer.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        return result

    # load parameters from parameter dict
    def load_param(self, param):
        if param is None:
            return
        for i in range(self.layer_nums):
            self.layers[i].load_param(param["layer_%d" % i])


############################
#  Each layer of GNN
############################

def gat_message(edges):
    if 'norm' in edges.src:
        msg = edges.src['ft'] * edges.src['norm']
        return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1'], 'norm': msg}
    return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1']}


class NASLayer(nn.Module):
    def __init__(self, attention_type, aggregator_type, act, head_num, in_channels, out_channels=8, concat=True,
                 dropout=0.6, pooling_dim=128, residual=False, batch_normal=True):
        '''
        build one layer of GNN
        :param attention_type:
        :param aggregator_type:
        :param act: Activation function
        :param head_num: head num, in another word repeat time of current ops
        :param in_channels: input dimension
        :param out_channels: output dimension
        :param concat: concat output. get average when concat is False
        :param dropout: dropput for current layer
        :param pooling_dim: hidden layer dimension; set for pooling aggregator
        :param residual: whether current layer has  skip-connection
        :param batch_normal: whether current layer need batch_normal
        '''
        super(NASLayer, self).__init__()
        # print("NASLayer", in_channels, concat, residual)
        self.attention_type = attention_type
        self.aggregator_type = aggregator_type
        self.act = NASLayer.act_map(act)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = int(head_num)
        self.concat = concat
        self.dropout = dropout
        self.attention_dim = 1
        self.pooling_dim = pooling_dim

        self.batch_normal = batch_normal

        if attention_type in ['cos', 'generalized_linear']:
            self.attention_dim = 64
        self.bn = nn.BatchNorm1d(self.in_channels, momentum=0.5)
        self.prp = nn.ModuleList()
        self.red = nn.ModuleList()
        self.fnl = nn.ModuleList()
        self.agg = nn.ModuleList()
        for hid in range(self.num_heads):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.prp.append(AttentionPrepare(in_channels, out_channels, self.attention_dim, dropout))
            agg = NASLayer.aggregator_map(aggregator_type, out_channels, pooling_dim)
            self.agg.append(agg)
            self.red.append(NASLayer.attention_map(attention_type, dropout, agg, self.attention_dim))
            self.fnl.append(GATFinalize(hid, in_channels,
                                        out_channels, NASLayer.act_map(act), residual))

    @staticmethod
    def aggregator_map(aggregator_type, in_dim, pooling_dim):
        if aggregator_type == "sum":
            return SumAggregator()
        elif aggregator_type == "mean":
            return MeanPoolingAggregator(in_dim, pooling_dim)
        elif aggregator_type == "max":
            return MaxPoolingAggregator(in_dim, pooling_dim)
        elif aggregator_type == "mlp":
            return MLPAggregator(in_dim, pooling_dim)
        elif aggregator_type == "lstm":
            return LSTMAggregator(in_dim, pooling_dim)
        elif aggregator_type == "gru":
            return GRUAggregator(in_dim, pooling_dim)
        else:
            raise Exception("wrong aggregator type", aggregator_type)

    @staticmethod
    def attention_map(attention_type, attn_drop, aggregator, attention_dim):
        if attention_type == "gat":
            return GATReduce(attn_drop, aggregator)
        elif attention_type == "cos":
            return CosReduce(attn_drop, aggregator)
        elif attention_type in ["none", "const"]:
            return ConstReduce(attn_drop, aggregator)
        elif attention_type == "gat_sym":
            return GatSymmetryReduce(attn_drop, aggregator)
        elif attention_type == "linear":
            return LinearReduce(attn_drop, aggregator)
        elif attention_type == "bilinear":
            return CosReduce(attn_drop, aggregator)
        elif attention_type == "generalized_linear":
            return GeneralizedLinearReduce(attn_drop, attention_dim, aggregator)
        elif attention_type == "gcn":
            return GCNReduce(attn_drop, aggregator)
        else:
            raise Exception("wrong attention type")

    @staticmethod
    def act_map(act):
        if act == "linear":
            return lambda x: x
        elif act == "elu":
            return F.elu
        elif act == "sigmoid":
            return torch.sigmoid
        elif act == "tanh":
            return torch.tanh
        elif act == "relu":
            return torch.nn.functional.relu
        elif act == "relu6":
            return torch.nn.functional.relu6
        elif act == "softplus":
            return torch.nn.functional.softplus
        elif act == "leaky_relu":
            return torch.nn.functional.leaky_relu
        else:
            raise Exception("wrong activate function")

    def get_param_dict(self):
        params = {}

        key = "%d_%d_%d_%s" % (self.in_channels, self.out_channels, self.num_heads, self.attention_type)
        prp_key = key + "_" + str(self.attention_dim) + "_prp"
        agg_key = key + "_" + str(self.pooling_dim) + "_" + self.aggregator_type
        fnl_key = key + "_fnl"
        bn_key = "%d_bn" % self.in_channels
        params[prp_key] = self.prp.state_dict()
        params[agg_key] = self.agg.state_dict()
        # params[key+"_"+self.attention_type] = self.red.state_dict()
        params[fnl_key] = self.fnl.state_dict()
        params[bn_key] = self.bn.state_dict()
        return params

    def load_param(self, param):
        key = "%d_%d_%d_%s" % (self.in_channels, self.out_channels, self.num_heads, self.attention_type)
        prp_key = key + "_" + str(self.attention_dim) + "_prp"
        agg_key = key + "_" + str(self.pooling_dim) + "_" + self.aggregator_type
        fnl_key = key + "_fnl"
        bn_key = "%d_bn" % self.in_channels
        if prp_key in param:
            self.prp.load_state_dict(param[prp_key])

        # red_key = key+"_"+self.attention_type
        if agg_key in param:
            self.agg.load_state_dict(param[agg_key])
            for i in range(self.num_heads):
                self.red[i].aggregator = self.agg[i]

        if fnl_key in param:
            self.fnl.load_state_dict(param[fnl_key])

        if bn_key in param:
            self.bn.load_state_dict(param[bn_key])

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def forward(self, features, g):
        if self.batch_normal:
            last = self.bn(features)
        else:
            last = features

        for hid in range(self.num_heads):
            i = hid
            # prepare
            g.ndata.update(self.prp[i](last))
            # message passing
            g.update_all(gat_message, self.red[i], self.fnl[i])
        # merge all the heads
        if not self.concat:
            output = g.pop_n_repr('head0')
            for hid in range(1, self.num_heads):
                output = torch.add(output, g.pop_n_repr('head%d' % hid))
            output = output / self.num_heads
        else:
            output = torch.cat(
                [g.pop_n_repr('head%d' % hid) for hid in range(self.num_heads)], dim=1)
        del last
        return output


'''
    The whole process of each layer of GNN includes ibut not limited following action：
    1.Feature transform (Prepare)
    2.Correlation measure (Reduce)
    3.Aggregation ( Aggregator in Reduce )
    4.Residual connection (Finalize)
'''


############################################
# Prepare, set Attention Weight
############################################

class AttentionPrepare(nn.Module):
    '''
        Attention Prepare Layer
    '''

    def __init__(self, input_dim, hidden_dim, attention_dim, drop):
        super(AttentionPrepare, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        if drop:
            self.drop = nn.Dropout(drop)
        else:
            self.drop = 0
        self.attn_l = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attn_r = nn.Linear(hidden_dim, attention_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.weight.data, gain=1.414)

    def forward(self, feats):
        h = feats
        if self.drop:
            h = self.drop(h)
        ft = self.fc(h)
        a1 = self.attn_l(ft)
        a2 = self.attn_r(ft)
        return {'h': h, 'ft': ft, 'a1': a1, 'a2': a2}


######################################################################
# Reduce, apply different attention action and execute aggregation
######################################################################
class GATReduce(nn.Module):
    def __init__(self, attn_drop, aggregator=None):
        super(GATReduce, self).__init__()
        if attn_drop:
            self.attn_drop = nn.Dropout(p=attn_drop)
        else:
            self.attn_drop = 0
        self.aggregator = aggregator

    def apply_agg(self, neighbor):
        if self.aggregator:
            return self.aggregator(neighbor)
        else:
            return torch.sum(neighbor, dim=1)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        a = a.sum(-1, keepdim=True)  # Just in case the dimension is not zero
        e = F.softmax(F.leaky_relu(a), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class ConstReduce(GATReduce):
    '''
        Attention coefficient is 1
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(ConstReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        if self.attn_drop:
            ft = self.attn_drop(ft)
        return {'accum': self.apply_agg(1 * ft)}  # shape (B, D)


class GCNReduce(GATReduce):
    '''
        Attention coefficient is 1
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(GCNReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        if 'norm' not in nodes.data:
            raise Exception("Wrong Data, has no norm")
        self_norm = nodes.data['norm']
        self_norm = self_norm.unsqueeze(1)
        results = nodes.mailbox['norm'] * self_norm
        return {'accum': self.apply_agg(results)}  # shape (B, D)


class LinearReduce(GATReduce):
    '''
        equal to neighbor's self-attention
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(LinearReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        a2 = nodes.mailbox['a2']
        a2 = a2.sum(-1, keepdim=True)  # shape (B, deg, D)
        # attention
        e = F.softmax(torch.tanh(a2), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class GatSymmetryReduce(GATReduce):
    '''
        gat Symmetry version ( Symmetry cannot be guaranteed after softmax)
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(GatSymmetryReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        b1 = torch.unsqueeze(nodes.data['a2'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        b2 = nodes.mailbox['a1']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        b = b1 + b2  # different attention_weight
        a = a + b
        a = a.sum(-1, keepdim=True)  # Just in case the dimension is not zero
        e = F.softmax(F.leaky_relu(a + b), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class CosReduce(GATReduce):
    '''
        used in Gaan
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(CosReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 * a2
        a = a.sum(-1, keepdim=True)  # shape (B, deg, 1)
        e = F.softmax(F.leaky_relu(a), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class GeneralizedLinearReduce(GATReduce):
    '''
        used in GeniePath
    '''

    def __init__(self, attn_drop, hidden_dim, aggregator=None):
        super(GeneralizedLinearReduce, self).__init__(attn_drop, aggregator)
        self.generalized_linear = nn.Linear(hidden_dim, 1, bias=False)
        if attn_drop:
            self.attn_drop = nn.Dropout(p=attn_drop)
        else:
            self.attn_drop = 0

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2
        a = torch.tanh(a)
        a = self.generalized_linear(a)
        e = F.softmax(a, dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


#######################################################
# Aggregators, aggregate information from neighbor
#######################################################

class SumAggregator(nn.Module):

    def __init__(self):
        super(SumAggregator, self).__init__()

    def forward(self, neighbor):
        return torch.sum(neighbor, dim=1)


class MaxPoolingAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MaxPoolingAggregator, self).__init__()
        out_dim = input_dim
        self.fc = nn.ModuleList()
        self.act = act
        if num_fc > 0:
            for i in range(num_fc - 1):
                self.fc.append(nn.Linear(out_dim, pooling_dim))
                out_dim = pooling_dim
            self.fc.append(nn.Linear(out_dim, input_dim))

    def forward(self, ft):
        for layer in self.fc:
            ft = self.act(layer(ft))

        return torch.max(ft, dim=1)[0]


class MeanPoolingAggregator(MaxPoolingAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MeanPoolingAggregator, self).__init__(input_dim, pooling_dim, num_fc, act)

    def forward(self, ft):
        for layer in self.fc:
            ft = self.act(layer(ft))

        return torch.mean(ft, dim=1)


class MLPAggregator(MaxPoolingAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MLPAggregator, self).__init__(input_dim, pooling_dim, num_fc, act)

    def forward(self, ft):
        ft = torch.sum(ft, dim=1)
        for layer in self.fc:
            ft = self.act(layer(ft))
        return ft


class LSTMAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(input_dim, pooling_dim, batch_first=True, bias=False)
        self.linear = nn.Linear(pooling_dim, input_dim)

    def forward(self, ft):
        torch.transpose(ft, 1, 0)
        hidden = self.lstm(ft)[0]
        return self.linear(torch.squeeze(hidden[-1], dim=0))


class GRUAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.GRU(input_dim, pooling_dim, batch_first=True, bias=False)
        self.linear = nn.Linear(pooling_dim, input_dim)

    def forward(self, ft):
        torch.transpose(ft, 1, 0)
        hidden = self.lstm(ft)[0]
        return self.linear(torch.squeeze(hidden[-1], dim=0))


######################################################################
# Finalize, introduce residual connection
######################################################################
class GATFinalize(nn.Module):
    '''
        concat + fully connected layer
    '''

    def __init__(self, headid, indim, hiddendim, activation, residual):
        super(GATFinalize, self).__init__()
        self.headid = headid
        self.activation = activation
        self.residual = residual
        self.residual_fc = None
        if residual:
            if indim != hiddendim:
                self.residual_fc = nn.Linear(indim, hiddendim, bias=False)
                nn.init.xavier_normal_(self.residual_fc.weight.data, gain=1.414)

    def forward(self, nodes):
        ret = nodes.data['accum']
        if self.residual:
            if self.residual_fc is not None:
                ret = self.residual_fc(nodes.data['h']) + ret
            else:
                ret = nodes.data['h'] + ret
        return {'head%d' % self.headid: self.activation(ret)}
