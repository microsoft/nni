import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import math
import numpy as np
import time
import pdb
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from math import isnan

import logging
logger = logging.getLogger('KSE')

def density_entropy(X):
    K = 5
    N, C, D = X.shape
    x = X.transpose(1, 0, 2).reshape(C, N, -1)
    score = np.zeros(C)
    dms = np.zeros(N)

    for c in range(C):
        nbrs = NearestNeighbors(n_neighbors=K + 1).fit(x[c])
        for i in range(N):
            dist, ind = nbrs.kneighbors(x[c, i].reshape(1, -1))
            dms[i] = sum([dist[0][j + 1] for j, id in enumerate(ind[0][1:])])
        
        dms_sum = sum(dms)
        score[c] = sum([-dms[i] / dms_sum * math.log(dms[i]/dms_sum, 2) for i in range(N)])
    return np.array(score)

class Conv2d_KSE(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=True, G=4, T=0):
        super(Conv2d_KSE, self).__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.isbias = bias
        self.T = T

        if G == 0:
            if output_channels >= input_channels:
                self.G = input_channels
            else:
                self.G = math.ceil(input_channels/output_channels)
        else:
            self.G = G
        self.group_num = self.G
        self.weight = nn.Parameter(torch.Tensor(output_channels, input_channels, *kernel_size))
        self.mask = nn.Parameter(torch.Tensor(input_channels), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channels))
        else:
            self.register_parameter('bias', None)

    def __repr__(self):
        return self.__class__.__name__ \
               + "({" + str(self.input_channels) \
               + "}, {" + str(self.output_channels) \
               + "}, kernel_size={" + str(self.kernel_size) + "}, stride={" + \
               str(self.stride) + "}, padding={" + str(self.padding) + "})"

    def forward(self, input):
        # overwrite nn.module forward method
        # transform cluster and index into weight for training
        return F.conv2d(torch.index_select(input, 1, self.channels_indexs), self.weight, self.bias,
                        stride=self.stride,
                        padding=self.padding)

    def KSE(self, G=None, T=None):
        if G is not None:
            self.G = G
        if T is not None:
            self.T = T
        weight = self.weight.data.cpu().numpy()
        weight = weight.transpose(1, 0, 2, 3).reshape(self.input_channels, self.output_channels, -1)

        # Calculate channels KSE indicator.
        # input channel is only 1
        if self.input_channels <= 1:
            indicator = np.array([1.0])
        else:
            ks_weight = np.sum(np.linalg.norm(weight, ord=1, axis=2), 1)
            ke_weight = density_entropy(weight.reshape(self.output_channels, self.input_channels, -1))
            ks_weight = (ks_weight - np.min(ks_weight)) / (np.max(ks_weight) - np.min(ks_weight))
            ke_weight = (ke_weight - np.min(ke_weight)) / (np.max(ke_weight) - np.min(ke_weight))
            indicator = np.sqrt(ks_weight / (1 + ke_weight))
            indicator = (indicator - np.min(indicator))/(np.max(indicator) - np.min(indicator))
        # Calculate each input channels kernel number and split them into different groups.
        # Each group has same kernel number.
        self.group_size = [0 for i in range(self.G)]
        self.cluster_num = [1 for i in range(self.G)]

        for i in range(self.input_channels):
            if math.floor(indicator[i] * self.G) == 0:
                self.mask[i] = 0
                self.group_size[0] += 1
            elif math.ceil(indicator[i] * self.G) == self.G:
                self.mask[i] = self.G - 1
                self.group_size[-1] += 1
            else:
                self.mask[i] = math.floor(indicator[i] * self.G)
                self.group_size[int(math.floor(indicator[i] * self.G))] += 1

        for i in range(self.G):
            if i == 0:
                self.cluster_num[i] = 0
            elif i == self.G - 1:
                self.cluster_num[i] = self.output_channels
            else:
                self.cluster_num[i] = math.ceil(self.output_channels * math.pow(2, i + 1 - self.T - self.G))

        # Calculate cluster and index by k-means
        # First, collect weight corresponding to each group(same kernel number)
        weight = self.weight.data.cpu().numpy()
        weight_group = []
        for g in range(self.G):
            if self.group_size[g] == 0:
                weight_group.append([])
                continue
            
            each_weight_group = []
            for c in range(self.input_channels):
                if self.mask[c] == g:
                    each_weight_group.append(np.expand_dims(weight[:, c], 1))
            each_weight_group = np.concatenate(each_weight_group, 1)
            weight_group.append(each_weight_group)
        self.full_weight = nn.Parameter(torch.Tensor(weight_group[-1]),requires_grad=True)

        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster_weight = weight_group[g]
            cluster_weight = cluster_weight.transpose((1, 0, 2, 3)).reshape((
                self.group_size[g], self.output_channels, -1))

            clusters = []
            indexs = []
            for c in range(cluster_weight.shape[0]):
                kmean = KMeans(n_clusters=self.cluster_num[g]).fit(cluster_weight[c])
                centroids = kmean.cluster_centers_
                assignments = kmean.labels_

                clusters.append(np.expand_dims(np.reshape(centroids, [
                    self.cluster_num[g], *self.kernel_size]), 1))
                indexs.append(np.expand_dims(assignments, 1))

            clusters = np.concatenate(clusters, 1)
            indexs = np.concatenate(indexs, 1)
            self.__setattr__("clusters_" + str(g), nn.Parameter(torch.Tensor(clusters), requires_grad=True))
            self.__setattr__("indexs_" + str(g), nn.Parameter(torch.Tensor(indexs), requires_grad=False))

    
    def forward_init(self):
        # record the channel index of each group

        full_index = []     # input channels index which kernel number = N
        cluster_indexs = [] # input channels index which kernel number != N && != 0
        all_indexs = []     # input channels index which kernel number != 0

        for i, m in enumerate(self.mask.data):
            if m == self.G - 1:
                full_index.append(i)
                all_indexs.append(i)

        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                cluster_indexs.append([])
                continue
            cluster_index = []
            for i, m in enumerate(self.mask.data):
                if m == g:
                    cluster_index.append(i)
                    all_indexs.append(i)
            cluster_indexs.append(cluster_index)

        self.channels_indexs = nn.Parameter(torch.zeros(self.input_channels - self.group_size[0]).long(), # 减去全部消除的 即完全不重要的
                                            requires_grad=False)

        # transform index for training
        if self.full_weight.is_cuda:
            self.channels_indexs.data = torch.LongTensor(all_indexs).cuda()
            self.channel_indexs = []
            for g in range(1, self.G - 1):
                if self.group_size[g] == 0:
                    continue
                index = self.__getattr__("indexs_" + str(g))
                self.__setattr__("cluster_indexs_" + str(g), nn.Parameter(
                    (index.data + self.cluster_num[g] * torch.Tensor(
                        [i for i in range(self.group_size[g])]).view(1, -1).cuda()).view(-1).long(),
                    requires_grad=False))

        else:
            self.channels_indexs.data = torch.LongTensor(all_indexs)
            self.channel_indexs = []
            for g in range(1, self.G - 1):
                if self.group_size[g] == 0:
                    continue
                index = self.__getattr__("indexs_" + str(g))
                self.__setattr__("cluster_indexs_" + str(g), nn.Parameter(
                    (index.data + self.cluster_num[g] * torch.Tensor(
                        [i for i in range(self.group_size[g])]).view(1, -1)).view(-1).long(),
                    requires_grad=False))

    def create_arch(self, G=None, T=None):
        if G is not None:
            self.G = G
        if T is not None:
            self.T = T

        # create architecture (clusters and indexs) base on mask
        mask = self.mask.data.cpu().numpy()
        self.group_size = [0 for i in range(self.G)]
        self.cluster_num = [1 for i in range(self.G)]

        for i in range(self.input_channels):
            self.group_size[int(mask[i])] += 1

        for i in range(self.G):
            if i == 0:
                self.cluster_num[i] = 0
            elif i == self.G - 1:
                self.cluster_num[i] = self.output_channels
            else:
                self.cluster_num[i] = math.ceil(self.output_channels * math.pow(2, i + 1 - self.T - self.G))

        self.full_weight = nn.Parameter(
            torch.Tensor(self.output_channels, self.group_size[-1], *self.kernel_size),
            requires_grad=True)

        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            else:
                cluster = nn.Parameter(torch.zeros(
                    self.cluster_num[g], self.group_size[g], *self.kernel_size), requires_grad=True)
                index = nn.Parameter(torch.ByteTensor(math.ceil(
                    math.ceil(math.log(self.cluster_num[g], 2)) * self.output_channels * self.group_size[g] / 8)),
                    requires_grad=False)
                self.__setattr__("clusters_" + str(g), cluster)
                self.__setattr__("indexs_" + str(g), index)

    def load(self):
        # tranform index
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster = self.__getattr__("clusters_" + str(g))
            index = self.__getattr__("indexs_"+str(g))

            Q = cluster.shape[0]
            bits = math.ceil(math.log(Q, 2))
            indexs = index.data.cpu().numpy()
            new_b = ""
            f = '{0:08b}'

            for n, i in enumerate(indexs):
                if (self.output_channels*self.group_size[g]*bits)% 8 != 0 and n == indexs.shape[0]-1:
                    continue
                new_b += f.format(i)

            if (self.output_channels * self.group_size[g] * bits) % 8 != 0:
                va = (self.output_channels * self.group_size[g] * bits) % 8
                new_b += f.format(indexs[-1])[-va:]

            new_index = []

            for i in range(int(len(new_b)/bits)):
                b = new_b[i*bits:(i+1)*bits]
                v = float(int(b, 2))
                new_index.append(v)

            ni = torch.Tensor(new_index).view(self.output_channels, self.group_size[g])
            self.__delattr__("indexs_" + str(g))
            self.__setattr__("indexs_"+str(g), nn.Parameter(ni, requires_grad=False))

    def save(self):
        # tranform index
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster = self.__getattr__("clusters_" + str(g))
            index = self.__getattr__("indexs_"+str(g))

            Q = cluster.shape[0]

            bits = math.ceil(math.log(Q, 2))

            new_b = ""
            indexs = index.data.cpu().numpy().reshape(-1)
            f = '{0:0'+str(bits)+'b}'
            for i in indexs:
                nb = f.format(int(i))
                new_b += nb

            new_index = []
            for i in range(int(len(new_b)*1.0/8)):
                new_value = int(new_b[i*8:(i+1)*8], 2)
                new_index.append(new_value)

            if len(new_b) % 8 != 0:
                new_value = int(new_b[int(len(new_b)*1.0/8)*8:], 2)
                new_index.append(new_value)

            self.__delattr__("indexs_"+str(g))
            self.__setattr__("indexs_"+str(g), nn.Parameter(torch.ByteTensor(new_index), requires_grad=False))
            self.__delattr__("cluster_indexs_" + str(g))
        self.__delattr__("channels_indexs")
        self.__delattr__("group_size")
        self.__delattr__("cluster_num")
        self.__delattr__("weight")
        if self.bias is not None:
            self.__delattr__("bias")