from .._nnimc_torch import TorchQuantizer, _torch_detect_module, _torch_default_get_configure
from .conv2d_kse import Conv2d_KSE
import torch
import torch.nn as nn

class KSEQuantizer(TorchQuantizer):
    """
    Use algorithm from "ExploitingKernelSparsityandEntropyforInterpretableCNNCompression" 
    https://arxiv.org/abs/1812.04368
    """
    def __init__(self, configure_list):
        """
            configure Args:
                G: compression granularity
                T: compression ratio
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')

    def get_GT(self, configure):
        if not isinstance(configure, dict):
            logger.warning('WARNING: you should input a dict to get_GT, set DEFAULT { }')
            configure = {}
        G = configure.get('G', 4)
        T = configure.get('T', 0)
        return G, T
    
    def compress(self, model):
        for layer_info in _torch_detect_module(model, nn.Conv2d):
            G, T = self.get_GT(_torch_default_get_configure(self.configure_list, layer_info))
            # replace origin conv2d with conv2d_kse
            conv2d_kse = Conv2d_KSE(
                input_channels = layer_info.layer.in_channels, 
                output_channels = layer_info.layer.out_channels, 
                kernel_size = layer_info.layer.kernel_size, 
                stride = layer_info.layer.stride, 
                padding = layer_info.layer.padding, 
                bias=False, 
                G=G, 
                T=T)
            conv2d_kse.weight = layer_info.layer.weight
            conv2d_kse.bias = layer_info.layer.bias

            # calculate clusters and index
            conv2d_kse.KSE(G, T)
            conv2d_kse.forward_init()

            # replace origin conv2d with conv2d_kse module
            layer_info.layer = conv2d_kse
            setattr(model, layer_info.name, layer_info.layer)
            self._instrument_layer(layer_info)

    def quantize_weight(self, layer_info, weight):
        cluster_weights = []
        layer = layer_info.layer
        for g in range(1, layer.G - 1):
            if layer.group_size[g] == 0:
                continue
            cluster = layer.__getattr__("clusters_" + str(g))
            clusters = cluster.permute(1, 0, 2, 3).contiguous().view(
                layer.cluster_num[g] * cluster.shape[1], *layer.kernel_size)
            cluster_weight = clusters[
                layer.__getattr__("cluster_indexs_" + str(g)).data].contiguous().view(
                layer.output_channels, cluster.shape[1], *layer.kernel_size)
            cluster_weights.append(cluster_weight)

        if len(cluster_weights) == 0:
            weight = layer.full_weight
        else:
            weight = torch.cat((layer.full_weight, torch.cat(cluster_weights, dim=1)), dim=1)
        return weight