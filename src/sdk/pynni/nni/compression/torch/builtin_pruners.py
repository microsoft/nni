import logging
import torch
from .compressor import Pruner

__all__ = ['LevelPruner', 'AGP_Pruner', 'FilterPruner', 'SlimPruner']

logger = logging.getLogger('torch pruner')


class LevelPruner(Pruner):
    """Prune to an exact pruning level specification

    Attributes
    ----------
    mask_list : dict
        Dictionary for saving masks.

    """

    def __init__(self, model, config_list):
        """
        config_list: supported keys:
            - sparsity

        Parameters
        ----------
        config_list : list
            List on pruning configs

        """
        super().__init__(model, config_list)
        self.mask_list = {}
        self._if_init_list = {}

    def calc_mask(self, layer, config):
        """Calculate the mask of given layer

        Parameters
        ----------
        weight : torch.nn.Parameter
            weight of layer to prune
        config : dict
            layer's pruning config
        op_name : str
            layer name to be pruned
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            mask of the layer's weight

        """

        weight = layer.module.weight.data
        op_name = layer.name
        if self._if_init_list.get(op_name, True):
            w_abs = weight.abs()
            k = int(weight.numel() * config['sparsity'])
            if k == 0:
                return torch.ones(weight.shape).type_as(weight)
            threshold = torch.topk(w_abs.view(-1), k, largest=False).values.max()
            mask = torch.gt(w_abs, threshold).type_as(weight)
            self.mask_list.update({op_name: mask})
            self._if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask


class AGP_Pruner(Pruner):
    """Prune to an exact pruning level specification
    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.

    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    https://arxiv.org/pdf/1710.01878.pdf

    Attributes
    ----------
    mask_list : dict
        Dictionary for saving masks.

    """

    def __init__(self, model, config_list):
        """
        config_list: supported keys:
            - initial_sparsity
            - final_sparsity: you should make sure initial_sparsity <= final_sparsity
            - start_epoch: start epoch number begin update mask
            - end_epoch: end epoch number stop update mask, you should make sure start_epoch <= end_epoch
            - frequency: if you want update every 2 epoch, you can set it 2

        Parameters
        ----------
        config_list : list
            List on pruning configs

        """
        super().__init__(model, config_list)
        self.mask_list = {}
        self.now_epoch = 0
        self._if_init_list = {}

    def calc_mask(self, layer, config):
        """Calculate the mask of given layer

        Parameters
        ----------
        weight : torch.nn.Parameter
            Weight of layer to prune
        config : dict
            Layer's pruning config
        op_name : str
            Layer name to be pruned
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            Mask of the layer's weight

        """
        weight = layer.module.weight.data
        op_name = layer.name
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        if self.now_epoch >= start_epoch and self._if_init_list.get(op_name, True) and (
                self.now_epoch - start_epoch) % freq == 0:
            mask = self.mask_list.get(op_name, torch.ones(weight.shape).type_as(weight))
            target_sparsity = self.compute_target_sparsity(config)
            k = int(weight.numel() * target_sparsity)
            if k == 0 or target_sparsity >= 1 or target_sparsity <= 0:
                return mask
            # if we want to generate new mask, we should update weigth first
            w_abs = weight.abs() * mask
            threshold = torch.topk(w_abs.view(-1), k, largest=False).values.max()
            new_mask = torch.gt(w_abs, threshold).type_as(weight)
            self.mask_list.update({op_name: new_mask})
            self._if_init_list.update({op_name: False})
        else:
            new_mask = self.mask_list.get(op_name, torch.ones(weight.shape).type_as(weight))
        return new_mask

    def compute_target_sparsity(self, config):
        """Calculate the sparsity for pruning

        Parameters
        ----------
        config : dict
            Layer's pruning config

        Returns
        -------
        float
            Target sparsity to be pruned

        """
        end_epoch = config.get('end_epoch', 1)
        start_epoch = config.get('start_epoch', 0)
        freq = config.get('frequency', 1)
        final_sparsity = config.get('final_sparsity', 0)
        initial_sparsity = config.get('initial_sparsity', 0)
        if end_epoch <= start_epoch or initial_sparsity >= final_sparsity:
            logger.warning('your end epoch <= start epoch or initial_sparsity >= final_sparsity')
            return final_sparsity

        if end_epoch <= self.now_epoch:
            return final_sparsity

        span = ((end_epoch - start_epoch - 1) // freq) * freq
        assert span > 0
        target_sparsity = (final_sparsity +
                           (initial_sparsity - final_sparsity) *
                           (1.0 - ((self.now_epoch - start_epoch) / span)) ** 3)
        return target_sparsity

    def update_epoch(self, epoch):
        """Update epoch

        Parameters
        ----------
        epoch : int
            current training epoch

        """
        if epoch > 0:
            self.now_epoch = epoch
            for k in self._if_init_list.keys():
                self._if_init_list[k] = True


class FilterPruner(Pruner):
    """A structured pruning algorithm that prunes the filters of smallest magnitude
    weights sum in the convolution layers to achieve a preset level of network sparsity.

    Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf,
    "PRUNING FILTERS FOR EFFICIENT CONVNETS", 2017 ICLR
    https://arxiv.org/abs/1608.08710

    Attributes
    ----------
    mask_list : dict
        Dictionary for saving masks.

    """

    def __init__(self, config_list):
        """Initiate `FilterPruner` from `config_list`
        config_list: supported keys:
            - sparsity

        Parameters
        ----------
        config_list : list
            List of pruning configs

        """

        super().__init__(config_list)
        self.mask_list = {}
        self._if_init_list = {}

    def calc_mask(self, layer, config):
        """Calculate the mask of given layer

         Parameters
         ----------
         weight : torch.nn.Parameter
             Weight of layer to prune
         config : dict
             Layer's pruning config
         op_name : str
             Layer name to be pruned
         op_type : str
             Layer type to be pruned
         **kwargs
             Arbitrary keyword arguments.

         Returns
         -------
         torch.Tensor
             Mask of the layer's weight

         """
        weight = layer.module.weight.data
        op_name = layer.name
        op_type = layer.type
        assert op_type == 'Conv2d', 'FilterPruner only supports 2d convolution layer pruning'
        if self._if_init_list.get(op_name, True):
            kernels = weight.shape[0]
            w_abs = weight.abs()
            k = int(kernels * config['sparsity'])
            if k == 0:
                return torch.ones(weight.shape).type_as(weight)
            w_abs_structured = w_abs.view(kernels, -1).sum(dim=1)
            threshold = torch.topk(w_abs_structured.view(-1), k, largest=False).values.max()
            mask = torch.gt(w_abs_structured, threshold)[:, None, None, None].expand_as(weight).type_as(weight)
            self.mask_list.update({op_name: mask})
            self._if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask


class SlimPruner(Pruner):
    """A structured pruning algorithm that prunes channels by pruning the weights of BN layers.

    Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan and Changshui Zhang
    "Learning Efficient Convolutional Networks through Network Slimming", 2017 ICCV
    https://arxiv.org/pdf/1708.06519.pdf

    Attributes
    ----------
    mask_list : dict
        Dictionary for saving masks.

    """

    def __init__(self, config_list):
        """Initiate `FilterPruner` from `config_list`
        config_list: supported keys:
            - sparsity

        Parameters
        ----------
        config_list : list
            List of pruning configs

        """
        super().__init__(config_list)
        self.mask_list = {}
        self._if_init_list = {}

    def bind_model(self, model):
        """Calculate the global threshold for pruning

        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned

        """
        weight_list = []
        config = self.config_list[0]
        op_types = config.get('op_types')
        op_names = config.get('op_names')
        if op_types is not None:
            assert op_types == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
            for name, m in model.named_modules():
                if type(m).__name__ == 'BatchNorm2d':
                    weight_list.append(m.weight.data.clone())
        else:
            for name, m in model.named_modules():
                if name in op_names:
                    assert type(
                        m).__name__ == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
                    weight_list.append(m.weight.data.clone())
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * config['sparsity'])
        self.global_threshold = torch.topk(all_bn_weights.view(-1), k, largest=False).values.max()

    def calc_mask(self, layer, config):
        """Calculate the mask of given layer

         Parameters
         ----------
         weight : torch.nn.Parameter
             Weight of layer to prune
         config : dict
             Layer's pruning config
         op_name : str
             Layer name to be pruned
         op_type : str
             Layer type to be pruned
         **kwargs
             Arbitrary keyword arguments.

         Returns
         -------
         torch.Tensor
             Mask of the layer's weight

         """
        weight = layer.module.weight.data
        op_name = layer.name
        op_type = layer.type
        assert op_type == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
        if self._if_init_list.get(op_name, True):
            w_abs = weight.abs()
            mask = torch.gt(w_abs, self.global_threshold).type_as(weight)
            self.mask_list.update({op_name: mask})
            self._if_init_list.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask
