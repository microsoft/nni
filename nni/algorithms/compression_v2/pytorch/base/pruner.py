import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression_v2.pytorch.base.compressor import Compressor, LayerInfo
from nni.algorithms.compression_v2.pytorch.base.common import DataCollector, MetricsCalculator, SparsityAllocator

_logger = logging.getLogger(__name__)


class PrunerModuleWrapper(Module):
    def __init__(self, module: Module, module_name: str, module_type: str, config: Dict, pruner: Compressor):
        """
        Wrap an module to enable data parallel, forward method customization and buffer registeration.

        Parameters
        ----------
        module
            The module user wants to compress.
        config
            The configurations that users specify for compression.
        module_name
            The name of the module to compress, wrapper module shares same name.
        module_type
            The type of the module to compress.
        pruner
            The pruner used to calculate mask.
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type = module_type
        # config and pruner
        self.config = config
        self.pruner = pruner

        # register buffer for mask
        self.register_buffer("weight_mask", torch.ones(self.module.weight.shape))
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.register_buffer("bias_mask", torch.ones(self.module.bias.shape))
        else:
            self.register_buffer("bias_mask", None)

    def forward(self, *inputs):
        # apply mask to weight, bias
        self.module.weight.data = self.module.weight.data.mul_(self.weight_mask)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.module.bias.data = self.module.bias.data.mul_(self.bias_mask)
        return self.module(*inputs)


class Pruner(Compressor):
    """
    The abstract class for pruning algorithm. Inherit this class and implement the `_reset_tools` to customize a pruner.
    """

    def reset(self, model: Optional[Module] = None, config_list: Optional[List[Dict]] = None):
        super().reset(model=model, config_list=config_list)
        self.data_collector: Optional[DataCollector] = None
        self.metrics_calculator: Optional[MetricsCalculator] = None
        self.sparsity_allocator: Optional[SparsityAllocator] = None
        self._reset_tools()

    def _reset_tools(self):
        """
        This function is used to reset `self.data_collector`, `self.metrics_calculator` and `self.sparsity_allocator`.
        The subclass need implement this function to complete the pruning process.
        See `compress()` to understand how NNI use these three part to generate mask for the bound model.
        """
        raise NotImplementedError()

    def _wrap_modules(self, layer: LayerInfo, config: Dict):
        """
        Create a wrapper module to replace the original one.

        Parameters
        ----------
        layer
            The layer to instrument the mask.
        config
            The configuration for generating the mask.
        """
        _logger.debug("Module detected to compress : %s.", layer.name)
        wrapper = PrunerModuleWrapper(layer.module, layer.name, layer.type, config, self)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        # move newly registered buffers to the same device of weight
        wrapper.to(layer.module.weight.device)
        return wrapper

    def load_masks(self, masks: Dict[str, Dict[str, Tensor]]):
        wrappers = self._get_modules_wrapper()
        for name, layer_mask in masks.items():
            assert name in wrappers, '{} is not in wrappers of this pruner, can not apply the mask.'.format(name)
            for mask_type, mask in layer_mask.items():
                assert hasattr(wrappers[name], mask_type), 'there is no attribute {} in wrapper'.format(mask_type)
                setattr(wrappers[name], mask_type, mask)

    def compress(self) -> Tuple[Module, Dict]:
        """
        Used to generate the mask. Pruning process is divided in three stages.
        `self.data_collector` collect the data used to calculate the specify metric.
        `self.metrics_calculator` calculate the metric and `self.sparsity_allocator` generate the mask depend on the metric.

        Returns
        -------
        Tuple[Module, Dict]
            Return the wrapped model and mask.
        """
        data = self.data_collector.collect()
        _logger.debug('Collected Data:\n%s', data)
        metrics = self.metrics_calculator.calculate_metrics(data)
        _logger.debug('Metrics Calculate:\n%s', metrics)
        masks = self.sparsity_allocator.generate_sparsity(metrics)
        _logger.debug('Masks:\n%s', masks)
        self.load_masks(masks)
        return self.bound_model, masks

    # NOTE: need refactor dim with supporting list
    def show_pruned_weights(self, dim: int = 0):
        """
        Log the simulated prune sparsity.

        Parameters
        ----------
        dim
            The pruned dim.
        """
        for _, wrapper in self._get_modules_wrapper().items():
            weight_mask = wrapper.weight_mask
            mask_size = weight_mask.size()
            if len(mask_size) == 1:
                index = torch.nonzero(weight_mask.abs() != 0, as_tuple=False).tolist()
            else:
                sum_idx = list(range(len(mask_size)))
                sum_idx.remove(dim)
                index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0, as_tuple=False).tolist()
            _logger.info(f'simulated prune {wrapper.name} remain/total: {len(index)}/{weight_mask.size(dim)}')
