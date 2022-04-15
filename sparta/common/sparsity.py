import torch
import torch.nn as nn

__all__ = ['TeSA', 'SparseModuleInfo', 'ModelSparsityInfo']

# TODO: more elegant definition for TesaAttr
TesaAttr = {-1: 'constant',
            0: 'pruned',
            4: 'int4',
            7: 'uint8',
            8: 'int8',
            16: 'float16',
            31: 'int32',
            32: 'float32',
            33: 'nonpruned'}

class TeSA:
    def __init__(self, tesaattr_tensor: torch.Tensor):
        self.tesa: torch.Tensor = tesaattr_tensor
        # NOTE: can be refactored here to support balanced sparsity pattern
        self.block_size: tuple = None
        # number of different bits in this tesa
        self.n_bits: int = None

    def set_transform_meta(self, block_size: tuple, n_bits: int):
        # this meta information is for guiding kernel specialization
        self.block_size = block_size
        self.n_bits = n_bits

class SparseModuleInfo:
    """
    Attributes
    ----------
    ...
    """
    def __init__(self, module_name: str,
                 module_obj: nn.Module,
                 weight_tesa: torch.Tensor = None,
                 input_tesa: torch.Tensor = None,
                 output_tesa: torch.Tensor = None):
        self.module_name = module_name
        self.module_obj = module_obj
        self.weight_tesa = TeSA(weight_tesa)
        self.input_tesa = TeSA(input_tesa)
        self.output_tesa = TeSA(output_tesa)

class ModelSparsityInfo:
    """
    Attributes
    ----------
    ...
    """
    def __init__(self):
        self.modules_info: dict = {}

    def update(self, info: SparseModuleInfo):
        if info.module_name in self.modules_info:
            # merge sparsity
            ...
        else:
            self.modules_info[info.module_name] = info
    
    def items(self):
        return self.modules_info.items()

class ModelDataLayouts:
    def __init__(self):
        ...