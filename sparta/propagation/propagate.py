import torch
import torch.nn as nn

from sparta.common import ModelSparsityInfo, SparseModuleInfo

__all__ = ['propagate_sparsity']

def extract_sparsity(model: nn.Module):
    model_sparsity = ModelSparsityInfo()
    for name, module in model.named_modules():
        print('module name: ', name)
        weight_tesa = input_tesa = output_tesa = None
        if hasattr(module, 'weight_tesa'):
            weight_tesa = module.weight_tesa
        if hasattr(module, 'input_tesa'):
            input_tesa = module.input_tesa
        if hasattr(module, 'output_tesa'):
            output_tesa = module.output_tesa
        if weight_tesa != None or input_tesa != None or output_tesa != None:
            model_sparsity.update(
                SparseModuleInfo(name, module, weight_tesa,
                    input_tesa, output_tesa)
            )
    return model_sparsity

def propagate_sparsity(model: nn.Module) -> ModelSparsityInfo:
    pre_sparsity = extract_sparsity(model)
    # mocked
    post_sparsity = pre_sparsity
    return post_sparsity
