import torch
from torch import nn
import numpy as np
from copy import deepcopy
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification, HubertConfig, HubertModel
from transformers.optimization import AdamW
from datasets import load_dataset
import soundfile as sf

from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni.compression.pytorch.utils.utils import get_module_by_name
from torch.utils.data import Dataset, DataLoader






def finegrain_pruned_hubert(model, sparsity):
    config_list = [{'op_types':['Linear'], 'sparsity':sparsity}]
    pruner = LevelPruner(model, config_list)
    pruner.compress()
    return model, pruner





def copy_tensor(t1, t2):
    shape_list = list(t1.size())
    index = []
    for _size in shape_list:
        index.append(slice(0, _size))
    t1.data = t2.data[index]

def inherit_weight(model, ori_model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            print(type(module))
            # if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            #     import pdb; pdb.set_trace()
            _, ori_module = get_module_by_name(ori_model, name)
            copy_tensor(module.weight, ori_module.weight)
            # import pdb; pdb.set_trace()
            if hasattr(module, 'bias') and module.bias is not None:
                copy_tensor(module.bias, ori_module.bias)



def apply_mask(model, mask, device=torch.device('cuda')):
    cfglist = []
    for layer in mask:
        # here the sparsity ratio doesn't matter, because we will
        # assign the mask manually
        cfglist.append({'sparsity':0.9, 'op_types':['Linear'], 'op_names':[layer]})
    tmp_pruner = LevelPruner(model, cfglist)
    tmp_pruner.compress()
    
    for name in mask:
        _, module = get_module_by_name(model, name)
        for key in mask[name]:
            # print(module)
            # import ipdb; ipdb.set_trace()
            assert(hasattr(module,key + '_mask'))
            setattr(module, key+'_mask', mask[name][key].to(device))
