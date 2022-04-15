import argparse
import glob
import json
import logging

import os
import random
import math

import numpy as np
import torch
# from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from nni.compression.pytorch.utils import bert_compression_utils, get_module_by_name
from emmental import MaskedBertConfig, MaskedBertForSequenceClassification
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from emmental.modules.masked_nn import MaskedLinear
import nni
import torch
import sys
import os
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils.bert_compression_utils import BertCompressModule
from bert_utils_ptq import *
from nni.algorithms.compression.pytorch.pruning import TransformerHeadPruner
from nni.algorithms.compression.pytorch.quantization import ObserverQuantizer

def calibration(model, device, test_loader):
    model.eval()
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calibrating"):
            batch = tuple(t.to(device) for t in batch)
            
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]}
            
            inputs["token_type_ids"] = batch[2] 
                    # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            model(**inputs)

device = torch.device('cpu')
config = torch.load('Coarse_bert_config')
dummy_input = torch.load('dummy_input.pth', map_location=device)
data = (dummy_input['input_ids'].to(device), dummy_input['attention_mask'].to(device), dummy_input['token_type_ids'].to(device))
norm_model = BertForSequenceClassification(config=config).to(device)
# import ipdb; ipdb.set_trace()
mlp_prune_cfg = torch.load('checkpoints/coarsegrained/mlp_coarseprune_cfg')
bert_head_size = 64
token = BertTokenizer.from_pretrained('checkpoints/finegrained/checkpoint-220000')
mask_file = 'checkpoints/coarsegrained/coarse_baseline_mask.pth'
# norm_model.prune_heads(mlp_prune_cfg)

# coarse_mask = {}
# for layer_id in mlp_prune_cfg:
#      query_layer = 'bert.encoder.layer.{}.attention.self.query'.format(layer_id)
#      key_layer = 'bert.encoder.layer.{}.attention.self.key'.format(layer_id)
#      value_layer = 'bert.encoder.layer.{}.attention.self.value'.format(layer_id)
#      _, module_query = get_module_by_name(norm_model, query_layer)
#      _, module_key = get_module_by_name(norm_model, key_layer)
#      _, module_value = get_module_by_name(norm_model, value_layer)
#      coarse_mask[query_layer] = {'weight':torch.ones_like(module_query.weight), 'bias':torch.ones_like(module_query.bias) }
#      coarse_mask[key_layer] = { 'weight':torch.ones_like(module_key.weight), 'bias':torch.ones_like(module_key.bias) }
#      coarse_mask[value_layer] = {'weight':torch.ones_like(module_value.weight), 'bias':torch.ones_like(module_value.bias) }
#      for headid in mlp_prune_cfg[layer_id]:
#          _start = headid * bert_head_size
#          _end = (headid + 1) * bert_head_size
#          coarse_mask[query_layer]['weight'].data[_start:_end] = 0
#          coarse_mask[query_layer]['bias'].data[_start:_end] = 0         
#          coarse_mask[key_layer]['weight'].data[_start:_end] = 0
#          coarse_mask[key_layer]['bias'].data[_start:_end] = 0
#          coarse_mask[value_layer]['weight'].data[_start:_end] = 0
#          coarse_mask[value_layer]['bias'].data[_start:_end] = 0
# for name, module in norm_model.named_modules():
#     if isinstance(module, torch.nn.Linear) and name not in coarse_mask:
#         _, module = get_module_by_name(norm_model, name)
#         coarse_mask[name] = {'weight':torch.ones_like(module.weight), 'bias':torch.ones_like(module.bias)}
# torch.save(coarse_mask, 'checkpoints/coarsegrained/Bert_coarse_mask.pth')

ms = ModelSpeedup(norm_model, data, mask_file, break_points=[], confidence=32)

# get the propagated mask
propagated_mask = ms.propagate_mask()
ori_mask =  torch.load(mask_file)
for name in propagated_mask:
    print('New Sparsity ', name, 1-torch.sum(propagated_mask[name]['weight'])/propagated_mask[name]['weight'].numel(), 1-torch.sum(ori_mask[name]['weight'])/ori_mask[name]['weight'].numel())


BertCompressModule(norm_model, propagated_mask, mlp_prune_cfg)
norm_model.load_state_dict(torch.load('checkpoints/coarsegrained/nni_weights.pth'))
# import ipdb; ipdb.set_trace()
# apply_mask(norm_model, propagated_mask)
acc = evaluate(norm_model.cuda(), token)
train_dataset = load_and_cache_examples("qqp", token, evaluate=False)
# optimizer = train(train_dataset, norm_model, token, num_train_epochs=100)
print('Propagate done')

####### apply mask
for name, module in norm_model.named_modules():
    if name in propagated_mask:
        module.weight_mask = propagated_mask[name]['weight']
        module.bias_mask = propagated_mask[name]['bias']
#######

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configure_list = [{
    'quant_types': ['weight', 'output'],
    'quant_bits': {
        'weight': 8,
        'output': 8
    }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
    'op_types':['Linear']
}]

test_dataloader = get_evaluate_dataloader(token)

optimizer = torch.optim.SGD(norm_model.parameters(), lr=1e-5, momentum=0.5)
quantizer = ObserverQuantizer(norm_model, configure_list, optimizer)
norm_model.to(device)
# calibration(norm_model, device, test_dataloader)
quantizer.compress()

calibration_path = "bert_ptq_calibration.pth"
calibration_config = torch.load(calibration_path)
quantizer.load_calibration_config(calibration_config)

acc_quant = evaluate(norm_model, token)

import ipdb; ipdb.set_trace()