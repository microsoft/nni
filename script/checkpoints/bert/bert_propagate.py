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
from nni.compression.pytorch.utils import get_module_by_name
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

from ModelVisual2 import ModelVisual
from shape_hook import ShapeHook
device = torch.device('cpu')
config = torch.load('Coarse_bert_config')
dummy_input = torch.load('dummy_input.pth', map_location=device)
data = (dummy_input['input_ids'].to(device), dummy_input['attention_mask'].to(device), dummy_input['token_type_ids'].to(device))
norm_model = BertForSequenceClassification(config=config).to(device)
encoder_input = torch.load('encoder_input.pth')

# def _forward_hook(model, input, output):
#     import pdb; pdb.set_trace()

# sh = ShapeHook(norm_model, data, debug_point=['bert.encoder'])
# sh.export('bert_ori_shape.json')
# mv = ModelVisual(norm_model, data)
# mv.visualize('ori_bert_unpack_output', unpack=True)
# exit()
# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
# with torch.no_grad():
#     norm_model.bert.encoder.register_forward_hook(_forward_hook)
#     norm_model(*data)

# pruner = LevelPruner(norm_model, [{'sparsity': 0.95, 'op_types':['Linear']}])
# pruner.compress()
# pruner.export_model('tmp_weight.pth', 'tmp_mask.pth')
# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
ms = ModelSpeedup(norm_model, data, 'Bert_finegrained_mask.pth', break_points=[], confidence=32)
import ipdb; ipdb.set_trace()

propagated_mask = ms.propagate_mask()
pass
pass
exit()
import pdb; pdb.set_trace()



# head_prune_cfg = torch.load('head_prune_cfg')
# norm_model.prune_heads(head_prune_cfg)

norm_model.load_state_dict(torch.load('/data/znx/SpargenCks/bert_coarse_cks/nni_weight.pth') )
task_name = "qqp"
token = BertTokenizer.from_pretrained('/data/znx/SpargenCks/bert_coarse_cks/token_pretrain/checkpoint-220000')
# acc = evaluate(norm_model, token)

# import pdb; pdb.set_trace()
