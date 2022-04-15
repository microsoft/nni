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
from bert_utils import *
from nni.algorithms.compression.pytorch.pruning import TransformerHeadPruner
from sparta.common.utils import export_tesa
device = torch.device('cpu')
config = torch.load('Coarse_bert_config')
dummy_input = torch.load('dummy_input.pth', map_location=device)
data = (dummy_input['input_ids'].to(device), dummy_input['attention_mask'].to(device), dummy_input['token_type_ids'].to(device))
norm_model = BertForSequenceClassification(config=config).to(device)


bert_head_size = 64
token = BertTokenizer.from_pretrained('checkpoints/finegrained/checkpoint-220000')
mask_file = 'checkpoints/finegrained/Bert_finegrained_mask.pth'


ori_mask =  torch.load(mask_file)


export_tesa(norm_model.cpu(), data, 'artifact_bert_finegrained_no_propagation_onnx_with_tesa', ori_mask)