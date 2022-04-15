
import copy
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
import torch
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from bert_utils import *
from sparta.common.utils import export_tesa
from shape_hook import ShapeHook
device = torch.device('cpu')
config = torch.load('Coarse_bert_config')
dummy_input = torch.load('dummy_input.pth', map_location=device)
data = (dummy_input['input_ids'].to(device), dummy_input['attention_mask'].to(device), dummy_input['token_type_ids'].to(device))
norm_model = BertForSequenceClassification(config=config).to(device)
token = BertTokenizer.from_pretrained('checkpoints/finegrained/checkpoint-220000')

ms = ModelSpeedup(norm_model, data, 'checkpoints/finegrained/Bert_finegrained_mask.pth', break_points=[], confidence=32)

# get the propagated mask
propagated_mask = ms.propagate_mask()
for layer in propagated_mask:
    print('New sparsity ratio of', layer, ' :', 1-torch.sum(propagated_mask[layer]['weight'])/propagated_mask[layer]['weight'].numel())

# mask = torch.load('checkpoints/finegrained/Bert_finegrained_mask.pth', map_location='cuda')
norm_model.load_state_dict(torch.load('checkpoints/finegrained/weight.pth') )
# apply the mask
norm_model = norm_model.cuda()

task_name = "qqp"

cfg_list = [{'op_types':['Linear'], 'sparsity':0.95}]
pruner = LevelPruner(norm_model, cfg_list)
pruner.compress()
# apply the propagated mask
for name in propagated_mask:
    _, module = get_module_by_name(norm_model, name)
    for key in propagated_mask[name]:
        # print(module)
        # import ipdb; ipdb.set_trace()
        assert(hasattr(module,key + '_mask'))
        setattr(module, key+'_mask', propagated_mask[name][key].cuda())
norm_model.load_state_dict(torch.load('checkpoints/finegrained/propagated_withmask.pth'))
print('Accuracy:', evaluate(norm_model, token))
pruner._unwrap_model()
# import ipdb; ipdb.set_trace()
tmp_model = copy.deepcopy(norm_model.cpu())
export_tesa(norm_model.cpu(), data, 'artifact_bert_finegrained_onnx_with_tesa', propagated_mask)
sh = ShapeHook(tmp_model, data)
sh.export('artifact_bert_finegrained_onnx_with_tesa/shape.json')
# import ipdb; ipdb.set_trace()
# for layer in propagated_mask:
#     print('New sparsity ratio of', layer, ' :', 1-torch.sum(propagated_mask[layer]['weight'])/propagated_mask[layer]['weight'].numel())
# train_dataset = load_and_cache_examples("qqp", token, evaluate=False)
# train(train_dataset, norm_model, token, num_train_epochs=100)
# import ipdb; ipdb.set_trace()
