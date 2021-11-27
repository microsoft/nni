import time
import torch

from utils import *
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from SparGen.Common.Utils import *
config = [{'sparsity':0.95, 'op_types':['Linear']}]
model = create_model('mobilenet_v1').cpu()
data = torch.rand(32,3,224,224).cpu()
pruner = LevelPruner(model, config)
pruner.compress()
model(data)
pruner._unwrap_model()

export_tesa(model, data, './mobilenet_sota_finegrained')


model = create_model('mobilenet_v1').cpu()
pruned = int(model.classifier[0].weight.size(0) * 0.6) 
model.classifier[0].weight.data[:pruned] = 0

export_tesa(model, data, 'mobilenet_sota_coarsegrained')
# import pdb; pdb.set_trace()