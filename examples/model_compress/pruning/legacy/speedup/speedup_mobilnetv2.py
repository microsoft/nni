import torch
from torchvision.models import mobilenet_v2
from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner


model = mobilenet_v2(pretrained=True)
dummy_input  = torch.rand(8, 3, 416, 416)

cfg_list = [{'op_types':['Conv2d'], 'sparsity':0.5}]
pruner = L1FilterPruner(model, cfg_list)
pruner.compress()
pruner.export_model('./model', './mask')
# need call _unwrap_model if you want run the speedup on the same model
pruner._unwrap_model()

# Speedup the nanodet
ms = ModelSpeedup(model, dummy_input, './mask')
ms.speedup_model()

model(dummy_input)