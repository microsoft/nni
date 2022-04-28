import torch
from pytorchyolo import models

from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.compression.pytorch.utils import not_safe_to_prune

# The Yolo can be downloaded at https://github.com/eriklindernoren/PyTorch-YOLOv3.git
prefix = '/home/user/PyTorch-YOLOv3' # replace this path with yours
# Load the YOLO model
model = models.load_model(
  "%s/config/yolov3.cfg" % prefix, 
  "%s/yolov3.weights" % prefix).cpu()
model.eval()
dummy_input = torch.rand(8, 3, 320, 320)
model(dummy_input)
# Generate the config list for pruner
# Filter the layers that may not be able to prune
not_safe = not_safe_to_prune(model, dummy_input)
cfg_list = []
for name, module in model.named_modules():
  if name in not_safe:
    continue
  if isinstance(module, torch.nn.Conv2d):
    cfg_list.append({'op_types':['Conv2d'], 'sparsity':0.6, 'op_names':[name]})
# Prune the model
pruner = L1FilterPruner(model, cfg_list)
pruner.compress()
pruner.export_model('./model', './mask')
pruner._unwrap_model()
# Speedup the model
ms = ModelSpeedup(model, dummy_input, './mask')

ms.speedup_model()
model(dummy_input)

