import torch                                                                                                                                                                                                                                                                                                                                          
from nanodet.model.arch import build_model                                                                                                                                                                                                                                                                                                            
from nanodet.util import cfg, load_config                                                                                                                                                                                                                                                                  

from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner

"""
NanoDet model can be installed from https://github.com/RangiLyu/nanodet.git
"""

cfg_path = r"nanodet/config/nanodet-RepVGG-A0_416.yml"
load_config(cfg, cfg_path)

model = build_model(cfg.model).cpu()
dummy_input  = torch.rand(8, 3, 416, 416)

op_names = []
# these three conv layers are followed by reshape-like functions
# that cannot be replaced, so we skip these three conv layers,
# you can also get such layers by `not_safe_to_prune` function
excludes = ['head.gfl_cls.0', 'head.gfl_cls.1', 'head.gfl_cls.2']
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        if name not in excludes:
            op_names.append(name)

cfg_list = [{'op_types':['Conv2d'], 'sparsity':0.5, 'op_names':op_names}]
pruner = L1FilterPruner(model, cfg_list)
pruner.compress()
pruner.export_model('./model', './mask')
# need call _unwrap_model if you want run the speedup on the same model
pruner._unwrap_model()

# Speedup the nanodet
ms = ModelSpeedup(model, dummy_input, './mask')
ms.speedup_model()

model(dummy_input)