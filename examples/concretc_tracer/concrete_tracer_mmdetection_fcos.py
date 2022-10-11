# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    
import torch
import mmcv
from mmdet.apis import init_detector
from mmcv.parallel import collate
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from nni.common.concrete_trace_utils import concrete_trace, ConcreteTracer

folder_prefix = 'C:\\Works\\mmdetection' # replace this path with yours
# Specify the path to model config and checkpoint file
config_file = '%s/configs/fcos/fcos_center_r50_caffe_fpn_gn-head_1x_coco.py' % folder_prefix
config = mmcv.Config.fromfile(config_file)

# RoIAlign will cause different errors when use cpu. there are 4 ways to avoid it
# 1. add 'mmcv_ops.RoIAlign' to leaf_module when tracing, and set 'config_dict['use_torchvision'] = True'
# 2. add 'mmcv_ops.RoIAlign' to leaf_module when tracing, and set 'config_dict['aligned'] = False'
# 3. set 'config_dict['use_torchvision'] = True', and set 'config_dict['aligned'] = False', and
#      from torchvision.ops import roi_align as tv_roi_align
#      add 'tv_roi_align: (((torchvision.ops, 'roi_align'),), False, None),' to autowrap_leaf_function
# 4. set 'config_dict['aligned'] = False', and add 'from mmcv.ops.roi_align import roi_align as mmcv_roi_align' and
#      add 'mmcv_roi_align: ((), False, None),' to autowrap_leaf_function
RoIAlign_solution = 3
def roi_align_setter(config_dict: dict):
  if 'type' in config_dict:
    if config_dict['type'] == 'RoIAlign':
      if RoIAlign_solution in (1, 3):
        config_dict['use_torchvision'] = True
      if RoIAlign_solution in (2, 3, 4):
        config_dict['aligned'] = False
      pass
    else:
      for v in config_dict.values():
        if isinstance(v, dict):
          roi_align_setter(v)
roi_align_setter(config._cfg_dict['model'])

autowrap_leaf_function_append = dict()
if RoIAlign_solution == 3:
  from torchvision.ops import roi_align as tv_roi_align
  import torchvision
  autowrap_leaf_function_append = {tv_roi_align: (((torchvision.ops, 'roi_align'),), False, None),}
elif RoIAlign_solution == 4:
  from mmcv.ops.roi_align import roi_align as mmcv_roi_align
  autowrap_leaf_function_append = {mmcv_roi_align: ((), False, None),}
  
leaf_module_append = ()
if RoIAlign_solution in (1, 2):
  from mmcv import ops as mmcv_ops
  leaf_module_append = (mmcv_ops.RoIAlign,)

model = init_detector(config, device='cpu')

img = '%s/tests/data/color.jpg' % folder_prefix

def check_equal(a, b):
  if type(a) != type(b):
    return False
  if isinstance(a, (list, tuple, set)):
    if len(a) != len(b):
      return False
    for sub_a, sub_b in zip(a, b):
      if not check_equal(sub_a, sub_b):
        return False
    return True
  elif isinstance(a, dict):
    keys_a, kes_b = set(a.keys()), set(b.keys())
    if keys_a != kes_b:
      return False
    for key in keys_a:
      if not check_equal(a[key], b[key]):
        return False
    return True
  elif isinstance(a, torch.Tensor):
    return torch.equal(a, b)
  else:
    return a == b

with torch.no_grad():
  cfg = model.cfg
  cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
  test_pipeline = Compose(cfg.data.test.pipeline)
  img_data = test_pipeline(dict(img_info=dict(filename=img), img_prefix=None))['img']
  img_tensor = collate(img_data, 1).data[0]
  input_like = torch.rand_like(img_tensor)
  
  torch.manual_seed(100)
  out_orig = model.forward_dummy(img_tensor)

  torch.manual_seed(100)
  traced_model = concrete_trace(model, {'img': img_tensor}, False, forwrad_function_name='forward_dummy', autowrap_leaf_function = {
    **ConcreteTracer.default_autowrap_leaf_function,
    **autowrap_leaf_function_append,
  }, leaf_module = (
    *leaf_module_append,
  ))
  # print('traced code:\n', traced_model.code)

  torch.manual_seed(100)
  out_orig_traced = traced_model(img_tensor)
  assert check_equal(out_orig, out_orig_traced)
  
  torch.manual_seed(100)
  out_like = model.forward_dummy(input_like)
  torch.manual_seed(100)
  out_like_traced = traced_model(input_like)
  assert check_equal(out_like, out_like_traced)
  print('done')