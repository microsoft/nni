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
config_file = '%s/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' % folder_prefix
config = mmcv.Config.fromfile(config_file)

def roi_align_setter(config_dict: dict):
  if 'type' in config_dict:
    if config_dict['type'] == 'RoIAlign':
      # config_dict['aligned'] = False
      config_dict['use_torchvision'] = True
      pass
    else:
      for v in config_dict.values():
        if isinstance(v, dict):
          roi_align_setter(v)
roi_align_setter(config._cfg_dict['model'])
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
  
from torchvision.ops import roi_align as torchvision_ops_roi_align_roi_align
def forward(self, img_1):
  backbone_conv1 = self.backbone.conv1(img_1)
  backbone_bn1 = self.backbone.bn1(backbone_conv1);  backbone_conv1 = None
  backbone_relu = self.backbone.relu(backbone_bn1);  backbone_bn1 = None
  backbone_maxpool = self.backbone.maxpool(backbone_relu);  backbone_relu = None
  backbone_layer1_0_conv1 = self.backbone.layer1[0].conv1(backbone_maxpool)
  backbone_layer1_0_bn1 = self.backbone.layer1[0].bn1(backbone_layer1_0_conv1);  backbone_layer1_0_conv1 = None
  backbone_layer1_0_relu = self.backbone.layer1[0].relu(backbone_layer1_0_bn1);  backbone_layer1_0_bn1 = None
  backbone_layer1_0_conv2 = self.backbone.layer1[0].conv2(backbone_layer1_0_relu);  backbone_layer1_0_relu = None
  backbone_layer1_0_bn2 = self.backbone.layer1[0].bn2(backbone_layer1_0_conv2);  backbone_layer1_0_conv2 = None
  backbone_layer1_0_relu_1 = self.backbone.layer1[0].relu(backbone_layer1_0_bn2);  backbone_layer1_0_bn2 = None
  backbone_layer1_0_conv3 = self.backbone.layer1[0].conv3(backbone_layer1_0_relu_1);  backbone_layer1_0_relu_1 = None
  backbone_layer1_0_bn3 = self.backbone.layer1[0].bn3(backbone_layer1_0_conv3);  backbone_layer1_0_conv3 = None
  backbone_layer1_0_downsample_0 = self.backbone.layer1[0].downsample[0](backbone_maxpool);  backbone_maxpool = None
  backbone_layer1_0_downsample_1 = self.backbone.layer1[0].downsample[1](backbone_layer1_0_downsample_0);  backbone_layer1_0_downsample_0 = None
  add = backbone_layer1_0_bn3 + backbone_layer1_0_downsample_1;  backbone_layer1_0_bn3 = backbone_layer1_0_downsample_1 = None
  backbone_layer1_0_relu_2 = self.backbone.layer1[0].relu(add);  add = None
  backbone_layer1_1_conv1 = self.backbone.layer1[1].conv1(backbone_layer1_0_relu_2)
  backbone_layer1_1_bn1 = self.backbone.layer1[1].bn1(backbone_layer1_1_conv1);  backbone_layer1_1_conv1 = None
  backbone_layer1_1_relu = self.backbone.layer1[1].relu(backbone_layer1_1_bn1);  backbone_layer1_1_bn1 = None
  backbone_layer1_1_conv2 = self.backbone.layer1[1].conv2(backbone_layer1_1_relu);  backbone_layer1_1_relu = None
  backbone_layer1_1_bn2 = self.backbone.layer1[1].bn2(backbone_layer1_1_conv2);  backbone_layer1_1_conv2 = None
  backbone_layer1_1_relu_1 = self.backbone.layer1[1].relu(backbone_layer1_1_bn2);  backbone_layer1_1_bn2 = None
  backbone_layer1_1_conv3 = self.backbone.layer1[1].conv3(backbone_layer1_1_relu_1);  backbone_layer1_1_relu_1 = None
  backbone_layer1_1_bn3 = self.backbone.layer1[1].bn3(backbone_layer1_1_conv3);  backbone_layer1_1_conv3 = None
  add_1 = backbone_layer1_1_bn3 + backbone_layer1_0_relu_2;  backbone_layer1_1_bn3 = backbone_layer1_0_relu_2 = None
  backbone_layer1_1_relu_2 = self.backbone.layer1[1].relu(add_1);  add_1 = None
  backbone_layer1_2_conv1 = self.backbone.layer1[2].conv1(backbone_layer1_1_relu_2)
  backbone_layer1_2_bn1 = self.backbone.layer1[2].bn1(backbone_layer1_2_conv1);  backbone_layer1_2_conv1 = None
  backbone_layer1_2_relu = self.backbone.layer1[2].relu(backbone_layer1_2_bn1);  backbone_layer1_2_bn1 = None
  backbone_layer1_2_conv2 = self.backbone.layer1[2].conv2(backbone_layer1_2_relu);  backbone_layer1_2_relu = None
  backbone_layer1_2_bn2 = self.backbone.layer1[2].bn2(backbone_layer1_2_conv2);  backbone_layer1_2_conv2 = None
  backbone_layer1_2_relu_1 = self.backbone.layer1[2].relu(backbone_layer1_2_bn2);  backbone_layer1_2_bn2 = None
  backbone_layer1_2_conv3 = self.backbone.layer1[2].conv3(backbone_layer1_2_relu_1);  backbone_layer1_2_relu_1 = None
  backbone_layer1_2_bn3 = self.backbone.layer1[2].bn3(backbone_layer1_2_conv3);  backbone_layer1_2_conv3 = None
  add_2 = backbone_layer1_2_bn3 + backbone_layer1_1_relu_2;  backbone_layer1_2_bn3 = backbone_layer1_1_relu_2 = None
  backbone_layer1_2_relu_2 = self.backbone.layer1[2].relu(add_2);  add_2 = None
  backbone_layer2_0_conv1 = self.backbone.layer2[0].conv1(backbone_layer1_2_relu_2)
  backbone_layer2_0_bn1 = self.backbone.layer2[0].bn1(backbone_layer2_0_conv1);  backbone_layer2_0_conv1 = None
  backbone_layer2_0_relu = self.backbone.layer2[0].relu(backbone_layer2_0_bn1);  backbone_layer2_0_bn1 = None
  backbone_layer2_0_conv2 = self.backbone.layer2[0].conv2(backbone_layer2_0_relu);  backbone_layer2_0_relu = None
  backbone_layer2_0_bn2 = self.backbone.layer2[0].bn2(backbone_layer2_0_conv2);  backbone_layer2_0_conv2 = None
  backbone_layer2_0_relu_1 = self.backbone.layer2[0].relu(backbone_layer2_0_bn2);  backbone_layer2_0_bn2 = None
  backbone_layer2_0_conv3 = self.backbone.layer2[0].conv3(backbone_layer2_0_relu_1);  backbone_layer2_0_relu_1 = None
  backbone_layer2_0_bn3 = self.backbone.layer2[0].bn3(backbone_layer2_0_conv3);  backbone_layer2_0_conv3 = None
  backbone_layer2_0_downsample_0 = self.backbone.layer2[0].downsample[0](backbone_layer1_2_relu_2)
  backbone_layer2_0_downsample_1 = self.backbone.layer2[0].downsample[1](backbone_layer2_0_downsample_0);  backbone_layer2_0_downsample_0 = None
  add_3 = backbone_layer2_0_bn3 + backbone_layer2_0_downsample_1;  backbone_layer2_0_bn3 = backbone_layer2_0_downsample_1 = None
  backbone_layer2_0_relu_2 = self.backbone.layer2[0].relu(add_3);  add_3 = None
  backbone_layer2_1_conv1 = self.backbone.layer2[1].conv1(backbone_layer2_0_relu_2)
  backbone_layer2_1_bn1 = self.backbone.layer2[1].bn1(backbone_layer2_1_conv1);  backbone_layer2_1_conv1 = None
  backbone_layer2_1_relu = self.backbone.layer2[1].relu(backbone_layer2_1_bn1);  backbone_layer2_1_bn1 = None
  backbone_layer2_1_conv2 = self.backbone.layer2[1].conv2(backbone_layer2_1_relu);  backbone_layer2_1_relu = None
  backbone_layer2_1_bn2 = self.backbone.layer2[1].bn2(backbone_layer2_1_conv2);  backbone_layer2_1_conv2 = None
  backbone_layer2_1_relu_1 = self.backbone.layer2[1].relu(backbone_layer2_1_bn2);  backbone_layer2_1_bn2 = None
  backbone_layer2_1_conv3 = self.backbone.layer2[1].conv3(backbone_layer2_1_relu_1);  backbone_layer2_1_relu_1 = None
  backbone_layer2_1_bn3 = self.backbone.layer2[1].bn3(backbone_layer2_1_conv3);  backbone_layer2_1_conv3 = None
  add_4 = backbone_layer2_1_bn3 + backbone_layer2_0_relu_2;  backbone_layer2_1_bn3 = backbone_layer2_0_relu_2 = None
  backbone_layer2_1_relu_2 = self.backbone.layer2[1].relu(add_4);  add_4 = None
  backbone_layer2_2_conv1 = self.backbone.layer2[2].conv1(backbone_layer2_1_relu_2)
  backbone_layer2_2_bn1 = self.backbone.layer2[2].bn1(backbone_layer2_2_conv1);  backbone_layer2_2_conv1 = None
  backbone_layer2_2_relu = self.backbone.layer2[2].relu(backbone_layer2_2_bn1);  backbone_layer2_2_bn1 = None
  backbone_layer2_2_conv2 = self.backbone.layer2[2].conv2(backbone_layer2_2_relu);  backbone_layer2_2_relu = None
  backbone_layer2_2_bn2 = self.backbone.layer2[2].bn2(backbone_layer2_2_conv2);  backbone_layer2_2_conv2 = None
  backbone_layer2_2_relu_1 = self.backbone.layer2[2].relu(backbone_layer2_2_bn2);  backbone_layer2_2_bn2 = None
  backbone_layer2_2_conv3 = self.backbone.layer2[2].conv3(backbone_layer2_2_relu_1);  backbone_layer2_2_relu_1 = None
  backbone_layer2_2_bn3 = self.backbone.layer2[2].bn3(backbone_layer2_2_conv3);  backbone_layer2_2_conv3 = None
  add_5 = backbone_layer2_2_bn3 + backbone_layer2_1_relu_2;  backbone_layer2_2_bn3 = backbone_layer2_1_relu_2 = None
  backbone_layer2_2_relu_2 = self.backbone.layer2[2].relu(add_5);  add_5 = None
  backbone_layer2_3_conv1 = self.backbone.layer2[3].conv1(backbone_layer2_2_relu_2)
  backbone_layer2_3_bn1 = self.backbone.layer2[3].bn1(backbone_layer2_3_conv1);  backbone_layer2_3_conv1 = None
  backbone_layer2_3_relu = self.backbone.layer2[3].relu(backbone_layer2_3_bn1);  backbone_layer2_3_bn1 = None
  backbone_layer2_3_conv2 = self.backbone.layer2[3].conv2(backbone_layer2_3_relu);  backbone_layer2_3_relu = None
  backbone_layer2_3_bn2 = self.backbone.layer2[3].bn2(backbone_layer2_3_conv2);  backbone_layer2_3_conv2 = None
  backbone_layer2_3_relu_1 = self.backbone.layer2[3].relu(backbone_layer2_3_bn2);  backbone_layer2_3_bn2 = None
  backbone_layer2_3_conv3 = self.backbone.layer2[3].conv3(backbone_layer2_3_relu_1);  backbone_layer2_3_relu_1 = None
  backbone_layer2_3_bn3 = self.backbone.layer2[3].bn3(backbone_layer2_3_conv3);  backbone_layer2_3_conv3 = None
  add_6 = backbone_layer2_3_bn3 + backbone_layer2_2_relu_2;  backbone_layer2_3_bn3 = backbone_layer2_2_relu_2 = None
  backbone_layer2_3_relu_2 = self.backbone.layer2[3].relu(add_6);  add_6 = None
  backbone_layer3_0_conv1 = self.backbone.layer3[0].conv1(backbone_layer2_3_relu_2)
  backbone_layer3_0_bn1 = self.backbone.layer3[0].bn1(backbone_layer3_0_conv1);  backbone_layer3_0_conv1 = None
  backbone_layer3_0_relu = self.backbone.layer3[0].relu(backbone_layer3_0_bn1);  backbone_layer3_0_bn1 = None
  backbone_layer3_0_conv2 = self.backbone.layer3[0].conv2(backbone_layer3_0_relu);  backbone_layer3_0_relu = None
  backbone_layer3_0_bn2 = self.backbone.layer3[0].bn2(backbone_layer3_0_conv2);  backbone_layer3_0_conv2 = None
  backbone_layer3_0_relu_1 = self.backbone.layer3[0].relu(backbone_layer3_0_bn2);  backbone_layer3_0_bn2 = None
  backbone_layer3_0_conv3 = self.backbone.layer3[0].conv3(backbone_layer3_0_relu_1);  backbone_layer3_0_relu_1 = None
  backbone_layer3_0_bn3 = self.backbone.layer3[0].bn3(backbone_layer3_0_conv3);  backbone_layer3_0_conv3 = None
  backbone_layer3_0_downsample_0 = self.backbone.layer3[0].downsample[0](backbone_layer2_3_relu_2)
  backbone_layer3_0_downsample_1 = self.backbone.layer3[0].downsample[1](backbone_layer3_0_downsample_0);  backbone_layer3_0_downsample_0 = None
  add_7 = backbone_layer3_0_bn3 + backbone_layer3_0_downsample_1;  backbone_layer3_0_bn3 = backbone_layer3_0_downsample_1 = None
  backbone_layer3_0_relu_2 = self.backbone.layer3[0].relu(add_7);  add_7 = None
  backbone_layer3_1_conv1 = self.backbone.layer3[1].conv1(backbone_layer3_0_relu_2)
  backbone_layer3_1_bn1 = self.backbone.layer3[1].bn1(backbone_layer3_1_conv1);  backbone_layer3_1_conv1 = None
  backbone_layer3_1_relu = self.backbone.layer3[1].relu(backbone_layer3_1_bn1);  backbone_layer3_1_bn1 = None
  backbone_layer3_1_conv2 = self.backbone.layer3[1].conv2(backbone_layer3_1_relu);  backbone_layer3_1_relu = None
  backbone_layer3_1_bn2 = self.backbone.layer3[1].bn2(backbone_layer3_1_conv2);  backbone_layer3_1_conv2 = None
  backbone_layer3_1_relu_1 = self.backbone.layer3[1].relu(backbone_layer3_1_bn2);  backbone_layer3_1_bn2 = None
  backbone_layer3_1_conv3 = self.backbone.layer3[1].conv3(backbone_layer3_1_relu_1);  backbone_layer3_1_relu_1 = None
  backbone_layer3_1_bn3 = self.backbone.layer3[1].bn3(backbone_layer3_1_conv3);  backbone_layer3_1_conv3 = None
  add_8 = backbone_layer3_1_bn3 + backbone_layer3_0_relu_2;  backbone_layer3_1_bn3 = backbone_layer3_0_relu_2 = None
  backbone_layer3_1_relu_2 = self.backbone.layer3[1].relu(add_8);  add_8 = None
  backbone_layer3_2_conv1 = self.backbone.layer3[2].conv1(backbone_layer3_1_relu_2)
  backbone_layer3_2_bn1 = self.backbone.layer3[2].bn1(backbone_layer3_2_conv1);  backbone_layer3_2_conv1 = None
  backbone_layer3_2_relu = self.backbone.layer3[2].relu(backbone_layer3_2_bn1);  backbone_layer3_2_bn1 = None
  backbone_layer3_2_conv2 = self.backbone.layer3[2].conv2(backbone_layer3_2_relu);  backbone_layer3_2_relu = None
  backbone_layer3_2_bn2 = self.backbone.layer3[2].bn2(backbone_layer3_2_conv2);  backbone_layer3_2_conv2 = None
  backbone_layer3_2_relu_1 = self.backbone.layer3[2].relu(backbone_layer3_2_bn2);  backbone_layer3_2_bn2 = None
  backbone_layer3_2_conv3 = self.backbone.layer3[2].conv3(backbone_layer3_2_relu_1);  backbone_layer3_2_relu_1 = None
  backbone_layer3_2_bn3 = self.backbone.layer3[2].bn3(backbone_layer3_2_conv3);  backbone_layer3_2_conv3 = None
  add_9 = backbone_layer3_2_bn3 + backbone_layer3_1_relu_2;  backbone_layer3_2_bn3 = backbone_layer3_1_relu_2 = None
  backbone_layer3_2_relu_2 = self.backbone.layer3[2].relu(add_9);  add_9 = None
  backbone_layer3_3_conv1 = self.backbone.layer3[3].conv1(backbone_layer3_2_relu_2)
  backbone_layer3_3_bn1 = self.backbone.layer3[3].bn1(backbone_layer3_3_conv1);  backbone_layer3_3_conv1 = None
  backbone_layer3_3_relu = self.backbone.layer3[3].relu(backbone_layer3_3_bn1);  backbone_layer3_3_bn1 = None
  backbone_layer3_3_conv2 = self.backbone.layer3[3].conv2(backbone_layer3_3_relu);  backbone_layer3_3_relu = None
  backbone_layer3_3_bn2 = self.backbone.layer3[3].bn2(backbone_layer3_3_conv2);  backbone_layer3_3_conv2 = None
  backbone_layer3_3_relu_1 = self.backbone.layer3[3].relu(backbone_layer3_3_bn2);  backbone_layer3_3_bn2 = None
  backbone_layer3_3_conv3 = self.backbone.layer3[3].conv3(backbone_layer3_3_relu_1);  backbone_layer3_3_relu_1 = None
  backbone_layer3_3_bn3 = self.backbone.layer3[3].bn3(backbone_layer3_3_conv3);  backbone_layer3_3_conv3 = None
  add_10 = backbone_layer3_3_bn3 + backbone_layer3_2_relu_2;  backbone_layer3_3_bn3 = backbone_layer3_2_relu_2 = None
  backbone_layer3_3_relu_2 = self.backbone.layer3[3].relu(add_10);  add_10 = None
  backbone_layer3_4_conv1 = self.backbone.layer3[4].conv1(backbone_layer3_3_relu_2)
  backbone_layer3_4_bn1 = self.backbone.layer3[4].bn1(backbone_layer3_4_conv1);  backbone_layer3_4_conv1 = None
  backbone_layer3_4_relu = self.backbone.layer3[4].relu(backbone_layer3_4_bn1);  backbone_layer3_4_bn1 = None
  backbone_layer3_4_conv2 = self.backbone.layer3[4].conv2(backbone_layer3_4_relu);  backbone_layer3_4_relu = None
  backbone_layer3_4_bn2 = self.backbone.layer3[4].bn2(backbone_layer3_4_conv2);  backbone_layer3_4_conv2 = None
  backbone_layer3_4_relu_1 = self.backbone.layer3[4].relu(backbone_layer3_4_bn2);  backbone_layer3_4_bn2 = None
  backbone_layer3_4_conv3 = self.backbone.layer3[4].conv3(backbone_layer3_4_relu_1);  backbone_layer3_4_relu_1 = None
  backbone_layer3_4_bn3 = self.backbone.layer3[4].bn3(backbone_layer3_4_conv3);  backbone_layer3_4_conv3 = None
  add_11 = backbone_layer3_4_bn3 + backbone_layer3_3_relu_2;  backbone_layer3_4_bn3 = backbone_layer3_3_relu_2 = None
  backbone_layer3_4_relu_2 = self.backbone.layer3[4].relu(add_11);  add_11 = None
  backbone_layer3_5_conv1 = self.backbone.layer3[5].conv1(backbone_layer3_4_relu_2)
  backbone_layer3_5_bn1 = self.backbone.layer3[5].bn1(backbone_layer3_5_conv1);  backbone_layer3_5_conv1 = None
  backbone_layer3_5_relu = self.backbone.layer3[5].relu(backbone_layer3_5_bn1);  backbone_layer3_5_bn1 = None
  backbone_layer3_5_conv2 = self.backbone.layer3[5].conv2(backbone_layer3_5_relu);  backbone_layer3_5_relu = None
  backbone_layer3_5_bn2 = self.backbone.layer3[5].bn2(backbone_layer3_5_conv2);  backbone_layer3_5_conv2 = None
  backbone_layer3_5_relu_1 = self.backbone.layer3[5].relu(backbone_layer3_5_bn2);  backbone_layer3_5_bn2 = None
  backbone_layer3_5_conv3 = self.backbone.layer3[5].conv3(backbone_layer3_5_relu_1);  backbone_layer3_5_relu_1 = None
  backbone_layer3_5_bn3 = self.backbone.layer3[5].bn3(backbone_layer3_5_conv3);  backbone_layer3_5_conv3 = None
  add_12 = backbone_layer3_5_bn3 + backbone_layer3_4_relu_2;  backbone_layer3_5_bn3 = backbone_layer3_4_relu_2 = None
  backbone_layer3_5_relu_2 = self.backbone.layer3[5].relu(add_12);  add_12 = None
  backbone_layer4_0_conv1 = self.backbone.layer4[0].conv1(backbone_layer3_5_relu_2)
  backbone_layer4_0_bn1 = self.backbone.layer4[0].bn1(backbone_layer4_0_conv1);  backbone_layer4_0_conv1 = None
  backbone_layer4_0_relu = self.backbone.layer4[0].relu(backbone_layer4_0_bn1);  backbone_layer4_0_bn1 = None
  backbone_layer4_0_conv2 = self.backbone.layer4[0].conv2(backbone_layer4_0_relu);  backbone_layer4_0_relu = None
  backbone_layer4_0_bn2 = self.backbone.layer4[0].bn2(backbone_layer4_0_conv2);  backbone_layer4_0_conv2 = None
  backbone_layer4_0_relu_1 = self.backbone.layer4[0].relu(backbone_layer4_0_bn2);  backbone_layer4_0_bn2 = None
  backbone_layer4_0_conv3 = self.backbone.layer4[0].conv3(backbone_layer4_0_relu_1);  backbone_layer4_0_relu_1 = None
  backbone_layer4_0_bn3 = self.backbone.layer4[0].bn3(backbone_layer4_0_conv3);  backbone_layer4_0_conv3 = None
  backbone_layer4_0_downsample_0 = self.backbone.layer4[0].downsample[0](backbone_layer3_5_relu_2)
  backbone_layer4_0_downsample_1 = self.backbone.layer4[0].downsample[1](backbone_layer4_0_downsample_0);  backbone_layer4_0_downsample_0 = None
  add_13 = backbone_layer4_0_bn3 + backbone_layer4_0_downsample_1;  backbone_layer4_0_bn3 = backbone_layer4_0_downsample_1 = None
  backbone_layer4_0_relu_2 = self.backbone.layer4[0].relu(add_13);  add_13 = None
  backbone_layer4_1_conv1 = self.backbone.layer4[1].conv1(backbone_layer4_0_relu_2)
  backbone_layer4_1_bn1 = self.backbone.layer4[1].bn1(backbone_layer4_1_conv1);  backbone_layer4_1_conv1 = None
  backbone_layer4_1_relu = self.backbone.layer4[1].relu(backbone_layer4_1_bn1);  backbone_layer4_1_bn1 = None
  backbone_layer4_1_conv2 = self.backbone.layer4[1].conv2(backbone_layer4_1_relu);  backbone_layer4_1_relu = None
  backbone_layer4_1_bn2 = self.backbone.layer4[1].bn2(backbone_layer4_1_conv2);  backbone_layer4_1_conv2 = None
  backbone_layer4_1_relu_1 = self.backbone.layer4[1].relu(backbone_layer4_1_bn2);  backbone_layer4_1_bn2 = None
  backbone_layer4_1_conv3 = self.backbone.layer4[1].conv3(backbone_layer4_1_relu_1);  backbone_layer4_1_relu_1 = None
  backbone_layer4_1_bn3 = self.backbone.layer4[1].bn3(backbone_layer4_1_conv3);  backbone_layer4_1_conv3 = None
  add_14 = backbone_layer4_1_bn3 + backbone_layer4_0_relu_2;  backbone_layer4_1_bn3 = backbone_layer4_0_relu_2 = None
  backbone_layer4_1_relu_2 = self.backbone.layer4[1].relu(add_14);  add_14 = None
  backbone_layer4_2_conv1 = self.backbone.layer4[2].conv1(backbone_layer4_1_relu_2)
  backbone_layer4_2_bn1 = self.backbone.layer4[2].bn1(backbone_layer4_2_conv1);  backbone_layer4_2_conv1 = None
  backbone_layer4_2_relu = self.backbone.layer4[2].relu(backbone_layer4_2_bn1);  backbone_layer4_2_bn1 = None
  backbone_layer4_2_conv2 = self.backbone.layer4[2].conv2(backbone_layer4_2_relu);  backbone_layer4_2_relu = None
  backbone_layer4_2_bn2 = self.backbone.layer4[2].bn2(backbone_layer4_2_conv2);  backbone_layer4_2_conv2 = None
  backbone_layer4_2_relu_1 = self.backbone.layer4[2].relu(backbone_layer4_2_bn2);  backbone_layer4_2_bn2 = None
  backbone_layer4_2_conv3 = self.backbone.layer4[2].conv3(backbone_layer4_2_relu_1);  backbone_layer4_2_relu_1 = None
  backbone_layer4_2_bn3 = self.backbone.layer4[2].bn3(backbone_layer4_2_conv3);  backbone_layer4_2_conv3 = None
  add_15 = backbone_layer4_2_bn3 + backbone_layer4_1_relu_2;  backbone_layer4_2_bn3 = backbone_layer4_1_relu_2 = None
  backbone_layer4_2_relu_2 = self.backbone.layer4[2].relu(add_15);  add_15 = None
  tuple_1 = tuple([backbone_layer1_2_relu_2, backbone_layer2_3_relu_2, backbone_layer3_5_relu_2, backbone_layer4_2_relu_2]);  backbone_layer1_2_relu_2 = backbone_layer2_3_relu_2 = backbone_layer3_5_relu_2 = backbone_layer4_2_relu_2 = None
  len_1 = len(tuple_1)
  eq = len_1 == 4;  len_1 = None
  getitem = tuple_1[0]
  neck_lateral_convs_0_conv = self.neck.lateral_convs[0].conv(getitem);  getitem = None
  getitem_1 = tuple_1[1]
  neck_lateral_convs_1_conv = self.neck.lateral_convs[1].conv(getitem_1);  getitem_1 = None
  getitem_2 = tuple_1[2]
  neck_lateral_convs_2_conv = self.neck.lateral_convs[2].conv(getitem_2);  getitem_2 = None
  getitem_3 = tuple_1[3];  tuple_1 = None
  neck_lateral_convs_3_conv = self.neck.lateral_convs[3].conv(getitem_3);  getitem_3 = None
  getattr_1 = neck_lateral_convs_2_conv.shape
  getitem_4 = getattr_1[slice(2, None, None)];  getattr_1 = None
  interpolate = torch.nn.functional.interpolate(neck_lateral_convs_3_conv, size = getitem_4, mode = 'nearest');  getitem_4 = None
  add_16 = neck_lateral_convs_2_conv + interpolate;  neck_lateral_convs_2_conv = interpolate = None
  getattr_2 = neck_lateral_convs_1_conv.shape
  getitem_5 = getattr_2[slice(2, None, None)];  getattr_2 = None
  interpolate_1 = torch.nn.functional.interpolate(add_16, size = getitem_5, mode = 'nearest');  getitem_5 = None
  add_17 = neck_lateral_convs_1_conv + interpolate_1;  neck_lateral_convs_1_conv = interpolate_1 = None
  getattr_3 = neck_lateral_convs_0_conv.shape
  getitem_6 = getattr_3[slice(2, None, None)];  getattr_3 = None
  interpolate_2 = torch.nn.functional.interpolate(add_17, size = getitem_6, mode = 'nearest');  getitem_6 = None
  add_18 = neck_lateral_convs_0_conv + interpolate_2;  neck_lateral_convs_0_conv = interpolate_2 = None
  neck_fpn_convs_0_conv = self.neck.fpn_convs[0].conv(add_18);  add_18 = None
  neck_fpn_convs_1_conv = self.neck.fpn_convs[1].conv(add_17);  add_17 = None
  neck_fpn_convs_2_conv = self.neck.fpn_convs[2].conv(add_16);  add_16 = None
  neck_fpn_convs_3_conv = self.neck.fpn_convs[3].conv(neck_lateral_convs_3_conv);  neck_lateral_convs_3_conv = None
  max_pool2d = torch.nn.functional.max_pool2d(neck_fpn_convs_3_conv, 1, stride = 2)
  tuple_2 = tuple([neck_fpn_convs_0_conv, neck_fpn_convs_1_conv, neck_fpn_convs_2_conv, neck_fpn_convs_3_conv, max_pool2d]);  neck_fpn_convs_0_conv = neck_fpn_convs_1_conv = neck_fpn_convs_2_conv = neck_fpn_convs_3_conv = max_pool2d = None
  getitem_7 = tuple_2[0]
  rpn_head_rpn_conv = self.rpn_head.rpn_conv(getitem_7);  getitem_7 = None
  relu = torch.nn.functional.relu(rpn_head_rpn_conv, inplace = False);  rpn_head_rpn_conv = None
  rpn_head_rpn_cls = self.rpn_head.rpn_cls(relu)
  rpn_head_rpn_reg = self.rpn_head.rpn_reg(relu);  relu = None
  getitem_8 = tuple_2[1]
  rpn_head_rpn_conv_1 = self.rpn_head.rpn_conv(getitem_8);  getitem_8 = None
  relu_1 = torch.nn.functional.relu(rpn_head_rpn_conv_1, inplace = False);  rpn_head_rpn_conv_1 = None
  rpn_head_rpn_cls_1 = self.rpn_head.rpn_cls(relu_1)
  rpn_head_rpn_reg_1 = self.rpn_head.rpn_reg(relu_1);  relu_1 = None
  getitem_9 = tuple_2[2]
  rpn_head_rpn_conv_2 = self.rpn_head.rpn_conv(getitem_9);  getitem_9 = None
  relu_2 = torch.nn.functional.relu(rpn_head_rpn_conv_2, inplace = False);  rpn_head_rpn_conv_2 = None
  rpn_head_rpn_cls_2 = self.rpn_head.rpn_cls(relu_2)
  rpn_head_rpn_reg_2 = self.rpn_head.rpn_reg(relu_2);  relu_2 = None
  getitem_10 = tuple_2[3]
  rpn_head_rpn_conv_3 = self.rpn_head.rpn_conv(getitem_10);  getitem_10 = None
  relu_3 = torch.nn.functional.relu(rpn_head_rpn_conv_3, inplace = False);  rpn_head_rpn_conv_3 = None
  rpn_head_rpn_cls_3 = self.rpn_head.rpn_cls(relu_3)
  rpn_head_rpn_reg_3 = self.rpn_head.rpn_reg(relu_3);  relu_3 = None
  getitem_11 = tuple_2[4]
  rpn_head_rpn_conv_4 = self.rpn_head.rpn_conv(getitem_11);  getitem_11 = None
  relu_4 = torch.nn.functional.relu(rpn_head_rpn_conv_4, inplace = False);  rpn_head_rpn_conv_4 = None
  rpn_head_rpn_cls_4 = self.rpn_head.rpn_cls(relu_4)
  rpn_head_rpn_reg_4 = self.rpn_head.rpn_reg(relu_4);  relu_4 = None
  tuple_3 = tuple([(rpn_head_rpn_cls, rpn_head_rpn_reg), (rpn_head_rpn_cls_1, rpn_head_rpn_reg_1), (rpn_head_rpn_cls_2, rpn_head_rpn_reg_2), (rpn_head_rpn_cls_3, rpn_head_rpn_reg_3), (rpn_head_rpn_cls_4, rpn_head_rpn_reg_4)]);  rpn_head_rpn_cls = rpn_head_rpn_reg = rpn_head_rpn_cls_1 = rpn_head_rpn_reg_1 = rpn_head_rpn_cls_2 = rpn_head_rpn_reg_2 = rpn_head_rpn_cls_3 = rpn_head_rpn_reg_3 = rpn_head_rpn_cls_4 = rpn_head_rpn_reg_4 = None
  iter_1 = iter(tuple_3)
  len_2 = len(tuple_3);  tuple_3 = None
  next_1 = next(iter_1)
  next_2 = next(iter_1)
  next_3 = next(iter_1)
  next_4 = next(iter_1)
  next_5 = next(iter_1);  iter_1 = None
  list_1 = list([next_1, next_2, next_3, next_4, next_5]);  next_1 = next_2 = next_3 = next_4 = next_5 = None
  getitem_12 = list_1[0]
  getitem_13 = list_1[1]
  getitem_14 = list_1[2]
  getitem_15 = list_1[3]
  getitem_16 = list_1[4];  list_1 = None
  zip_1 = zip(getitem_12, getitem_13, getitem_14, getitem_15, getitem_16);  getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = None
  iter_2 = iter(zip_1);  zip_1 = None
  next_6 = next(iter_2)
  next_7 = next(iter_2);  iter_2 = None
  tuple_4 = tuple((next_6, next_7));  next_6 = next_7 = None
  getitem_17 = tuple_4[0]
  iter_3 = iter(getitem_17)
  len_3 = len(getitem_17);  getitem_17 = None
  next_8 = next(iter_3)
  next_9 = next(iter_3)
  next_10 = next(iter_3)
  next_11 = next(iter_3)
  next_12 = next(iter_3);  iter_3 = None
  list_2 = list([next_8, next_9, next_10, next_11, next_12]);  next_8 = next_9 = next_10 = next_11 = next_12 = None
  getitem_18 = tuple_4[1];  tuple_4 = None
  iter_4 = iter(getitem_18)
  len_4 = len(getitem_18);  getitem_18 = None
  next_13 = next(iter_4)
  next_14 = next(iter_4)
  next_15 = next(iter_4)
  next_16 = next(iter_4)
  next_17 = next(iter_4);  iter_4 = None
  list_3 = list([next_13, next_14, next_15, next_16, next_17]);  next_13 = next_14 = next_15 = next_16 = next_17 = None
  tuple_5 = tuple([list_2, list_3]);  list_2 = list_3 = None
  iter_5 = iter(tuple_5)
  len_5 = len(tuple_5);  tuple_5 = None
  next_18 = next(iter_5)
  next_19 = next(iter_5);  iter_5 = None
  tuple_6 = tuple((next_18, next_19));  next_18 = next_19 = None
  randn = torch.randn(1000, 4)
  getattr_4 = img_1.device;  img_1 = None
  to = randn.to(getattr_4);  randn = getattr_4 = None
  size = to.size(0)
  gt = size > 0;  size = None
  size_1 = to.size(0)
  new_full = to.new_full((size_1, 1), 0);  size_1 = None
  getitem_19 = to[(slice(None, None, None), slice(None, 4, None))];  to = None
  cat = torch.cat([new_full, getitem_19], dim = -1);  new_full = getitem_19 = None
  cat_1 = torch.cat([cat], 0);  cat = None
  getitem_20 = tuple_2[slice(None, 4, None)];  tuple_2 = None
  len_6 = len(getitem_20)
  getitem_21 = getitem_20[0]
  size_2 = cat_1.size(0)
  new_zeros = getitem_21.new_zeros(size_2, 256, 7, 7);  getitem_21 = size_2 = None
  eq_1 = len_6 == 1
  getitem_22 = cat_1[(slice(None, None, None), 3)]
  getitem_23 = cat_1[(slice(None, None, None), 1)]
  sub = getitem_22 - getitem_23;  getitem_22 = getitem_23 = None
  getitem_24 = cat_1[(slice(None, None, None), 4)]
  getitem_25 = cat_1[(slice(None, None, None), 2)]
  sub_1 = getitem_24 - getitem_25;  getitem_24 = getitem_25 = None
  mul = sub * sub_1;  sub = sub_1 = None
  sqrt = torch.sqrt(mul);  mul = None
  truediv = sqrt / 56;  sqrt = None
  add_19 = truediv + 1e-06;  truediv = None
  log2 = torch.log2(add_19);  add_19 = None
  floor = torch.floor(log2);  log2 = None
  sub_2 = len_6 - 1
  clamp = floor.clamp(min = 0, max = sub_2);  floor = sub_2 = None
  long = clamp.long();  clamp = None
  range_1 = range(len_6);  len_6 = None
  eq_2 = long == 0
  nonzero = eq_2.nonzero(as_tuple = False);  eq_2 = None
  squeeze = nonzero.squeeze(1);  nonzero = None
  numel = squeeze.numel()
  gt_1 = numel > 0;  numel = None
  getitem_26 = cat_1[squeeze];  cat_1 = None
  getitem_27 = getitem_20[0]
  new_tensor = getitem_26.new_tensor([0.0, 2.0, 2.0, 2.0, 2.0])
  sub_3 = getitem_26 - new_tensor;  getitem_26 = new_tensor = None
  roi_align = torchvision_ops_roi_align_roi_align(getitem_27, sub_3, (7, 7), 0.25, 0);  getitem_27 = sub_3 = None
  new_zeros[squeeze] = roi_align;  setitem = new_zeros;  squeeze = roi_align = None
  eq_3 = long == 1
  nonzero_1 = eq_3.nonzero(as_tuple = False);  eq_3 = None
  squeeze_1 = nonzero_1.squeeze(1);  nonzero_1 = None
  numel_1 = squeeze_1.numel();  squeeze_1 = None
  gt_2 = numel_1 > 0;  numel_1 = None
  add_20 = new_zeros + 0.0;  new_zeros = None
  getitem_28 = getitem_20[1]
  sum_1 = getitem_28.sum();  getitem_28 = None
  mul_1 = sum_1 * 0.0;  sum_1 = None
  add_21 = add_20 + mul_1;  add_20 = mul_1 = None
  eq_4 = long == 2
  nonzero_2 = eq_4.nonzero(as_tuple = False);  eq_4 = None
  squeeze_2 = nonzero_2.squeeze(1);  nonzero_2 = None
  numel_2 = squeeze_2.numel();  squeeze_2 = None
  gt_3 = numel_2 > 0;  numel_2 = None
  add_22 = add_21 + 0.0;  add_21 = None
  getitem_29 = getitem_20[2]
  sum_2 = getitem_29.sum();  getitem_29 = None
  mul_2 = sum_2 * 0.0;  sum_2 = None
  add_23 = add_22 + mul_2;  add_22 = mul_2 = None
  eq_5 = long == 3;  long = None
  nonzero_3 = eq_5.nonzero(as_tuple = False);  eq_5 = None
  squeeze_3 = nonzero_3.squeeze(1);  nonzero_3 = None
  numel_3 = squeeze_3.numel();  squeeze_3 = None
  gt_4 = numel_3 > 0;  numel_3 = None
  add_24 = add_23 + 0.0;  add_23 = None
  getitem_30 = getitem_20[3];  getitem_20 = None
  sum_3 = getitem_30.sum();  getitem_30 = None
  mul_3 = sum_3 * 0.0;  sum_3 = None
  add_25 = add_24 + mul_3;  add_24 = mul_3 = None
  flatten = add_25.flatten(1);  add_25 = None
  roi_head_bbox_head_shared_fcs_0 = self.roi_head.bbox_head.shared_fcs[0](flatten);  flatten = None
  roi_head_bbox_head_relu = self.roi_head.bbox_head.relu(roi_head_bbox_head_shared_fcs_0);  roi_head_bbox_head_shared_fcs_0 = None
  roi_head_bbox_head_shared_fcs_1 = self.roi_head.bbox_head.shared_fcs[1](roi_head_bbox_head_relu);  roi_head_bbox_head_relu = None
  roi_head_bbox_head_relu_1 = self.roi_head.bbox_head.relu(roi_head_bbox_head_shared_fcs_1);  roi_head_bbox_head_shared_fcs_1 = None
  dim = roi_head_bbox_head_relu_1.dim()
  gt_5 = dim > 2;  dim = None
  dim_1 = roi_head_bbox_head_relu_1.dim()
  gt_6 = dim_1 > 2;  dim_1 = None
  roi_head_bbox_head_fc_cls = self.roi_head.bbox_head.fc_cls(roi_head_bbox_head_relu_1)
  roi_head_bbox_head_fc_reg = self.roi_head.bbox_head.fc_reg(roi_head_bbox_head_relu_1);  roi_head_bbox_head_relu_1 = None
  return (tuple_6, (roi_head_bbox_head_fc_cls, roi_head_bbox_head_fc_reg))

with torch.no_grad():
  cfg = model.cfg
  cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
  test_pipeline = Compose(cfg.data.test.pipeline)
  img_data = test_pipeline(dict(img_info=dict(filename=img), img_prefix=None))['img']
  img_tensor = collate(img_data, 1).data[0]
  input_like = torch.rand_like(img_tensor)
  
  torch.manual_seed(100)
  out_orig = model.forward_dummy(img_tensor)
  # torch.manual_seed(100)
  # assert check_equal(out_orig, forward(model, img_tensor))
  # torch.manual_seed(100)
  # assert check_equal(out_orig, model.forward_dummy(img_tensor))

  torch.manual_seed(100)
  from mmcv.ops.roi_align import roi_align as mmcv_roi_align
  from torchvision.ops import roi_align as tv_roi_align
  import torchvision
  traced_model = concrete_trace(model, {'img': img_tensor}, False, forwrad_function_name='forward_dummy', autowrap_leaf_function = {
    **ConcreteTracer.default_autowrap_leaf_function,
    mmcv_roi_align: ((), False, None),
    tv_roi_align: (((torchvision.ops, 'roi_align'),), False, None),
  })
  print('traced code:\n', traced_model.code)

  torch.manual_seed(100)
  out_orig_traced = traced_model(img_tensor)
  assert check_equal(out_orig, out_orig_traced)
  
  torch.manual_seed(100)
  out_like = model.forward_dummy(input_like)
  torch.manual_seed(100)
  out_like_traced = traced_model(input_like)
  assert check_equal(out_like, out_like_traced)