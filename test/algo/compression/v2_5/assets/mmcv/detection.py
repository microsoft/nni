# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from functools import partial
from os.path import dirname, exists, join

import mmcv.cnn as mmcv_cnn
import mmcv.ops as mmcv_ops
from mmdet.apis import init_detector
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmengine import Config

from ..registry import model_zoo

def config_list(*_):
    return [{
        'sparsity': 0.2,
        'op_types': ['Conv2d'],
    }]

extra_kwargs = dict(
    leaf_module=(
        mmcv_ops.RoIAlign,
        mmcv_cnn.bricks.wrappers.Conv2d,
        mmcv_cnn.bricks.wrappers.Conv3d,
        mmcv_cnn.bricks.wrappers.ConvTranspose2d,
        mmcv_cnn.bricks.wrappers.ConvTranspose3d,
        mmcv_cnn.bricks.wrappers.Linear,
        mmcv_cnn.bricks.wrappers.MaxPool2d,
        mmcv_cnn.bricks.wrappers.MaxPool3d,
    ),
)

def _get_config_directory():
    """Find the predefined detector config directory."""
    import mmdet
    repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath

def _get_image_directory():
    """Find the predefined image directory."""
    import mmdet
    repo_dpath = dirname(dirname(mmdet.__file__))
    img_dpath = join(repo_dpath, 'tests', 'data', 'color.jpg')
    if not exists(img_dpath):
        raise Exception('Cannot find image path')
    return img_dpath

def _get_config_module(fname) -> Config:
    """Load a configuration as a python module."""
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config = Config.fromfile(config_fpath)
    return config

def _roi_align_setter(config_dict: dict):
    if 'type' in config_dict:
        if config_dict['type'] == 'RoIAlign':
            config_dict['use_torchvision'] = True
            config_dict['aligned'] = False
            pass
        else:
            for v in config_dict.values():
                if isinstance(v, dict):
                    _roi_align_setter(v)

def dummy_input_from_cfg(self, fname):
    packed_inputs = demo_mm_inputs()
    return self.data_preprocessor(packed_inputs, False)

def build_detector_from_cfg(fname):
    """Build a detector from a config dict.
    Args:
        cfg (dict): The detector config.
    Returns:
        torch.nn.Module: The constructed detector.
    """
    config = _get_config_module(fname) 
    _roi_align_setter(config._cfg_dict['model'])
    model = init_detector(config)
    return model


model_zoo.register(
    'mmdet', 'atss',
    partial(build_detector_from_cfg, 'atss/atss_r50_fpn_1x_coco.py'),
    dummy_inputs = partial(dummy_input_from_cfg, fname='atss/atss_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'autoassign',
    partial(build_detector_from_cfg, 'autoassign/autoassign_r50-caffe_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='autoassign/autoassign_r50-caffe_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'cascade_rcnn',
    partial(build_detector_from_cfg, 'cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'centernet',
    partial(build_detector_from_cfg, 'centernet/centernet_resnet18_140e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='centernet/centernet_resnet18_140e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'centripetalnet',
    partial(build_detector_from_cfg, 'centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'cityscapes',
    partial(build_detector_from_cfg, 'cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'cornernet',
    partial(build_detector_from_cfg, 'cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'dcn',
    partial(build_detector_from_cfg, 'dcn/cascade-mask-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='dcn/cascade-mask-rcnn_r50-dconv-c3-c5_fpn_1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot trace (deform_conv error)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'dcnv2',
    partial(build_detector_from_cfg, 'dcnv2/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='dcnv2/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot trace (deform_conv error)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'deformable_detr',
    partial(build_detector_from_cfg, 'deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'double_heads',
    partial(build_detector_from_cfg, 'double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'dynamic_rcnn',
    partial(build_detector_from_cfg, 'dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'faster_rcnn',
    partial(build_detector_from_cfg, 'faster_rcnn/faster-rcnn_r18_fpn_8xb8-amp-lsj-200e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='faster_rcnn/faster-rcnn_r18_fpn_8xb8-amp-lsj-200e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'fcos',
    partial(build_detector_from_cfg, 'fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'foveabox',
    partial(build_detector_from_cfg, 'foveabox/fovea_r50_fpn_4xb4-1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='foveabox/fovea_r50_fpn_4xb4-1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'fpg',
    partial(build_detector_from_cfg, 'fpg/faster-rcnn_r50_fpg_crop640-50e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='fpg/faster-rcnn_r50_fpg_crop640-50e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)
    

model_zoo.register(
    'mmdet', 'free_anchor',
    partial(build_detector_from_cfg, 'free_anchor/freeanchor_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='free_anchor/freeanchor_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'fsaf',
    partial(build_detector_from_cfg, 'fsaf/fsaf_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='fsaf/fsaf_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'gcnet',
    partial(build_detector_from_cfg, 'gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco'),
    config_list=config_list,
    skip_reason='cannot run (syncbn need gpu)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'gfl',
    partial(build_detector_from_cfg, 'gfl/gfl_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='gfl/gfl_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'ghm',
    partial(build_detector_from_cfg, 'ghm/retinanet_r50_fpn_ghm-1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='ghm/retinanet_r50_fpn_ghm-1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'gn',
    partial(build_detector_from_cfg, 'gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'gn+ws',
    partial(build_detector_from_cfg, 'gn+ws/faster-rcnn_r50_fpn_gn_ws-all_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='gn+ws/faster-rcnn_r50_fpn_gn_ws-all_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'grid_rcnn',
    partial(build_detector_from_cfg, 'grid_rcnn/grid-rcnn_r50_fpn_gn-head_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='grid_rcnn/grid-rcnn_r50_fpn_gn-head_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'guided_anchoring',
    partial(build_detector_from_cfg, 'guided_anchoring/ga-retinanet_r50-caffe_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='guided_anchoring/ga-retinanet_r50-caffe_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'hrnet',
    partial(build_detector_from_cfg, 'hrnet/cascade-mask-rcnn_hrnetv2p-w18_20e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='hrnet/cascade-mask-rcnn_hrnetv2p-w18_20e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'htc',
    partial(build_detector_from_cfg, 'htc/htc_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='htc/htc_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'libra_rcnn',
    partial(build_detector_from_cfg, 'libra_rcnn/cascade-mask-rcnn_r50_fpn_instaboost-4x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='libra_rcnn/cascade-mask-rcnn_r50_fpn_instaboost-4x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'lvis',
    partial(build_detector_from_cfg, 'lvis/mask-rcnn_r50_fpn_sample1e-3_mstrain-1x_lvis-v1.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='lvis/mask-rcnn_r50_fpn_sample1e-3_mstrain-1x_lvis-v1.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'mask_rcnn',
    partial(build_detector_from_cfg, 'mask_rcnn/mask-rcnn_r18_fpn_8xb8-amp-lsj-200e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='mask_rcnn/mask-rcnn_r18_fpn_8xb8-amp-lsj-200e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'ms_rcnn',
    partial(build_detector_from_cfg, 'ms_rcnn/ms-rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='ms_rcnn/ms-rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'nas_fcos',
    partial(build_detector_from_cfg, 'nas_fcos/nas-fcos_r50-caffe_fpn_fcoshead-gn-head_4xb4-1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='nas_fcos/nas-fcos_r50-caffe_fpn_fcoshead-gn-head_4xb4-1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot trace (modulated_deform_conv)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'nas_fpn',
    partial(build_detector_from_cfg, 'nas_fpn/retinanet_r50_fpn_crop640-50e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='nas_fpn/retinanet_r50_fpn_crop640-50e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'openimages',
    partial(build_detector_from_cfg, 'openimages/faster-rcnn_r50_fpn_32xb2-1x_openimages.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='openimages/faster-rcnn_r50_fpn_32xb2-1x_openimages.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'paa',
    partial(build_detector_from_cfg, 'paa/paa_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='paa/paa_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'pafpn',
    partial(build_detector_from_cfg, 'pafpn/faster-rcnn_r50_pafpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='pafpn/faster-rcnn_r50_pafpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'point_rend',
    partial(build_detector_from_cfg, 'point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'pvt',
    partial(build_detector_from_cfg, 'pvt/retinanet_pvt-l_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='pvt/retinanet_pvt-l_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'queryinst',
    partial(build_detector_from_cfg, 'queryinst/queryinst_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='queryinst/queryinst_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'regnet',
    partial(build_detector_from_cfg, 'regnet/cascade-mask-rcnn_regnetx-1.6GF_fpn_ms-3x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='regnet/cascade-mask-rcnn_regnetx-1.6GF_fpn_ms-3x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'reppoints',
    partial(build_detector_from_cfg, 'reppoints/reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='reppoints/reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'res2net',
    partial(build_detector_from_cfg, 'res2net/cascade-mask-rcnn_res2net-101_fpn_20e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='res2net/cascade-mask-rcnn_res2net-101_fpn_20e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'resnet_strikes_back',
    partial(build_detector_from_cfg, 'resnet_strikes_back/cascade-mask-rcnn_r50-rsb-pre_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='resnet_strikes_back/cascade-mask-rcnn_r50-rsb-pre_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'retinanet',
    partial(build_detector_from_cfg, 'retinanet/retinanet_r18_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='retinanet/retinanet_r18_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'rpn',
    partial(build_detector_from_cfg, 'rpn/rpn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='rpn/rpn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'sabl',
    partial(build_detector_from_cfg, 'sabl/sabl-cascade-rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='sabl/sabl-cascade-rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'scnet',
    partial(build_detector_from_cfg, 'scnet/scnet_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='scnet/scnet_r50_fpn_1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot run (int object not subscriptable in SCNetRoIHead)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'scratch',
    partial(build_detector_from_cfg, 'scratch/faster-rcnn_r50-scratch_fpn_gn-all_6x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='scratch/faster-rcnn_r50-scratch_fpn_gn-all_6x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'sparse_rcnn',
    partial(build_detector_from_cfg, 'sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'solo',
    partial(build_detector_from_cfg, 'solo/decoupled-solo_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='solo/decoupled-solo_r50_fpn_1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot run (no ``forward_dummy``)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'solo_v2',
    partial(build_detector_from_cfg, 'solov2/solov2_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='solov2/solov2_r50_fpn_1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot run (no ``forward_dummy``)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'ssd',
    partial(build_detector_from_cfg, 'ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'swin',
    partial(build_detector_from_cfg, 'swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'tridentnet',
    partial(build_detector_from_cfg, 'tridentnet/tridentnet_r50-caffe_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='tridentnet/tridentnet_r50-caffe_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'tood',
    partial(build_detector_from_cfg, 'tood/tood_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='tood/tood_r50_fpn_1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot trace (deform_conv error)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'vfnet',
    partial(build_detector_from_cfg, 'vfnet/vfnet_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='vfnet/vfnet_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'yolact',
    partial(build_detector_from_cfg, 'yolact/yolact_r50_1xb8-55e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='yolact/yolact_r50_1xb8-55e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'yolo',
    partial(build_detector_from_cfg, 'yolo/yolov3_d53_320_273e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='yolo/yolov3_d53_320_273e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'yolof',
    partial(build_detector_from_cfg, 'yolof/yolof_r50-c5_8xb8-1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='yolof/yolof_r50-c5_8xb8-1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'yolox',
    partial(build_detector_from_cfg, 'yolox/yolox_l_8xb8-300e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='yolox/yolox_s_8x8_300e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)
