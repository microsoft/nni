# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from functools import partial
from os.path import dirname, exists, join

import mmcv.cnn as mmcv_cnn
import mmcv.ops as mmcv_ops
import mmdet.core as mmdet_core

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
    fake_middle_class=(
        mmdet_core.anchor.anchor_generator.AnchorGenerator,
    ),
    forward_function_name='forward_dummy',
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

def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config = Config.fromfile(config_fpath)
    return config

def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.
    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    _roi_align_setter(model)
    return model

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
    from mmcv.parallel import collate
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    cfg = _get_config_module(fname)
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    img_data = test_pipeline(dict(img_info=dict(filename=_get_image_directory()), img_prefix=None))['img']
    img_tensor = collate(img_data, 1).data[0]
    return dict(
        img=img_tensor,
    )

def build_detector_from_cfg(fname):
    """Build a detector from a config dict.
    Args:
        cfg (dict): The detector config.
    Returns:
        torch.nn.Module: The constructed detector.
    """
    model = _get_detector_cfg(fname)
    from mmdet.models import build_detector
    model = build_detector(model)
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
    partial(build_detector_from_cfg, 'autoassign/autoassign_r50_fpn_8x2_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='autoassign/autoassign_r50_fpn_8x2_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'cascade_rcnn',
    partial(build_detector_from_cfg, 'cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco.py'),
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
    partial(build_detector_from_cfg, 'centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py'),
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
    partial(build_detector_from_cfg, 'cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'dcn',
    partial(build_detector_from_cfg, 'dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'),
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
    partial(build_detector_from_cfg, 'deformable_detr/deformable_detr_r50_16x2_50e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='deformable_detr/deformable_detr_r50_16x2_50e_coco.py'),
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
    partial(build_detector_from_cfg, 'dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'faster_rcnn',
    partial(build_detector_from_cfg, 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'),
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
    partial(build_detector_from_cfg, 'foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'free_anchor',
    partial(build_detector_from_cfg, 'free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py'),
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
    partial(build_detector_from_cfg, 'gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py'),
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
    'mmdet', 'gn',
    partial(build_detector_from_cfg, 'gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'gn+ws',
    partial(build_detector_from_cfg, 'gn+ws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='gn+ws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'grid_rcnn',
    partial(build_detector_from_cfg, 'grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'guided_anchoring',
    partial(build_detector_from_cfg, 'guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'hrnet',
    partial(build_detector_from_cfg, 'hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py'),
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
    partial(build_detector_from_cfg, 'libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'mask_rcnn',
    partial(build_detector_from_cfg, 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'ms_rcnn',
    partial(build_detector_from_cfg, 'ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'nas_fcos',
    partial(build_detector_from_cfg, 'nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot trace (modulated_deform_conv)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'nas_fpn',
    partial(build_detector_from_cfg, 'nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py'),
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
    partial(build_detector_from_cfg, 'pafpn/faster_rcnn_r50_pafpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='pafpn/faster_rcnn_r50_pafpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'point_rend',
    partial(build_detector_from_cfg, 'point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'regnet',
    partial(build_detector_from_cfg, 'regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'reppoints',
    partial(build_detector_from_cfg, 'reppoints/reppoints_moment_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='reppoints/reppoints_moment_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'res2net',
    partial(build_detector_from_cfg, 'res2net/faster_rcnn_r2_101_fpn_2x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='res2net/faster_rcnn_r2_101_fpn_2x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'resnest',
    partial(build_detector_from_cfg, 'resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py'),
    config_list=config_list,
    skip_reason='cannot run (syncbn need gpu)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'retinanet',
    partial(build_detector_from_cfg, 'retinanet/retinanet_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='retinanet/retinanet_r50_fpn_1x_coco.py'),
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
    partial(build_detector_from_cfg, 'sabl/sabl_retinanet_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='sabl/sabl_retinanet_r50_fpn_1x_coco.py'),
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
    partial(build_detector_from_cfg, 'scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'sparse_rcnn',
    partial(build_detector_from_cfg, 'sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'solo',
    partial(build_detector_from_cfg, 'solo/decoupled_solo_light_r50_fpn_3x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='solo/decoupled_solo_light_r50_fpn_3x_coco.py'),
    config_list=config_list,
    skip_reason='cannot run (no ``forward_dummy``)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'solo_v2',
    partial(build_detector_from_cfg, 'solov2/solov2_light_r18_fpn_3x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='solov2/solov2_light_r18_fpn_3x_coco.py'),
    config_list=config_list,
    skip_reason='cannot run (no ``forward_dummy``)',
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'ssd',
    partial(build_detector_from_cfg, 'ssd/ssd300_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='ssd/ssd300_coco.py'),
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
    partial(build_detector_from_cfg, 'tridentnet/tridentnet_r50_caffe_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='tridentnet/tridentnet_r50_caffe_1x_coco.py'),
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
    partial(build_detector_from_cfg, 'yolact/yolact_r50_1x8_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='yolact/yolact_r50_1x8_coco.py'),
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
    partial(build_detector_from_cfg, 'yolof/yolof_r50_c5_8x8_1x_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='yolof/yolof_r50_c5_8x8_1x_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)

model_zoo.register(
    'mmdet', 'yolox',
    partial(build_detector_from_cfg, 'yolox/yolox_s_8x8_300e_coco.py'),
    dummy_inputs=partial(dummy_input_from_cfg, fname='yolox/yolox_s_8x8_300e_coco.py'),
    config_list=config_list,
    **extra_kwargs,
)
