# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import torch
import mmcv
import mmcv.cnn as mmcv_cnn
import mmdet.core as mmdet_core
from mmdet.apis import init_detector
from mmcv.parallel import collate
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from nni.common.concrete_trace_utils import concrete_trace, ConcreteTracer

folder_prefix = 'C:\\Works\\mmdetection' # replace this path with yours
img = '%s/tests/data/color.jpg' % folder_prefix

config_files_correct = (
    'atss/atss_r50_fpn_1x_coco',
    'autoassign/autoassign_r50_fpn_8x2_1x_coco',
    'cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco',
    'centernet/centernet_resnet18_140e_coco',
    'centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco',
    'cityscapes/faster_rcnn_r50_fpn_1x_cityscapes',
    'cornernet/cornernet_hourglass104_mstest_8x6_210e_coco',
    'dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco',
    'dcnv2/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco',
    'ddod/ddod_r50_fpn_1x_coco',
    'deepfashion/mask_rcnn_r50_fpn_15e_deepfashion',
    'deformable_detr/deformable_detr_r50_16x2_50e_coco',
    'detr/detr_r50_8x2_150e_coco',
    'double_heads/dh_faster_rcnn_r50_fpn_1x_coco',
    'dyhead/atss_r50_caffe_fpn_dyhead_1x_coco',
    'dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco',
    'empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco',
    'faster_rcnn/faster_rcnn_r50_fpn_1x_coco',
    'fcos/fcos_center_r50_caffe_fpn_gn-head_1x_coco',
    'foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco',
    'fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco',
    'free_anchor/retinanet_free_anchor_r50_fpn_1x_coco',
    'fsaf/fsaf_r50_fpn_1x_coco',
    'gfl/gfl_r50_fpn_1x_coco',
    'ghm/retinanet_ghm_r50_fpn_1x_coco',
    'gn/mask_rcnn_r50_fpn_gn-all_2x_coco',
    'gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco',
    'grid_rcnn/grid_rcnn_r50_fpn_gn-head_1x_coco',
    'groie/faster_rcnn_r50_fpn_groie_1x_coco',
    'hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco',
    'htc/htc_r50_fpn_1x_coco',
    'instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco',
    'legacy_1.x/faster_rcnn_r50_fpn_1x_coco_v1',
    'lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1',
    'mask_rcnn/mask_rcnn_r50_caffe_c4_1x_coco',
    'ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco',
    'nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco',
    'nas_fpn/retinanet_r50_fpn_crop640_50e_coco',
    'openimages/faster_rcnn_r50_fpn_32x2_1x_openimages',
    'paa/paa_r50_fpn_1x_coco',
    'pafpn/faster_rcnn_r50_pafpn_1x_coco',
    'pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712',
    'pisa/pisa_faster_rcnn_r50_fpn_1x_coco',
    'point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco',
    'pvt/retinanet_pvt-l_fpn_1x_coco',
    'queryinst/queryinst_r50_fpn_1x_coco',
    'regnet/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco',
    'reppoints/bbox_r50_grid_center_fpn_gn-neck+head_1x_coco',
    'res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco',
    'resnet_strikes_back/cascade_mask_rcnn_r50_fpn_rsb-pretrain_1x_coco',
    'retinanet/retinanet_r18_fpn_1x_coco',
    'rpn/rpn_r50_caffe_c4_1x_coco',
    'sabl/sabl_cascade_rcnn_r50_fpn_1x_coco',
    'seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1',
    'scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco',
    'sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco',
    'ssd/ssdlite_mobilenetv2_scratch_600e_coco',
    'swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco',
    'timm_example/retinanet_timm_efficientnet_b1_fpn_1x_coco',
    'tood/tood_r50_fpn_1x_coco',
    'tridentnet/tridentnet_r50_caffe_1x_coco',
    'vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco',
    'wider_face/ssd300_wider_face',
    'yolact/yolact_r50_1x8_coco',
    'yolo/yolov3_d53_320_273e_coco',
    'yolof/yolof_r50_c5_8x8_1x_coco',
    'yolox/yolox_nano_8x8_300e_coco',
)

# has exceptions:

# cannot get model:
# 'MaskRCNN: StandardRoIHead: Shared4Conv1FCBBoxHead: Default process group has not been initialized, please make sure to call init_process_group.'
config_files_maskrcnn = (
    'simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco',
    'strong_baselines/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco',
)

# cannot run model: need gpu
config_files_need_gpu = (
    'carafe/faster_rcnn_r50_fpn_carafe_1x_coco',
    'convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco',
    'efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco',
    'gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco',
    'resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco',
    'selfsup_pretrain/mask_rcnn_r50_fpn_mocov2-pretrain_1x_coco',
)

# cannot run model: need argument img_metas
config_files_img_metas = (
    'mask2former/mask2former_r50_lsj_8x2_50e_coco',
    'maskformer/maskformer_r50_mstrain_16x1_75e_coco',
)

# cannot run model: no forward_dummy
config_files_no_forward_dummy = (
    'panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco',
    'solo/decoupled_solo_light_r50_fpn_3x_coco',
    'solov2/solov2_light_r18_fpn_3x_coco',
)

# cannot build input: 'proposals'
config_files_proposals = (
    'cascade_rpn/crpn_fast_rcnn_r50_caffe_fpn_1x_coco',
    'fast_rcnn/fast_rcnn_r50_caffe_fpn_1x_coco',
    'guided_anchoring/ga_fast_r50_caffe_fpn_1x_coco',
    'libra_rcnn/libra_fast_rcnn_r50_fpn_1x_coco',
)

# unknown/other
config_files_other = (
    # cannot compare result
    'lad/lad_r50_paa_r101_fpn_coco_1x',
    # cannot get model: other files do not exist
    'ld/ld_r18_gflv1_r101_fpn_coco_1x',
    # cannot run forward_dummy
    'scnet/scnet_r50_fpn_1x_coco',
    # bad result: output is tensor(nan, nan...), so cannot compare.
    'detectors/cascade_rcnn_r50_rfp_1x_coco',
)

# check equal
config_files_check_equal = (
)

for config_file in (*config_files_correct,):
    try:
        # Specify the path to model config and checkpoint file
        config = mmcv.Config.fromfile(
            folder_prefix + '/configs/' + config_file + '.py')

        # RoIAlign will cause many errors when use cpu. there are 4 ways to avoid it
        # 1. add 'mmcv_ops.RoIAlign' to leaf_module when tracing, and set 'config_dict['use_torchvision'] = True' recursively
        # 2. add 'mmcv_ops.RoIAlign' to leaf_module when tracing, and set 'config_dict['aligned'] = False' recursively
        # 3. set 'config_dict['use_torchvision'] = True', and set 'config_dict['aligned'] = False' recursively, and
        #            from torchvision.ops import roi_align as tv_roi_align
        #            add 'tv_roi_align: (((torchvision.ops, 'roi_align'),), False, None),' to autowrap_leaf_function
        # 4. set 'config_dict['aligned'] = False' recursively, and add 'from mmcv.ops.roi_align import roi_align as mmcv_roi_align' and
        #            add 'mmcv_roi_align: ((), False, None),' to autowrap_leaf_function
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

        # we should wrap `mmcv.ops.deform_conv.deform_conv2d` in this way.
        # TL;DR.
        # the `mmcv.ops.deform_conv.deform_conv2d` is a local variable of `torch.autograd.Function.apply`.
        # if we only wrap `DeformConv2dFunction.apply`, we only change the connection from `DeformConv2dFunction` to `DeformConv2dFunction.apply`,
        # but `mmcv.ops.deform_conv.deform_conv2d` is still the original `DeformConv2dFunction.apply`.
        # so we should wrap the classmethod apply functions manaully.
        from mmcv.ops.deform_conv import deform_conv2d as mmcv_deform_conv2d
        from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d as mmcv_modulated_deform_conv2d
        leaf_function_append = {
            mmcv_deform_conv2d: ((), False, None),
            mmcv_modulated_deform_conv2d: ((), False, None),
        }
        if RoIAlign_solution == 3:
            # we should wrap `torchvision.ops.roi_align` in this way.
            # TL;DR.
            # the real op is at `torch.ops.torchvision.roi_align`. but we will get a `torch._ops.OpOverloadPacket`. it's a lazy-loader.
            # so it's better to wrap the function `torchvision.ops.roi_align`.
            # the `roi_align` is at `torchvision.ops.roi_align` and `torchvision.ops.roi_align.roi_align`.
            from torchvision.ops import roi_align as tv_roi_align
            import torchvision
            leaf_function_append[tv_roi_align] = (((torchvision.ops, 'roi_align'),), False, None)
        elif RoIAlign_solution == 4:
            from mmcv.ops.roi_align import roi_align as mmcv_roi_align
            leaf_function_append[mmcv_roi_align] = ((), False, None)

        leaf_module_append = ()
        if RoIAlign_solution in (1, 2):
            from mmcv import ops as mmcv_ops
            leaf_module_append = (mmcv_ops.RoIAlign,)

        model = init_detector(config, device='cpu')

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
            traced_model = concrete_trace(model, {'img': img_tensor},
                                          use_function_patch = False, forwrad_function_name='forward_dummy',
                                          autowrap_leaf_function = {
                **ConcreteTracer.default_autowrap_leaf_function,
                **leaf_function_append,
                all:                                                                    ((), False, None),
                min:                                                                    ((), False, None),
                max:                                                                    ((), False, None),
            }, autowrap_leaf_class = {
                **ConcreteTracer.default_autowrap_leaf_class,
                int:        ((), False),
                reversed:   ((), False),
                torch.Size: ((), False),
            }, leaf_module = (
                *leaf_module_append,
                mmcv_cnn.bricks.wrappers.Conv2d,
                mmcv_cnn.bricks.wrappers.Conv3d,
                mmcv_cnn.bricks.wrappers.ConvTranspose2d,
                mmcv_cnn.bricks.wrappers.ConvTranspose3d,
                mmcv_cnn.bricks.wrappers.Linear,
                mmcv_cnn.bricks.wrappers.MaxPool2d,
                mmcv_cnn.bricks.wrappers.MaxPool3d,
            ), fake_middle_class = (
                mmdet_core.anchor.anchor_generator.AnchorGenerator,
            ))
            # print('traced code:\n', traced_model.code)

            torch.manual_seed(100)
            out_orig_traced = traced_model(img_tensor)
            assert check_equal(out_orig, out_orig_traced), 'check_equal failure 1'

            torch.manual_seed(100)
            out_like = model.forward_dummy(input_like)
            torch.manual_seed(100)
            out_like_traced = traced_model(input_like)
            assert check_equal(out_like, out_like_traced), 'check_equal failure 2'
    except Exception as e:
        print('\n\nmodel file:', config_file)
        print('status: exception')
        print('exception:', e.with_traceback(sys.exc_info()[2]))
    else:
        print('\n\nmodel file:', config_file)
        print('status: done')
