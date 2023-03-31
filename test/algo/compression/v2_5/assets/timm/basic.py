# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import timm.models as tm
import torch

from ..registry import model_zoo


def config_list(*_):
    return [{
        'sparsity': 0.2,
        'op_types': ['Conv2d']
    }]
    
def dummy_inputs(*_):
    return {'x': torch.randn(2, 3, 224, 224)}

model_zoo.register(
    'timm', 'beit',
    tm.beit_base_patch16_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'beitv2',
    tm.beitv2_base_patch16_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'cait',
    tm.cait_s24_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'coat',
    tm.coat_lite_mini,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'convit',
    tm.convit_base,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'deit',
    tm.deit3_base_patch16_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'dm_nfnet',
    tm.dm_nfnet_f0,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (__bool__ proxy)',
)

model_zoo.register(
    'timm', 'eca_nfnet',
    tm.eca_nfnet_l0,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot prune (empty mask -- customized conv)',
)

model_zoo.register(
    'timm', 'efficientformer',
    tm.efficientformer_l1,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'timm', 'ese_vovnet',
    tm.ese_vovnet19b_dw,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    need_run=True,
    skip_reason='cannot prune (native_batch_norm not differentiable)',
)

model_zoo.register(
    'timm', 'gmixer',
    tm.gmixer_12_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'gmlp',
    tm.gmlp_b16_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'hardcorenas',
    tm.hardcorenas_a,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'hrnet',
    tm.hrnet_w18_small,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'timm', 'inception',
    tm.inception_v3,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'timm', 'mixer',
    tm.mixer_b16_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'nf_ecaresnet',
    tm.nf_ecaresnet101,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot prune (empty mask -- customized conv)',
)

model_zoo.register(
    'timm', 'nf_regnet',
    tm.nf_regnet_b0,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'regnet',
    tm.regnetv_040,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'skresnet',
    tm.skresnet18,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (diff # of nodes)',
)

model_zoo.register(
    'timm', 'swin',
    tm.swin_base_patch4_window7_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'tnt',
    tm.tnt_b_patch16_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'timm', 'vgg',
    tm.vgg11,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)

model_zoo.register(
    'timm', 'vit',
    tm.vit_base_patch16_18x2_224,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
    skip_reason='cannot trace (torch._assert)',
)

model_zoo.register(
    'timm', 'wide_resnet',
    tm.wide_resnet50_2,
    dummy_inputs=dummy_inputs,
    config_list=config_list,
)
