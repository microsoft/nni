# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is an exmaple for how to fuse pruning and distillation.
"""

import pickle

import torch

from examples.compression.models import (
    build_resnet18,
    prepare_dataloader,
    prepare_optimizer,
    train,
    training_step,
    evaluate,
    device
)

from nni.compression import TorchEvaluator
from nni.compression.base.compressor import Quantizer
from nni.compression.distillation import DynamicLayerwiseDistiller
from nni.compression.pruning import TaylorPruner, AGPPruner
from nni.compression.quantization import QATQuantizer
from nni.compression.utils import auto_set_denpendency_group_ids
from nni.compression.speedup import ModelSpeedup


if __name__ == '__main__':
    # finetuning resnet18 on Cifar10
    model = build_resnet18()
    optimizer = prepare_optimizer(model)
    train(model, optimizer, training_step, lr_scheduler=None, max_steps=None, max_epochs=30)
    _, test_loader = prepare_dataloader()
    print('Original model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Original model after 10 epochs finetuning acc: ', evaluate(model, test_loader), '%')

    # build a teacher model
    teacher_model = build_resnet18()
    teacher_model.load_state_dict(pickle.loads(pickle.dumps(model.state_dict())))

    # create pruner
    bn_list = [module_name for module_name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
    p_config_list = [{
        'op_types': ['Conv2d'],
        'sparse_ratio': 0.5
    }, *[{
        'op_names': [name],
        'target_names': ['_output_'],
        'target_settings': {
            '_output_': {
                'align': {
                    'module_name': name.replace('bn', 'conv') if 'bn' in name else name.replace('downsample.1', 'downsample.0'),
                    'target_name': 'weight',
                    'dims': [0],
                },
                'granularity': 'per_channel'
            }
        }
    } for name in bn_list]]
    dummy_input = torch.rand(8, 3, 224, 224).to(device)
    p_config_list = auto_set_denpendency_group_ids(model, p_config_list, dummy_input)

    optimizer = prepare_optimizer(model)
    evaluator = TorchEvaluator(train, optimizer, training_step)
    sub_pruner = TaylorPruner(model, p_config_list, evaluator, training_steps=100)
    scheduled_pruner = AGPPruner(sub_pruner, interval_steps=100, total_times=30)

    # create quantizer
    q_config_list = [{
        'op_types': ['Conv2d'],
        'quant_dtype': 'int8',
        'target_names': ['_input_'],
        'granularity': 'per_channel'
    }, {
        'op_types': ['Conv2d'],
        'quant_dtype': 'int8',
        'target_names': ['weight'],
        'granularity': 'out_channel'
    }, {
        'op_types': ['BatchNorm2d'],
        'quant_dtype': 'int8',
        'target_names': ['_output_'],
        'granularity': 'per_channel'
    }]

    quantizer = QATQuantizer.from_compressor(scheduled_pruner, q_config_list, quant_start_step=100)

    # create distiller
    def teacher_predict(batch, teacher_model):
        return teacher_model(batch[0])

    d_config_list = [{
        'op_types': ['Conv2d'],
        'lambda': 0.1,
        'apply_method': 'mse',
    }]
    distiller = DynamicLayerwiseDistiller.from_compressor(quantizer, d_config_list, teacher_model, teacher_predict, 0.1)

    # max_steps contains (30 iterations 100 steps agp taylor pruning, and 3000 steps finetuning)
    distiller.compress(max_steps=100 * 60, max_epochs=None)
    distiller.unwrap_model()
    distiller.unwrap_teacher_model()

    # speed up model
    masks = scheduled_pruner.get_masks()
    speedup = ModelSpeedup(model, dummy_input, masks)
    model = speedup.speedup_model()

    print('Compressed model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Compressed model without finetuning & qsim acc: ', evaluate(model, test_loader), '%')

    # simulate quantization
    calibration_config = quantizer.get_calibration_config()

    def trans(calibration_config, speedup: ModelSpeedup):
        for node, node_info in speedup.node_infos.items():
            if node.op == 'call_module' and node.target in calibration_config:
                # assume the module only has one input and one output
                input_mask = speedup.node_infos[node.args[0]].output_masks
                param_mask = node_info.param_masks
                output_mask = node_info.output_masks

                module_cali_config = calibration_config[node.target]
                if '_input_0' in module_cali_config:
                    reduce_dims = list(range(len(input_mask.shape)))
                    reduce_dims.remove(1)
                    idxs = torch.nonzero(input_mask.sum(reduce_dims), as_tuple=True)[0].cpu()
                    module_cali_config['_input_0']['scale'] = module_cali_config['_input_0']['scale'].index_select(1, idxs)
                    module_cali_config['_input_0']['zero_point'] = module_cali_config['_input_0']['zero_point'].index_select(1, idxs)
                if '_output_0' in module_cali_config:
                    reduce_dims = list(range(len(output_mask.shape)))
                    reduce_dims.remove(1)
                    idxs = torch.nonzero(output_mask.sum(reduce_dims), as_tuple=True)[0].cpu()
                    module_cali_config['_output_0']['scale'] = module_cali_config['_output_0']['scale'].index_select(1, idxs)
                    module_cali_config['_output_0']['zero_point'] = module_cali_config['_output_0']['zero_point'].index_select(1, idxs)
                if 'weight' in module_cali_config:
                    reduce_dims = list(range(len(param_mask['weight'].shape)))
                    reduce_dims.remove(0)
                    idxs = torch.nonzero(param_mask['weight'].sum(reduce_dims), as_tuple=True)[0].cpu()
                    module_cali_config['weight']['scale'] = module_cali_config['weight']['scale'].index_select(0, idxs)
                    module_cali_config['weight']['zero_point'] = module_cali_config['weight']['zero_point'].index_select(0, idxs)
                if 'bias' in module_cali_config:
                    idxs = torch.nonzero(param_mask['bias'], as_tuple=True)[0].cpu()
                    module_cali_config['bias']['scale'] = module_cali_config['bias']['scale'].index_select(0, idxs)
                    module_cali_config['bias']['zero_point'] = module_cali_config['bias']['zero_point'].index_select(0, idxs)
        return calibration_config

    calibration_config = trans(calibration_config, speedup)

    sim_quantizer = Quantizer(model, q_config_list)
    sim_quantizer.update_calibration_config(calibration_config)

    print('Compressed model without finetuning acc: ', evaluate(model, test_loader), '%')
