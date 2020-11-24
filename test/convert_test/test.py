import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from nni.retiarii.converter.graph_gen import convert_to_graph
from nni.retiarii.converter.visualize import visualize_model
from nni.retiarii import nn
from nni.retiarii.codegen.pytorch import model_to_pytorch_script

from base_mnasnet import MNASNet
from nni.experiment import Experiment

if __name__ == '__main__':
    _DEFAULT_DEPTHS = [16, 24, 40, 80, 96, 192, 320]
    _DEFAULT_CONVOPS = ["dconv", "mconv", "mconv", "mconv", "mconv", "mconv", "mconv"]
    _DEFAULT_SKIPS = [False, True, True, True, True, True, True]
    _DEFAULT_KERNEL_SIZES = [3, 3, 5, 5, 3, 5, 3]
    _DEFAULT_NUM_LAYERS = [1, 3, 3, 3, 2, 4, 1]
    nn.enable_record_args()
    base_model = MNASNet(0.5, _DEFAULT_DEPTHS, _DEFAULT_CONVOPS, _DEFAULT_KERNEL_SIZES,
                    _DEFAULT_NUM_LAYERS, _DEFAULT_SKIPS)
    recorded_module_args = nn.get_records()
    nn.disable_record_args()
    print(recorded_module_args)
    script_module = torch.jit.script(base_model)
    model = convert_to_graph(script_module, base_model, recorded_module_args)
    #code_script = model_to_pytorch_script(model)
    #print(code_script)
    print("Model: ", model)
    graph_ir = model._dump()
    print(graph_ir)
    #visualize_model(graph_ir)

    # TODO: new interface
    #exp = Experiment()
    #exp.start_retiarii_experiment(base_model, training_approach,
    #                              applied_mutators, strategy,
    #                              exp_config)

    exp_config = {'authorName': 'nni',
                  'experimentName': 'naive',
                  'trialConcurrency': 3,
                  'maxExecDuration': '1h',
                  'maxTrialNum': 10,
                  'trainingServicePlatform': 'local'
                }
    applied_mutators = [{'filepath': os.path.join(os.getcwd(), 'mutator.py'), 'classname': 'BlockMutator', 'args': {'target': 'mutable_0'}},
                        {'filepath': os.path.join(os.getcwd(), 'mutator.py'), 'classname': 'BlockMutator', 'args': {'target': 'mutable_1'}}]
    training_approach = {'modulename': 'nni.retiarii.trainer.PyTorchImageClassificationTrainer', 'args': {
        "dataset_cls": "CIFAR10",
        "dataset_kwargs": {
                "root": "data/cifar10",
                "download": True
        },
        "dataloader_kwargs": {
            "batch_size": 32
        },
        "optimizer_kwargs": {
            "lr": 1e-3
        },
        "trainer_kwargs": {
            "max_epochs": 1
        }
    }}
    strategy = {'filename': 'simple_strategy', 'funcname': 'simple_startegy', 'args': {}}
    exp = Experiment()
    exp.tmp_start_retiarii(graph_ir, training_approach,
                           applied_mutators, strategy,
                           exp_config)
