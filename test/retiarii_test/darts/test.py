import json
import os
import sys
import torch
from pathlib import Path

#sys.path.append(str(Path(__file__).resolve().parents[2]))
from nni.retiarii.converter.graph_gen import convert_to_graph
from nni.retiarii.converter.visualize import visualize_model
from nni.retiarii.codegen.pytorch import model_to_pytorch_script

from nni.retiarii.experiment import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.strategies import TPEStrategy
from nni.retiarii.trainer import PyTorchImageClassificationTrainer

from darts_model import CNN

if __name__ == '__main__':
    base_model = CNN(32, 3, 16, 10, 8)
    trainer = PyTorchImageClassificationTrainer(base_model, dataset_cls="CIFAR10",
            dataset_kwargs={"root": "data/cifar10", "download": True},
            dataloader_kwargs={"batch_size": 32},
            optimizer_kwargs={"lr": 1e-3},
            trainer_kwargs={"max_epochs": 1})
    #script_module = torch.jit.script(base_model)
    #model = convert_to_graph(script_module, base_model, tca.recorded_arguments)
    '''graph_ir = model._dump()
    with open('graph.json', 'w') as outfile:
        json.dump(graph_ir, outfile)'''
    #code_script = model_to_pytorch_script(model)
    #with open('graph_code.py', 'w') as outfile:
    #    outfile.write(code_script)
    #print(code_script)
    '''print("Model: ", model)
    graph_ir = model._dump()
    print(graph_ir)
    visualize_model(graph_ir)'''

    simple_startegy = TPEStrategy()

    exp = RetiariiExperiment(base_model, trainer, [], simple_startegy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'darts_search'
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 10
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = True

    exp.run(exp_config, 8081, debug=True)

    # start weight sharing experiment (i.e., start_weight_sharing_experiment),
    # and users can specify the name of weight sharing trainer without specifying trainer, strategy, and applied mutators.
    # we manage a mapping from weight sharing algorithms to training_approach and strategy