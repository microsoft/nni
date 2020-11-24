import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from nni.retiarii.converter.graph_gen import convert_to_graph
from nni.retiarii.converter.visualize import visualize_model
from nni.retiarii import nn
from nni.retiarii.codegen.pytorch import model_to_pytorch_script

from mnist_model import _model
from nni.experiment import Experiment

if __name__ == '__main__':
    nn.enable_record_args()
    base_model = _model()
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

