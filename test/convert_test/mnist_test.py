import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from nni.retiarii.converter.graph_gen import convert_to_graph
from nni.retiarii.converter.visualize import visualize_model
from nni.retiarii import nn
from nni.retiarii.codegen.pytorch import model_to_pytorch_script

from nni.retiarii.utils import TraceClassArguments

from mnist_model import _model
from nni.experiment import Experiment

if __name__ == '__main__':
    with TraceClassArguments() as tca:
        base_model = _model()
    script_module = torch.jit.script(base_model)
    model = convert_to_graph(script_module, base_model, tca.recorded_arguments)
    code_script = model_to_pytorch_script(model)
    print(code_script)
    '''print("Model: ", model)
    graph_ir = model._dump()
    print(graph_ir)
    visualize_model(graph_ir)'''

    # TODO: new interface
    #exp = Experiment()
    #exp.start_retiarii_experiment(base_model, training_approach,
    #                              applied_mutators, strategy,
    #                              exp_config)

