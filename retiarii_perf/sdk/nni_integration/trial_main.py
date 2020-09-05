import importlib
import nni
import os
import sys
import torch
import torch.nn.functional as F

from ..graph import Graph
from .. import codegen, utils

def trial_main():
    graph_dict = nni.get_next_parameter()
    graph_id = nni.get_trial_id()
    graph = Graph.load(graph_dict)
    
    graph.generate_code('pytorch', output_file=f'generated/graph_{graph_id}.py')
    graph_cls = utils.import_(f'generated.graph_{graph_id}.Graph')

    full_file_path = graph.training['file']
    file_name = os.path.basename(full_file_path)
    dir_name = os.path.dirname(full_file_path)
    sys.path.append(dir_name)
    module_name = os.path.splitext(file_name)[0]
    class_module = importlib.import_module(module_name)
    classname = graph.training['func']
    class_constructor = getattr(class_module, classname)
    args = graph.training['args']
    kwargs = graph.training['kwargs']
    model = graph_cls()
    training_instance = class_constructor(*args, **kwargs)
    training_instance.bind_model(model)
    optimizer = training_instance.configure_optimizer()
    training_instance.set_optimizer(optimizer)
    training_instance.training_logic()

trial_main()
