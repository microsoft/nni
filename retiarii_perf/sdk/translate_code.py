import os
import re
import inspect
from shutil import copyfile
from importlib import import_module

import torch

from .pytorch_graph._graph_utils import TorchModuleGraph

class TranslateCode:
    def __init__(self):
        pass

def inspect_module_args(module):
    """

    Parameters
    ----------
    module : nn.Module
        pytorch module

    Returns
    -------
    dict
        dict of parameters and values
    """
    argspec = inspect.getfullargspec(type(module))
    parameters = {}
    for arg in argspec.args:
        if arg == 'self':
            continue
        # specifically handle bias
        if arg == 'bias':
            if isinstance(getattr(module, 'bias'), torch.Tensor):
                parameters['bias'] = True
        else:
            parameters[arg] = getattr(module, arg)
    return parameters

def module_graph_to_ir(base_model, module_graph):
    """
    convert the module_graph object to our graph IR

    Returns
    -------
    json obj
        json format of our graph IR
    """
    # TODO: consider moving this name convertion to code_gen rather than here
    def _format_name(name):
        return re.sub('[^0-9a-zA-Z_]', '__', name)

    def _format_node(name, op_type, parameters):
        return { "name": _format_name(name), "operation": { "type": op_type, **parameters } }

    def _format_edge(head, tail):
        return { "head": _format_name(head), "tail": _format_name(tail) }

    # TODO: check whether works for submodule
    name_to_module = {}
    for name, module in base_model.named_modules():
        print('module name: ', name)
        name_to_module[name] = module
    
    # fill hidden_nodes
    hidden_nodes = []
    for name, node_pygroup in module_graph.name_to_node.items():
        op_type = node_pygroup.op_type
        if node_pygroup.type == 'module':
            assert name in name_to_module, '{} does not exist in name_to_module'.format(name)
            parameters = inspect_module_args(name_to_module[name])
        elif node_pygroup.type == 'func':
            # here parameters is a list
            parameters = { 'input_arguments': node_pygroup.auxiliary['input_args'] }
        else:
            raise RuntimeError('unknown node_pygroup: {}'.format(node_pygroup))
        node_json = _format_node(name, op_type, parameters)
        hidden_nodes.append(node_json)
    
    # fill inputs and outputs
    inputs = module_graph.get_input_nodes()
    inputs_name_convert = { ori: re.sub('[^0-9a-zA-Z_]', '_', ori) for ori in inputs }
    inputs_json = [ { 'name': inputs_name_convert[name] } for name in inputs ]
    outputs = module_graph.get_output_nodes()
    outputs_name_convert = { ori: 'output_' + ori for ori in outputs}
    outputs_json = [ { 'name': outputs_name_convert[name] } for name in outputs ]

    # fill edges
    edges = []
    for name, node_pygroup in module_graph.name_to_node.items():
        head = name
        for output in node_pygroup.outputs:
            if output in module_graph.input_to_node:
                for each in module_graph.input_to_node[output]:
                    tail = each.unique_name
                    edge_json = _format_edge(head, tail)
                    edges.append(edge_json)
            else:
                if output not in outputs:
                    print('zql output: ', output)
                assert output in outputs
                tail = outputs_name_convert[output]
                edge_json = _format_edge(head, tail)
                edges.append(edge_json)
    for _input in inputs:
        head = _input
        assert _input in module_graph.input_to_node
        for each in module_graph.input_to_node[_input]:
            tail = each.unique_name
            edge_json = _format_edge(inputs_name_convert[head], tail)
            edges.append(edge_json)

    # complete whole graph json
    graph_json = {'graph': {}}
    graph_json['graph']['inputs'] = inputs_json
    graph_json['graph']['outputs'] = outputs_json
    graph_json['graph']['hidden_nodes'] = hidden_nodes
    graph_json['graph']['edges'] = edges

    print('graph json: ', graph_json)

    return graph_json

def hack_for_spos(module_graph, root_tnode):
    """
    replace features.x subgraph with one simple graph node
    """
    to_remove_nodes = {}
    new_nodes = []
    for i in range(20):
        name_slices = ['features', str(i)]
        name = '.'.join(name_slices)
        # get all nodes under this scope name
        subgraph_nodes = {}
        for unique_name, node_pygroup in module_graph.name_to_node.items():
            target_slices = node_pygroup.name.split('.')
            if target_slices[:2] == name_slices:
                subgraph_nodes[unique_name] = node_pygroup
                to_remove_nodes[unique_name] = node_pygroup
        # get predecessor or successor nodes
        #predecessors = []
        #successors = []
        subgraph_inputs = set() # should be set, as there might be several same inputs
        subgraph_outputs = set()
        for unique_name, node_pygroup in subgraph_nodes.items():
            nodes = module_graph.find_predecessors(unique_name)
            for _unique_name in nodes:
                if _unique_name not in subgraph_nodes:
                    #predecessors.append(node)
                    assert len(module_graph.name_to_node[_unique_name].outputs) == 1
                    subgraph_inputs.add(module_graph.name_to_node[_unique_name].outputs[0])
            nodes = module_graph.find_successors(unique_name)
            #print('subgraph nodes: ', subgraph_nodes)
            #print(name, unique_name, nodes)
            for _unique_name in nodes:
                #print('successor: ', unique_name)
                if _unique_name not in subgraph_nodes:
                    #successors.append(node)
                    #if len(module_graph.name_to_node[_unique_name].inputs) != 1:
                    #    print('assert: ', module_graph.name_to_node[_unique_name].inputs)
                    #for each in module_graph.name_to_node[_unique_name].inputs:
                    #    if each == 'input.179':
                    #        print('zql: ', name, unique_name, nodes)
                    if len(module_graph.name_to_node[_unique_name].inputs) == 1:
                        subgraph_outputs.add(module_graph.name_to_node[_unique_name].inputs[0])
                        #print('inputs: ', module_graph.name_to_node[_unique_name].inputs)
                    else:
                        d = set(node_pygroup.outputs).intersection(set(module_graph.name_to_node[_unique_name].inputs))
                        assert len(d) == 1
                        for each in d:
                            subgraph_outputs.add(each)
        # create one node
        from .pytorch_graph._graph_utils import NodePyGroup
        subgraph = NodePyGroup(name, name, 'module', 'ShuffleNetBlock', [], subgraph_inputs, subgraph_outputs)
        new_nodes.append(subgraph)
    # remove nodes and add new nodes, then rebuild index of nodes
    new_node_set = []
    for unique_name, node_pygroup in module_graph.name_to_node.items():
        if unique_name not in to_remove_nodes:
            new_node_set.append(node_pygroup)
    new_node_set.extend(new_nodes)
    print(new_node_set)
    #exit(1)
    module_graph.name_to_node, module_graph.input_to_node, module_graph.output_to_node = module_graph._build_index(new_node_set)

def hack_for_textnas(module_graph, root_tnode):
    """
    replace features.x subgraph with one simple graph node
    """
    to_remove_nodes = {}
    new_nodes = []
    for i in range(24):
        name_slices = ['layers', str(i), 'op']
        name = '.'.join(name_slices)
        # get all nodes under this scope name
        subgraph_nodes = {}
        for unique_name, node_pygroup in module_graph.name_to_node.items():
            target_slices = node_pygroup.name.split('.')
            if target_slices[:3] == name_slices:
                subgraph_nodes[unique_name] = node_pygroup
                to_remove_nodes[unique_name] = node_pygroup
        # get predecessor or successor nodes
        #predecessors = []
        #successors = []
        subgraph_inputs = set() # should be set, as there might be several same inputs
        subgraph_outputs = set()
        for unique_name, node_pygroup in subgraph_nodes.items():
            nodes = module_graph.find_predecessors(unique_name)
            for _unique_name in nodes:
                if _unique_name not in subgraph_nodes:
                    #predecessors.append(node)
                    assert len(module_graph.name_to_node[_unique_name].outputs) == 1
                    subgraph_inputs.add(module_graph.name_to_node[_unique_name].outputs[0])
            for _input in module_graph.name_to_node[unique_name].inputs:
                if _input in module_graph.get_input_nodes():
                    subgraph_inputs.add(_input)
            nodes = module_graph.find_successors(unique_name)
            #print('subgraph nodes: ', subgraph_nodes)
            #print(name, unique_name, nodes)
            for _unique_name in nodes:
                #print('successor: ', unique_name)
                if _unique_name not in subgraph_nodes:
                    #successors.append(node)
                    #if len(module_graph.name_to_node[_unique_name].inputs) != 1:
                    #    print('assert: ', module_graph.name_to_node[_unique_name].inputs)
                    #for each in module_graph.name_to_node[_unique_name].inputs:
                    #    if each == 'input.179':
                    #        print('zql: ', name, unique_name, nodes)
                    if len(module_graph.name_to_node[_unique_name].inputs) == 1:
                        subgraph_outputs.add(module_graph.name_to_node[_unique_name].inputs[0])
                        #print('inputs: ', module_graph.name_to_node[_unique_name].inputs)
                    else:
                        d = set(node_pygroup.outputs).intersection(set(module_graph.name_to_node[_unique_name].inputs))
                        assert len(d) == 1
                        for each in d:
                            subgraph_outputs.add(each)
        # create one node
        from .pytorch_graph._graph_utils import NodePyGroup
        subgraph = NodePyGroup(name, name, 'module', 'WrapperOp', [], subgraph_inputs, subgraph_outputs)
        new_nodes.append(subgraph)
    # remove nodes and add new nodes, then rebuild index of nodes
    new_node_set = []
    for unique_name, node_pygroup in module_graph.name_to_node.items():
        if unique_name not in to_remove_nodes:
            new_node_set.append(node_pygroup)
    new_node_set.extend(new_nodes)
    print(new_node_set)
    #exit(1)
    module_graph.name_to_node, module_graph.input_to_node, module_graph.output_to_node = module_graph._build_index(new_node_set)

def hack_for_nasnet(module_graph, root_tnode):
    """
    replace features.x subgraph with one simple graph node
    """
    to_remove_nodes = {}
    new_nodes = []
    for i in range(16):
        if i < 12:
            name_slices = ['cell_'+str(i)]
        elif i == 12:
            name_slices = ['cell_stem_0']
        elif i == 13:
            name_slices = ['cell_stem_1']
        elif i == 14:
            name_slices = ['reduction_cell_0']
        elif i == 15:
            name_slices = ['reduction_cell_1']
        else:
            raise
        name = '.'.join(name_slices)
        # get all nodes under this scope name
        subgraph_nodes = {}
        for unique_name, node_pygroup in module_graph.name_to_node.items():
            target_slices = node_pygroup.name.split('.')
            if target_slices[0] == name_slices[0]:
                subgraph_nodes[unique_name] = node_pygroup
                to_remove_nodes[unique_name] = node_pygroup
        # get predecessor or successor nodes
        #predecessors = []
        #successors = []
        subgraph_inputs = set() # should be set, as there might be several same inputs
        subgraph_outputs = set()
        for unique_name, node_pygroup in subgraph_nodes.items():
            nodes = module_graph.find_predecessors(unique_name)
            for _unique_name in nodes:
                if _unique_name not in subgraph_nodes:
                    #predecessors.append(node)
                    assert len(module_graph.name_to_node[_unique_name].outputs) == 1
                    subgraph_inputs.add(module_graph.name_to_node[_unique_name].outputs[0])
            for _input in module_graph.name_to_node[unique_name].inputs:
                if _input in module_graph.get_input_nodes():
                    subgraph_inputs.add(_input)
            nodes = module_graph.find_successors(unique_name)
            #print('subgraph nodes: ', subgraph_nodes)
            #print(name, unique_name, nodes)
            for _unique_name in nodes:
                #print('successor: ', unique_name)
                if _unique_name not in subgraph_nodes:
                    #successors.append(node)
                    #if len(module_graph.name_to_node[_unique_name].inputs) != 1:
                    #    print('assert: ', module_graph.name_to_node[_unique_name].inputs)
                    #for each in module_graph.name_to_node[_unique_name].inputs:
                    #    if each == 'input.179':
                    #        print('zql: ', name, unique_name, nodes)
                    if len(module_graph.name_to_node[_unique_name].inputs) == 1:
                        subgraph_outputs.add(module_graph.name_to_node[_unique_name].inputs[0])
                        #print('inputs: ', module_graph.name_to_node[_unique_name].inputs)
                    else:
                        d = set(node_pygroup.outputs).intersection(set(module_graph.name_to_node[_unique_name].inputs))
                        assert len(d) == 1
                        for each in d:
                            subgraph_outputs.add(each)
        # create one node
        from .pytorch_graph._graph_utils import NodePyGroup
        if name_slices[0] == 'cell_stem_0':
            subgraph = NodePyGroup(name, name, 'module', 'CellStem0', [], subgraph_inputs, subgraph_outputs)
        elif name_slices[0] == 'cell_stem_1':
            subgraph = NodePyGroup(name, name, 'module', 'CellStem1', [], subgraph_inputs, subgraph_outputs)
        else:
            subgraph = NodePyGroup(name, name, 'module', 'Cell', [], subgraph_inputs, subgraph_outputs)
        new_nodes.append(subgraph)
    # remove nodes and add new nodes, then rebuild index of nodes
    new_node_set = []
    for unique_name, node_pygroup in module_graph.name_to_node.items():
        if unique_name not in to_remove_nodes:
            new_node_set.append(node_pygroup)
    new_node_set.extend(new_nodes)
    print(new_node_set)
    #exit(1)
    module_graph.name_to_node, module_graph.input_to_node, module_graph.output_to_node = module_graph._build_index(new_node_set)


def gen_pytorch_graph(base_model, dummy_input=None, name=''):
    assert dummy_input is not None
    module_graph = TorchModuleGraph(base_model, dummy_input)
    root_tnode = module_graph.build_module_hierarchy()
    if 'spos' in name:
        hack_for_spos(module_graph, root_tnode)
    elif 'textnas' in name:
        hack_for_textnas(module_graph, root_tnode)
    elif 'mnasnet' not in name and 'nasnet' in name:
        hack_for_nasnet(module_graph, root_tnode)
    return module_graph_to_ir(base_model, module_graph)
