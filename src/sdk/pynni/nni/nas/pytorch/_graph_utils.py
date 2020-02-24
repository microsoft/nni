# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file

# This file is copied from PyTorch 1.4, with bug fixes.
# Likely to be removed in future.

import torch
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef
from torch.utils.tensorboard._pytorch_graph import GraphPy, CLASSTYPE_KIND, GETATTR_KIND, NodePyIO, NodePyOP


def parse(graph, trace, args=None, omit_useless_nodes=True):
    """This method parses an optimized PyTorch model graph and produces
    a list of nodes and node stats for eventual conversion to TensorBoard
    protobuf format.

    Args:
      graph (PyTorch module): The model graph to be parsed.
      trace (PyTorch JIT TracedModule): The model trace to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    n_inputs = len(args)

    scope = {}
    nodes_py = GraphPy()
    for node in graph.inputs():
        if omit_useless_nodes:
            if len(node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
                continue

        if node.type().kind() != CLASSTYPE_KIND:
            nodes_py.append(NodePyIO(node, 'input'))

    attr_to_scope = dict()
    node_to_name = lambda d: str(d).split(":")[0].strip()
    for node in graph.nodes():
        if node.kind() == GETATTR_KIND:
            attr_name = node.s('name')
            node_name = node_to_name(node)
            parent = node.input().node()
            if parent.kind() == GETATTR_KIND:  # If the parent node is not the top-level "self" node
                parent_attr_name = parent.s('name')
                parent_scope = attr_to_scope[node_to_name(parent)]
                attr_scope = parent_scope.split('/')[-1]
                attr_to_scope[node_name] = '{}/{}.{}'.format(parent_scope, attr_scope, attr_name)
            else:
                attr_to_scope[node_name] = '__module.{}'.format(attr_name)
            # We don't need classtype nodes; scope will provide this information
            if node.output().type().kind() != CLASSTYPE_KIND:
                node_py = NodePyOP(node)
                node_py.scopeName = attr_to_scope[node_name]
                nodes_py.append(node_py)
        else:
            nodes_py.append(NodePyOP(node))

    for i, node in enumerate(graph.outputs()):  # Create sink nodes for output ops
        node_py = NodePyIO(node, 'output')
        node_py.debugName = "output.{}".format(i + 1)
        node_py.inputs = [node.debugName()]
        nodes_py.append(node_py)

    def parse_traced_name(module_name):
        prefix = 'TracedModule['
        suffix = ']'
        if module_name.startswith(prefix) and module_name.endswith(suffix):
            module_name = module_name[len(prefix):-len(suffix)]
        return module_name

    alias_to_name = dict()
    base_name = parse_traced_name(trace._name)
    for name, module in trace.named_modules(prefix='__module'):
        mod_name = parse_traced_name(module._name)
        attr_name = name.split('.')[-1]
        alias_to_name[name] = '{}[{}]'.format(mod_name, attr_name)

    for node in nodes_py.nodes_op:
        module_aliases = node.scopeName.split('/')[-1].split('.')
        module_name = ''
        for i, alias in enumerate(module_aliases):
            if i == 0:
                module_name = alias
                node.scopeName = base_name
            else:
                module_name += '.' + alias
                node.scopeName += '/' + (alias_to_name[module_name] if module_name in alias_to_name else alias)

    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py.to_proto()


def graph(model, args, verbose=False):
    """
    This method processes a PyTorch model and produces a `GraphDef` proto
    that can be logged to TensorBoard.

    Args:
      model (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      verbose (bool): Whether to print out verbose information while
        processing.
    """
    with torch.onnx.set_training(model, False):  # TODO: move outside of torch.onnx?
        try:
            trace = torch.jit.trace(model, args)
            graph = trace.graph
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e

    if verbose:
        print(graph)
    list_of_nodes = parse(graph, trace, args)
    # We are hardcoding that this was run on CPU even though it might have actually
    # run on GPU. Note this is what is shown in TensorBoard and has no bearing
    # on actual execution.
    # TODO: See if we can extract GPU vs CPU information from the PyTorch model
    # and pass it correctly to TensorBoard.
    #
    # Definition of StepStats and DeviceStepStats can be found at
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/graph/tf_graph_common/test/graph-test.ts
    # and
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/step_stats.proto
    stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")]))
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
    # The producer version has been reverse engineered from standard
    # TensorBoard logged data.
