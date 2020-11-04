import json_tricks
import re
import torch

from ..graph import Graph, Node, Edge, Model
from ..operation import Cell, Operation

from .op_types import RETIARII_BASE_OPS, MODULE_EXCEPT_LIST, Type
from .utils import build_full_name


global_seq = 0
global_graph_id = 0
modules_arg = None

def _add_edge(ir_graph, node, graph_inputs, node_index, new_node, ignore_first=False):
    """
    Parameters
    ----------
    ir_graph : Graph
    node : torch._C.Node
    graph_inputs : List[torch._C.Value]
        a list of a script graph's inputs
    node_index : Dict
    new_node : Node
        newly created ir node corresponding to `node`
    ignore_first : bool
        if it is true, skip the first input
    """
    new_node_input_idx = 0
    for _input in node.inputs():
        if ignore_first:
            ignore_first = False
            continue

        # handle source node
        if _input in graph_inputs:
            idx = graph_inputs.index(_input)
            src_node = ir_graph.input_node
            src_node_idx = idx
        else:
            predecessor_node = _input.node()
            assert predecessor_node in node_index, 'predecessor node: {}'.format(predecessor_node)
            # find out the index of _input in the outputs of predecessor_node
            predecessor_outputs = [_output for _output in predecessor_node.outputs()]
            idx = predecessor_outputs.index(_input)
            ir_predecessor_node = node_index[predecessor_node]
            src_node_idx = idx
            # get source node
            # the input is output of a basic node
            assert isinstance(ir_predecessor_node, Node)
            src_node = ir_predecessor_node

        # handle destination node
        dst_node = new_node
        dst_node_idx = new_node_input_idx

        # create edge
        ir_graph.add_edge(head=(src_node, src_node_idx), tail=(dst_node, dst_node_idx))

        new_node_input_idx += 1

def _handle_inputs(ir_graph, node, graph_inputs, node_index, module_name, ignore_first=False):
    """
    create prim::GetAttr node when necessary. because for some cases prim::GetAttr nodes are removed,
    for example, the prim::GetAttr used in prim::CallMethod
    """
    global global_seq
    for _input in node.inputs():
        # for CallMethod and CallFunction
        if ignore_first:
            ignore_first = False
            continue
        if _input in graph_inputs:
            continue
        if _input.node().kind() == 'prim::Constant':
            assert _input.node() in node_index
        if _input.node().kind() == 'prim::GetAttr':
            if _input.node() not in node_index:
                node_type, attrs = handle_prim_attr_node(_input.node())
                global_seq += 1
                new_node = ir_graph.add_node(build_full_name(module_name, Type.Attr, global_seq),
                                             node_type,
                                             **attrs)
                node_index[_input.node()] = new_node
                print('==handle inputs getattr==: ', _input.node())

def create_prim_constant_node(ir_graph, node, module_name):
    global global_seq
    attrs = {}
    if node.outputsAt(0).toIValue() is not None:
        attrs = {'value': node.outputsAt(0).toIValue()}
    global_seq += 1
    new_node = ir_graph.add_node(build_full_name(module_name, Type.Constant, global_seq),
                                 node.kind(),
                                 **attrs)
    return new_node

def handle_prim_attr_node(node):
    assert node.hasAttribute('name')
    assert node.inputsAt(0).debugName() == 'self'
    assert node.inputsAt(0).unique() == 0
    attrs = {'name': node.s('name'), 'input': node.inputsAt(0).debugName()}
    return node.kind(), attrs

def _remove_mangle(module_type_str):
    return re.sub('\\.___torch_mangle_\\d+', '', module_type_str)

def remove_unconnected_nodes(ir_graph):
    """
    Parameters
    ----------
    ir_graph : Graph
        our ir graph representation
    """
    # build index of outputs of Node(s)
    node_fanout = set()
    for edge in ir_graph.edges:
        if edge.head.id not in node_fanout:
            node_fanout.add(edge.head.id)
    to_removes = []
    for hidden_node in ir_graph.hidden_nodes:
        if hidden_node.id not in node_fanout:
            assert isinstance(hidden_node, Node)
            to_removes.append(hidden_node)
            # some constant is not used, for example, function name as prim::Constant
            assert hidden_node.operation.type == 'prim::Constant', 'the type is {}'.format(hidden_node.operation.type)
    for hidden_node in to_removes:
        hidden_node.remove()

def handle_graph_nodes(script_module, sm_graph, module, module_name, ir_model, ir_graph):
    """
    Parameters
    ----------
    script_module : torch.jit.RecursiveScriptModule
    sm_graph : torch._C.Graph
    module : nn.Module
    module_name : str
    ir_model : Model
    ir_graph : Graph
    """
    # handle inputs
    graph_inputs = []
    for _input in sm_graph.inputs():
        if _input.debugName() == 'self':
            assert _input.unique() == 0
            continue
        graph_inputs.append(_input)
        # TODO: add scope name
        ir_graph._add_input(_input.debugName())

    node_index = {} # graph node to graph ir node

    def handle_if_node(node):
        """
        Parameters
        ----------
        node : torch._C.Node
            the node from TorchScript graph

        Returns
        -------
        Node
            the created ir node
        """
        # only deal with input of prim::If is constant or attribute for now
        # TODO: support constant expression
        inputs = [i for i in node.inputs()]
        assert len(inputs) == 1
        if not inputs[0].node().kind() in ['prim::Constant', 'prim::GetAttr']:
            raise RuntimeError('"if" whose condition is not constant or attribute has not been supported yet!')
        chosen_block = None
        if inputs[0].node().kind() == 'prim::Constant':
            chosen_block = 0 if inputs[0].toIValue() else 1
        if inputs[0].node().kind() == 'prim::GetAttr':
            chosen_block = 0 if getattr(module, inputs[0].node().s('name')) else 1
        blocks = [block for block in node.blocks()]
        assert len(blocks) == 2
        last_block_node = None
        for node in blocks[chosen_block].nodes():
            last_block_node = handle_single_node(node)
        assert last_block_node is not None
        return last_block_node

    def handle_single_node(node):
        """
        Parameters
        ----------
        node : torch._C.Node
            the node from TorchScript graph

        Returns
        -------
        Node
            the created ir node
        """
        global global_seq
        if node.kind() == 'prim::CallMethod':
            # get and handle the first input, which should be an nn.Module
            assert node.hasAttribute('name')
            if node.s('name') == 'forward':
                # node.inputsAt(0).type() is <class 'torch._C.ClassType'>
                submodule_type_str = _remove_mangle(node.inputsAt(0).type().str())
                submodule = node.inputsAt(0).node()
                assert submodule.kind() == 'prim::GetAttr'
                assert submodule.hasAttribute('name')
                submodule_name = submodule.s('name')
                assert submodule_name in script_module._modules

                submodule_full_name = build_full_name(module_name, submodule_name)
                subgraph, sub_m_attrs = convert_module(script_module._modules[submodule_name],
                                                       getattr(module, submodule_name),
                                                       submodule_full_name, ir_model)
                # TODO: try not-connected placeholder in TorchScript
                # TODO: match subgraph with maintained graphs
                # build cell
                if subgraph is None:
                    # if we do not parse this module's graph, we create Node for this module
                    subcell = ir_graph.add_node(name=submodule_full_name, type=submodule_type_str, **sub_m_attrs)
                else:
                    # Graph already created, create Cell for it
                    new_cell = Cell(cell_name=submodule_full_name, parameters=sub_m_attrs)
                    subcell = ir_graph.add_node(name=submodule_full_name, type=new_cell)
                node_index[node] = subcell
                _handle_inputs(ir_graph, node, graph_inputs, node_index, module_name, ignore_first=True)
                # connect the cell into graph
                _add_edge(ir_graph, node, graph_inputs, node_index, subcell, ignore_first=True)
            else:
                raise RuntimeError('unsupported CallMethod {}'.format(node.s('name')))
        elif node.kind() == 'prim::CallFunction':
            func_type_str = _remove_mangle(node.inputsAt(0).type().str())
            func = node.inputsAt(0).node()
            assert func.kind() == 'prim::Constant'
            assert func.hasAttribute('name')
            func_name = func.s('name')
            # create node for func
            global_seq += 1
            func_node = ir_graph.add_node(name=build_full_name(module_name, func_name, global_seq), type=func_type_str)
            node_index[node] = func_node
            _handle_inputs(ir_graph, node, graph_inputs, node_index, module_name, ignore_first=True)
            _add_edge(ir_graph, node, graph_inputs, node_index, func_node, ignore_first=True)
        elif node.kind() == 'prim::Constant':
            # TODO: how about calling a function twice? two constant nodes or one?
            new_node = create_prim_constant_node(ir_graph, node, module_name)
            node_index[node] = new_node
        elif node.kind() == 'prim::ListConstruct':
            global_seq += 1
            new_node = ir_graph.add_node(build_full_name(module_name, Type.ListConstruct, global_seq), node.kind())
            node_index[node] = new_node
            _handle_inputs(ir_graph, node, graph_inputs, node_index, module_name)
            _add_edge(ir_graph, node, graph_inputs, node_index, new_node)
        elif node.kind().startswith('aten::'):
            # handle aten::XXX
            global_seq += 1
            aten_node = ir_graph.add_node(build_full_name(module_name, Type.BasicOpsPT[node.kind()], global_seq), node.kind())
            node_index[node] = aten_node
            _handle_inputs(ir_graph, node, graph_inputs, node_index, module_name)
            _add_edge(ir_graph, node, graph_inputs, node_index, aten_node)
        elif node.kind() == 'prim::Loop':
            raise RuntimeError('Loop has not been supported yet!')
        elif node.kind() == 'prim::If':
            last_block_node = handle_if_node(node)
            node_index[node] = last_block_node
        elif node.kind() == 'prim::GetAttr':
            pass
        else:
            raise RuntimeError('Unsupported kind: {}'.format(node.kind()))

        if node in node_index:
            return node_index[node]
        else:
            return None

    for node in sm_graph.nodes():
        handle_single_node(node)

    return node_index

def convert_module(script_module, module, module_name, ir_model):
    global global_graph_id
    global modules_arg

    assert id(module) in modules_arg, 'id not exist: {}, {}'.format(id(module), module_name)
    if isinstance(modules_arg[id(module)], tuple):
        positional_args, keyword_args = modules_arg[id(module)]
        m_attrs = keyword_args
        m_attrs['positional_args'] = positional_args
    else:
        m_attrs = modules_arg[id(module)]

    original_type_name = script_module.original_name
    if original_type_name in torch.nn.__dict__ and original_type_name not in MODULE_EXCEPT_LIST:
        # this is a basic module from pytorch, no need to parse its graph
        return None, m_attrs
    if original_type_name in RETIARII_BASE_OPS:
        return None, m_attrs

    # handle TorchScript graph
    sm_graph = script_module.graph
    global_graph_id += 1
    ir_graph = Graph(model=ir_model, graph_id=global_graph_id, name=module_name, _internal=True)

    # handle graph nodes
    node_index = handle_graph_nodes(script_module, sm_graph, module,
                                    module_name, ir_model, ir_graph)

    # handle graph outputs
    graph_outputs = []
    for _output in sm_graph.outputs():
        graph_outputs.append(_output) # <class 'torch._C.Value'>
        ir_graph._add_output(_output.debugName())
        predecessor_node_outputs = [o for o in _output.node().outputs()]
        src_node_idx = predecessor_node_outputs.index(_output)
        #edge = Edge(node_index[_output.node()], output_node, src_node_idx, 0)
        print('===: ', _output.node())
        print(script_module)
        ir_graph.add_edge(head=(node_index[_output.node()], src_node_idx),
                          tail=(ir_graph.output_node, 0))

    remove_unconnected_nodes(ir_graph)

    ir_graph._register()

    return ir_graph, m_attrs

def convert_to_graph(script_module, module, recorded_modules_arg):
    """
    Parameters
    ----------
    script_module : torch.jit.RecursiveScriptModule
        the script module obtain with torch.jit.script
    module : nn.Module
        the targeted module instance
    recorded_modules_arg : dict
        the recorded args of each module in the module

    Returns
    Model
        the constructed IR model
    """
    global modules_arg
    modules_arg = recorded_modules_arg

    model = Model(_internal=True)
    module_name = '_model'
    graph, m_attrs = convert_module(script_module, module, module_name, model)

    return model
