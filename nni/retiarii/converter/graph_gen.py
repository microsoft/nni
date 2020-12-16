import logging
import re

import torch

from ..graph import Graph, Model, Node
from ..nn.pytorch import InputChoice, LayerChoice, Placeholder
from ..operation import Cell
from .op_types import MODULE_EXCEPT_LIST, BasicOpsPT, OpTypeName
from .utils import _convert_name, build_full_name

_logger = logging.getLogger(__name__)

global_seq = 0
global_graph_id = 0
modules_arg = None


def _add_edge(ir_graph, node, graph_inputs, node_index, new_node, output_remap, ignore_first=False):
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
    output_remap : Dict
    ignore_first : bool
        if it is true, skip the first input
    """
    is_single_input = (len([_input for _input in node.inputs()]) - (1 if ignore_first else 0)) == 1
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
        elif _input in output_remap:
            assert output_remap[_input].kind() == 'aten::append'
            predecessor_node = output_remap[_input]
            assert predecessor_node in node_index, 'predecessor node: {}'.format(predecessor_node)
            src_node_idx = None
            src_node = node_index[predecessor_node]
            assert isinstance(src_node, Node)
        else:
            predecessor_node = _input.node()
            assert predecessor_node in node_index, 'predecessor node: {}'.format(predecessor_node)
            # find out the index of _input in the outputs of predecessor_node
            predecessor_outputs = [_output for _output in predecessor_node.outputs()]
            if len(predecessor_outputs) == 1:
                idx = None
            else:
                idx = predecessor_outputs.index(_input)
            ir_predecessor_node = node_index[predecessor_node]
            src_node_idx = idx
            assert isinstance(ir_predecessor_node, Node)
            src_node = ir_predecessor_node

        # handle destination node
        dst_node = new_node
        if is_single_input:
            dst_node_idx = None
        else:
            dst_node_idx = new_node_input_idx

        # create edge
        ir_graph.add_edge(head=(src_node, src_node_idx), tail=(dst_node, dst_node_idx))

        new_node_input_idx += 1


def create_prim_constant_node(ir_graph, node, module_name):
    global global_seq
    attrs = {}
    if node.outputsAt(0).toIValue() is not None:
        attrs = {'value': node.outputsAt(0).toIValue()}
    global_seq += 1
    new_node = ir_graph.add_node(build_full_name(module_name, OpTypeName.Constant, global_seq),
                                 node.kind(), attrs)
    return new_node


def handle_prim_attr_node(node):
    assert node.hasAttribute('name')
    attrs = {'name': node.s('name'), 'input': node.inputsAt(0).debugName()}
    return node.kind(), attrs


def _remove_mangle(module_type_str):
    return re.sub('\\.___torch_mangle_\\d+', '', module_type_str)


def remove_unconnected_nodes(ir_graph, targeted_type=None):
    """
    Parameters
    ----------
    ir_graph : Graph
        our ir graph representation
    targeted_type : str
        nodes with ```targeted_type``` will be removed from graph if their fanout is 0.
        ```None``` means removing all the nodes whose fanout is 0.
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
            if targeted_type is None:
                to_removes.append(hidden_node)
            elif hidden_node.operation.type == targeted_type:
                to_removes.append(hidden_node)

    for hidden_node in to_removes:
        hidden_node.remove()


def handle_graph_nodes(script_module, sm_graph, module, module_name, ir_model, ir_graph):
    """
    Convert torch script node to our node ir, and build our graph ir

    Parameters
    ----------
    script_module : torch.jit.RecursiveScriptModule
        the torch script of ```module```
    sm_graph : torch._C.Graph
        the graph in torch script
    module : nn.Module
        the targeted pytorch module
    module_name : str
        ```module```'s name
    ir_model : Model
        the whole graph ir
    ir_graph : Graph
        the graph ir of ```module```

    Returns
    -------
    dict
        the mapping from graph node to our graph ir node
    """
    # handle inputs
    graph_inputs = []
    for _input in sm_graph.inputs():
        if _input.debugName() == 'self':
            assert _input.unique() == 0
            continue
        graph_inputs.append(_input)
        # TODO: add scope name
        ir_graph._add_input(_convert_name(_input.debugName()))

    node_index = {}  # graph node to graph ir node

    # some node does not have output but it modifies a variable, for example aten::append
    # %17 : Tensor[] = aten::append(%out.1, %16)
    # %out.1 is updated, and %17 is None
    # we add output to this type of node and connect it to the following node which uses %out.1
    # key: tensor (%out.1), value: node (this node)
    output_remap = {}

    def handle_if_condition(cond_tensor):
        """
        to calculate the condition, we only deal with the following op types by tracing back
        `prim::GetAttr`, `aten::__getitem__`, `prim::Constant`, `aten::eq`

        generate the expression using recursive calls

        NOTE: do not support dynamic graph
        """
        def _generate_expr(tensor):
            if tensor.node().kind() == 'prim::GetAttr':
                return f'({getattr(module, tensor.node().s("name"))})'
            elif tensor.node().kind() == 'aten::__getitem__':
                t = _generate_expr(tensor.node().inputsAt(0))
                idx = _generate_expr(tensor.node().inputsAt(1))
                return f'({t}[{idx}])'
            elif tensor.node().kind() == 'prim::Constant':
                return f'{tensor.toIValue()}'
            elif tensor.node().kind() == 'aten::eq':
                left = _generate_expr(tensor.node().inputsAt(0))
                right = _generate_expr(tensor.node().inputsAt(1))
                return f'({left} == {right})'
            else:
                raise RuntimeError(f'Unsupported op type {tensor.node().kind()} in if condition')
        expr = _generate_expr(cond_tensor)
        return eval(expr)

    def handle_if_node(node):
        """
        Parameters
        ----------
        node : torch._C.Node
            the node from TorchScript graph

        Returns
        -------
        Node
            the created node ir
        """
        # only deal with input of prim::If is constant or attribute for now
        # will support constant expression in future
        inputs = [i for i in node.inputs()]
        assert len(inputs) == 1
        cond = handle_if_condition(inputs[0])
        chosen_block = 0 if cond else 1
        blocks = [block for block in node.blocks()]
        assert len(blocks) == 2
        last_block_node = None
        for node in blocks[chosen_block].nodes():
            last_block_node = handle_single_node(node)
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
            the created node ir
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

                if submodule.inputsAt(0).debugName() == 'self':
                    # module is usually instantiated in __init__.
                    # when calling a module in forward,
                    # prim::GetAttr is used to obtain the module in torch script.
                    # therefore, we do this check for a module. example below:
                    # %25 : __torch__.xxx = prim::GetAttr[name="input_switch"](%self)
                    # %27 : Tensor = prim::CallMethod[name="forward"](%25, %out.1)
                    assert submodule_name in script_module._modules, "submodule_name: {} not in script_module {}".format(
                        submodule_name, script_module._modules.keys())

                    submodule_full_name = build_full_name(module_name, submodule_name)
                    submodule_obj = getattr(module, submodule_name)
                    subgraph, sub_m_attrs = convert_module(script_module._modules[submodule_name],
                                                           submodule_obj,
                                                           submodule_full_name, ir_model)
                else:
                    # %8 : __torch__.nni.retiarii.model_apis.nn.___torch_mangle_37.ModuleList = prim::GetAttr[name="cells"](%self)
                    # %10 : __torch__.darts_model.Cell = prim::GetAttr[name="0"](%8)
                    # %s1.4 : Tensor = prim::CallMethod[name="forward"](%10, %4, %4)
                    if submodule.inputsAt(0).type().name() == 'ModuleList':
                        # handle ModuleList
                        predecessor = submodule.inputsAt(0).node()
                        assert predecessor.kind() == 'prim::GetAttr'
                        assert predecessor.hasAttribute('name')
                        assert predecessor.inputsAt(0).debugName() == 'self'
                        predecessor_name = predecessor.s('name')
                        # FIXME: exchange
                        submodule_full_name = build_full_name(module_name, [submodule_name, predecessor_name])
                        predecessor_obj = getattr(module, predecessor_name)
                        submodule_obj = getattr(predecessor_obj, submodule_name)
                        subgraph, sub_m_attrs = convert_module(script_module._modules[predecessor_name]._modules[submodule_name],
                                                               submodule_obj, submodule_full_name, ir_model)
                    else:
                        raise RuntimeError('Unsupported module case: {}'.format(submodule.inputsAt(0).type().str()))

                # TODO: match subgraph with maintained graphs
                # build cell
                if subgraph is None:
                    # if we do not parse this module's graph, we create Node for this module
                    subcell = ir_graph.add_node(submodule_full_name, submodule_type_str, sub_m_attrs)
                    if isinstance(submodule_obj, Placeholder):
                        subcell.update_label(submodule_obj.label)
                    elif isinstance(submodule_obj, (LayerChoice, InputChoice)):
                        subcell.update_label(sub_m_attrs['label'])
                else:
                    # Graph already created, create Cell for it
                    new_cell = Cell(cell_name=submodule_full_name, parameters=sub_m_attrs)
                    subcell = ir_graph.add_node(submodule_full_name, new_cell)
                node_index[node] = subcell
                # connect the cell into graph
                _add_edge(ir_graph, node, graph_inputs, node_index, subcell, output_remap, ignore_first=True)
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
            func_node = ir_graph.add_node(build_full_name(module_name, func_name, global_seq),
                                          '{}.{}'.format(func_type_str, func_name))
            node_index[node] = func_node
            _add_edge(ir_graph, node, graph_inputs, node_index, func_node, output_remap, ignore_first=True)
        elif node.kind() == 'prim::Constant':
            new_node = create_prim_constant_node(ir_graph, node, module_name)
            node_index[node] = new_node
        elif node.kind() == 'prim::ListConstruct':
            global_seq += 1
            new_node = ir_graph.add_node(build_full_name(module_name, OpTypeName.ListConstruct, global_seq), node.kind())
            node_index[node] = new_node
            _add_edge(ir_graph, node, graph_inputs, node_index, new_node, output_remap)
        elif node.kind() == 'aten::append':
            global_seq += 1
            aten_node = ir_graph.add_node(build_full_name(module_name, BasicOpsPT[node.kind()], global_seq), node.kind())
            node_index[node] = aten_node
            _add_edge(ir_graph, node, graph_inputs, node_index, aten_node, output_remap)
            output_remap[node.inputsAt(0)] = node
        elif node.kind().startswith('aten::'):
            # handle aten::XXX
            global_seq += 1
            aten_node = ir_graph.add_node(build_full_name(module_name, BasicOpsPT[node.kind()], global_seq), node.kind())
            node_index[node] = aten_node
            _add_edge(ir_graph, node, graph_inputs, node_index, aten_node, output_remap)
        elif node.kind() == 'prim::GetAttr':
            node_type, attrs = handle_prim_attr_node(node)
            global_seq += 1
            new_node = ir_graph.add_node(build_full_name(module_name, OpTypeName.Attr, global_seq),
                                         node_type, attrs)
            node_index[node] = new_node
        elif node.kind() == 'prim::If':
            last_block_node = handle_if_node(node)
            # last_block_node is None means no node in the branch block
            node_index[node] = last_block_node
        elif node.kind() == 'prim::Loop':
            # refer to https://gist.github.com/liuzhe-lz/90c35d9dd6fd7f3f32544940151ab186
            raise RuntimeError('Loop has not been supported yet!')
        else:
            raise RuntimeError('Unsupported kind: {}'.format(node.kind()))

        return node_index[node]

    for node in sm_graph.nodes():
        handle_single_node(node)

    return node_index


def merge_aten_slices(ir_graph):
    """
    if there is aten::slice node, merge the consecutive ones together.
    ```x[:, :, 1:, 1:]``` in python code will be converted into 4 node in torch script,
    each node has 5 inputs: tensor, dim, x, y, z (i.e., x:y:z)
    """
    head_slice_nodes = []
    has_slice_node = False
    for node in ir_graph.hidden_nodes:
        if node.operation.type == 'aten::slice':
            has_slice_node = True
            for pred in node.predecessors:
                if pred.operation.type not in ['aten::slice', 'prim::Constant']:
                    head_slice_nodes.append(node)
                    break
    if has_slice_node:
        assert head_slice_nodes

    for head_node in head_slice_nodes:
        slot = 0
        new_slice_node = ir_graph.add_node(build_full_name(head_node.name, 'merged'), OpTypeName.MergedSlice)
        if len(head_node.incoming_edges) == 4:
            # when slice is for one dimension list, there are only 4 inputs, thus merge is not needed
            break
        assert len(head_node.incoming_edges) == 5
        for edge in head_node.incoming_edges:
            edge.tail = new_slice_node
        slot += 5
        node = head_node
        while len(node.successors) == 1 and node.successors[0].operation.type == 'aten::slice':
            suc_node = node.successors[0]
            assert len(suc_node.incoming_edges) == 5
            for edge in suc_node.incoming_edges:
                if edge.tail_slot == 0:
                    edge.remove()
                else:
                    edge.tail = new_slice_node
                    edge.tail_slot = slot + edge.tail_slot - 1
            slot += 4
            ir_graph.hidden_nodes.remove(node)
            node = suc_node

        for edge in node.outgoing_edges:
            edge.head = new_slice_node
        ir_graph.hidden_nodes.remove(node)


def refine_graph(ir_graph):
    """
    Do the following process to simplify graph:
    1. remove unconnected constant node
    2. remove unconnected getattr node
    """
    # some constant is not used, for example, function name as prim::Constant
    remove_unconnected_nodes(ir_graph, targeted_type='prim::Constant')
    remove_unconnected_nodes(ir_graph, targeted_type='prim::GetAttr')
    merge_aten_slices(ir_graph)


def _handle_layerchoice(module):
    global modules_arg

    m_attrs = {}
    candidates = module.candidate_ops
    choices = []
    for cand in candidates:
        assert id(cand) in modules_arg, 'id not exist: {}'.format(id(cand))
        assert isinstance(modules_arg[id(cand)], dict)
        cand_type = '__torch__.' + cand.__class__.__module__ + '.' + cand.__class__.__name__
        choices.append({'type': cand_type, 'parameters': modules_arg[id(cand)]})
    m_attrs[f'choices'] = choices
    m_attrs['label'] = module.label
    return m_attrs


def _handle_inputchoice(module):
    m_attrs = {}
    m_attrs['n_chosen'] = module.n_chosen
    m_attrs['reduction'] = module.reduction
    m_attrs['label'] = module.label
    return m_attrs


def convert_module(script_module, module, module_name, ir_model):
    """
    Convert a module to its graph ir (i.e., Graph) along with its input arguments

    Parameters
    ----------
    script_module : torch.jit.RecursiveScriptModule
        the script module of ```module``` obtained with torch.jit.script
    module : nn.Module
        the targeted module instance
    module_name : str
        the constructed name space of ```module```
    ir_model : Model
        the whole graph ir

    Returns
    -------
    Graph
        the built graph ir from module, ```None``` means do not further parse the module
    dict
        the input arguments of this module
    """
    global global_graph_id
    global modules_arg

    # NOTE: have not supported nested LayerChoice, i.e., a candidate module
    # also has LayerChoice or InputChoice or ValueChoice
    original_type_name = script_module.original_name
    if original_type_name == OpTypeName.LayerChoice:
        m_attrs = _handle_layerchoice(module)
        return None, m_attrs
    if original_type_name == OpTypeName.InputChoice:
        m_attrs = _handle_inputchoice(module)
        return None, m_attrs
    if original_type_name == OpTypeName.Placeholder:
        m_attrs = modules_arg[id(module)]
        return None, m_attrs
    if original_type_name in torch.nn.__dict__ and original_type_name not in MODULE_EXCEPT_LIST:
        # this is a basic module from pytorch, no need to parse its graph
        assert id(module) in modules_arg, f'{original_type_name} arguments are not recorded'
        m_attrs = modules_arg[id(module)]
        return None, m_attrs

    # handle TorchScript graph
    sm_graph = script_module.graph
    global_graph_id += 1
    ir_graph = Graph(model=ir_model, graph_id=global_graph_id, name=module_name, _internal=True)

    # handle graph nodes
    node_index = handle_graph_nodes(script_module, sm_graph, module,
                                    module_name, ir_model, ir_graph)

    # handle graph outputs
    for _output in sm_graph.outputs():
        ir_graph._add_output(_convert_name(_output.debugName()))
        predecessor_node_outputs = [o for o in _output.node().outputs()]
        if len(predecessor_node_outputs) == 1:
            src_node_idx = None
        else:
            src_node_idx = predecessor_node_outputs.index(_output)
        ir_graph.add_edge(head=(node_index[_output.node()], src_node_idx),
                          tail=(ir_graph.output_node, None))

    refine_graph(ir_graph)

    ir_graph._register()

    if id(module) not in modules_arg:
        raise RuntimeError(f'{original_type_name} arguments are not recorded, \
            you might have forgotten to decorate this class with @register_module()')
    # TODO: if we parse this module, it means we will create a graph (module class)
    # for this module. Then it is not necessary to record this module's arguments
    # return ir_graph, modules_arg[id(module)].
    # That is, we can refactor this part, to allow users to annotate which module
    # should not be parsed further.
    return ir_graph, {}


def convert_to_graph(script_module, module, recorded_modules_arg):
    """
    Convert module to our graph ir, i.e., build a ```Model``` type

    Parameters
    ----------
    script_module : torch.jit.RecursiveScriptModule
        the script module obtained with torch.jit.script
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
    convert_module(script_module, module, module_name, model)

    return model
