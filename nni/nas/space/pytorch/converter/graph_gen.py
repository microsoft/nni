# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

import torch

from nni.nas.space.graph import Graph, GraphModelSpace, Node, Cell, Operation
from nni.nas.nn.pytorch import InputChoice, MutationAnchor, LayerChoice, MutableModule, Repeat
from nni.nas.utils import get_init_parameters_or_fail, get_importable_name
from .op_types import MODULE_EXCEPT_LIST, OpTypeName
from .utils import (
    _convert_name, build_full_name, _without_shape_info,
    _extract_info_from_trace_node, get_full_name_by_scope_name,
    is_layerchoice_node, match_node, build_cand_name,
    build_python_name
)


class GraphConverter:
    def __init__(self):
        self.global_seq = 0
        self.global_graph_id = 0

    def _add_edge_handle_source_node(self, _input, graph_inputs, ir_graph, output_remap, node_index):
        if _input in output_remap:
            assert output_remap[_input].kind() == 'aten::append'
            predecessor_node = output_remap[_input]
            assert predecessor_node in node_index, 'predecessor node: {}'.format(predecessor_node)
            src_node_idx = None
            src_node = node_index[predecessor_node]
            assert isinstance(src_node, Node)
        elif _input in graph_inputs:
            idx = graph_inputs.index(_input)
            src_node = ir_graph.input_node
            src_node_idx = idx
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
        return src_node, src_node_idx

    def _add_edge(self, ir_graph, node, graph_inputs, node_index, new_node, output_remap, ignore_first=False):
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
            src_node, src_node_idx = self._add_edge_handle_source_node(_input, graph_inputs, ir_graph, output_remap, node_index)
            # handle destination node
            dst_node = new_node
            if is_single_input:
                dst_node_idx = None
            else:
                dst_node_idx = new_node_input_idx
            # create edge
            ir_graph.add_edge(head=(src_node, src_node_idx), tail=(dst_node, dst_node_idx))

            new_node_input_idx += 1

    def create_prim_constant_node(self, ir_graph, node, module_name):
        # NOTE: compare with string not type, because the type is defined in pytorch C code.
        # `.kind()` can also be used here
        if node.outputsAt(0).type().str() == 'None':
            attrs = {'type': 'None'}
        else:
            attrs = {'type': node.outputsAt(0).type().str(), 'value': node.outputsAt(0).toIValue()}
        self.global_seq += 1
        new_node = ir_graph.add_node(build_full_name(module_name, OpTypeName.Constant, self.global_seq),
                                     node.kind(), attrs)
        return new_node

    def handle_prim_attr_node(self, node, module):
        assert node.hasAttribute('name')
        value = None
        if node.inputsAt(0).debugName() == 'self':
            _val = getattr(module, node.s('name'))
            # TODO: serialize complex data type, and output proper error message
            if isinstance(_val, (int, float, str, bool)):
                value = _val
        attrs = {'name': node.s('name'), 'input': node.inputsAt(0).debugName(), 'value': value}
        return node.kind(), attrs

    def _remove_mangle(self, module_type_str):
        return re.sub('\\.___torch_mangle_\\d+', '', module_type_str)

    def remove_unconnected_nodes(self, ir_graph, targeted_type=None):
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

    def handle_graph_nodes(self, script_module, sm_graph,
                           module, module_name, module_python_name,
                           ir_model, ir_graph,
                           shared_module_index=None):
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
        shared_module_index : dict
            it is used for knowing which module has been created an ir node,
            if created and invoked again, then the new ir node can simply reference that ir node.
            this way we can identify shared modules (i.e., one module invoked multiple times in `forward` function)

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
        if shared_module_index is None:
            shared_module_index = {}

        # some node does not have output but it modifies a variable, for example aten::append
        # %17 : Tensor[] = aten::append(%out.1, %16)
        # %out.1 is updated, and %17 is None
        # we add output to this type of node and connect it to the following node which uses %out.1
        # key: tensor (%out.1), value: node (this node)
        output_remap = {}

        # ===================handle control flow: if===================
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
                elif tensor.node().kind() == 'aten::le':
                    left = _generate_expr(tensor.node().inputsAt(0))
                    right = _generate_expr(tensor.node().inputsAt(1))
                    return f'({left} <= {right})'
                elif tensor.node().kind() == 'aten::ge':
                    left = _generate_expr(tensor.node().inputsAt(0))
                    right = _generate_expr(tensor.node().inputsAt(1))
                    return f'({left} >= {right})'
                elif tensor.node().kind() == 'aten::__not__':
                    value = _generate_expr(tensor.node().inputsAt(0))
                    return f'(not {value})'
                elif tensor.node().kind() == 'aten::Bool':
                    value = _generate_expr(tensor.node().inputsAt(0))
                    return f'bool({value})'
                elif tensor.node().kind() == 'aten::__is__':
                    left = _generate_expr(tensor.node().inputsAt(0))
                    right = _generate_expr(tensor.node().inputsAt(1))
                    return f'({left} is {right})'
                elif tensor.node().kind() == 'aten::__isnot__':
                    left = _generate_expr(tensor.node().inputsAt(0))
                    right = _generate_expr(tensor.node().inputsAt(1))
                    return f'({left} is not {right})'
                elif tensor.node().kind() == 'aten::ne':
                    left = _generate_expr(tensor.node().inputsAt(0))
                    right = _generate_expr(tensor.node().inputsAt(1))
                    return f'({left} != {right})'
                elif tensor.node().kind() == 'aten::gt':
                    left = _generate_expr(tensor.node().inputsAt(0))
                    right = _generate_expr(tensor.node().inputsAt(1))
                    return f'({left} > {right})'
                elif tensor.node().kind() == 'aten::lt':
                    left = _generate_expr(tensor.node().inputsAt(0))
                    right = _generate_expr(tensor.node().inputsAt(1))
                    return f'({left} < {right})'
                elif tensor.node().kind() == 'prim::If':
                    raise RuntimeError('Have not supported `if A and/or B`, please use two `if` statements instead.')
                elif tensor.node().kind() == 'aten::abs':
                    value = _generate_expr(tensor.node().inputsAt(0))
                    return f'(torch.abs({value}))'
                elif tensor.node().kind() == 'aten::sum':
                    value = _generate_expr(tensor.node().inputsAt(0))
                    return f'(torch.sum({value}))'
                elif tensor.node().kind() == 'aten::item':
                    value = _generate_expr(tensor.node().inputsAt(0))
                    return f'({value}.item())'
                else:
                    raise RuntimeError(f'Unsupported op type {tensor.node().kind()} in if condition, '
                                       'you are suggested to decorate the corresponding class with "@basic_unit".')
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
            self.global_seq += 1
            new_node = ir_graph.add_node(build_full_name(module_name, 'noop_identity', self.global_seq), 'noop_identity')
            self._add_edge(ir_graph, blocks[chosen_block].returnNode(), graph_inputs, node_index, new_node, output_remap)
            last_block_node = new_node
            return last_block_node

        # ===================handle function call===================
        def handle_function_callmethod(node):
            # get and handle the first input, which should be an nn.Module
            assert node.hasAttribute('name')
            # NOTE: "forward__0" is hacky, LSTM instance is parsed to call forward__0 in torchscript
            if node.s('name') in ['forward', 'forward__0']:
                # node.inputsAt(0).type() is <class 'torch._C.ClassType'>
                submodule_type_str = self._remove_mangle(node.inputsAt(0).type().str())
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
                    submodule_python_name = build_python_name(module_python_name, submodule_name)
                    submodule_obj = getattr(module, submodule_name)
                    subgraph, sub_m_attrs = self._convert_module(script_module._modules[submodule_name],
                                                                 submodule_obj,
                                                                 submodule_full_name, submodule_python_name,
                                                                 ir_model)
                else:
                    # %8 : __torch__.nni.retiarii.model_apis.nn.___torch_mangle_37.ModuleList = prim::GetAttr[name="cells"](%self)
                    # %10 : __torch__.darts_model.Cell = prim::GetAttr[name="0"](%8)
                    # %s1.4 : Tensor = prim::CallMethod[name="forward"](%10, %4, %4)
                    if submodule.inputsAt(0).type().name() == 'ModuleList':
                        # handle ModuleList
                        predecessor = submodule.inputsAt(0).node()
                        module_name_space = [submodule_name]
                        while predecessor.inputsAt(0).debugName() != 'self':
                            # this is for dealing with nested ModuleList. below is an example
                            # %3 : __torch__.torch.nn.modules.container.___torch_mangle_0.ModuleList = prim::GetAttr[name="ops"](%self)
                            # %5 : __torch__.torch.nn.modules.container.ModuleList = prim::GetAttr[name="0"](%3)
                            # %7 : __torch__.torch.nn.modules.container.ModuleList = prim::GetAttr[name="1"](%3)
                            # %9 : __torch__.torch.nn.modules.container.ModuleList = prim::GetAttr[name="2"](%3)
                            # %11 : __torch__.torch.nn.modules.container.ModuleList = prim::GetAttr[name="3"](%3)
                            # %14 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="0"](%5)
                            # %16 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="1"](%5)
                            # %state.2 : Tensor = prim::CallMethod[name="forward"](%14, %x.1) # modulelist.py:18:24
                            # %state.4 : Tensor = prim::CallMethod[name="forward"](%16, %state.2) # modulelist.py:18:24
                            assert predecessor.kind() == 'prim::GetAttr'
                            module_name_space.append(predecessor.s('name'))
                            predecessor = predecessor.inputsAt(0).node()
                        assert predecessor.kind() == 'prim::GetAttr'
                        assert predecessor.hasAttribute('name')
                        module_name_space.append(predecessor.s('name'))
                        submodule_full_name = build_full_name(module_name, list(reversed(module_name_space)))
                        submodule_python_name = build_python_name(module_python_name, list(reversed(module_name_space)))
                        submodule_obj = module
                        script_submodule = script_module
                        for each_name in list(reversed(module_name_space)):
                            submodule_obj = getattr(submodule_obj, each_name)
                            script_submodule = script_submodule._modules[each_name]
                        subgraph, sub_m_attrs = self._convert_module(script_submodule, submodule_obj, submodule_full_name,
                                                                     submodule_python_name, ir_model)
                    else:
                        raise RuntimeError('Unsupported module case: {}'.format(submodule.inputsAt(0).type().str()))

                if submodule_full_name in shared_module_index:
                    # this module is invoked more than once, the ir node has already been created
                    # create a reference node for it.
                    # example: {"name": "conv2", "operation": {"type": "shared", "parameters": {"reference": "conv1"}}}
                    self.global_seq += 1
                    shared_node_name = build_full_name(submodule_full_name, '', self.global_seq)
                    shared_node_python_name = build_python_name(submodule_python_name, self.global_seq)
                    shared_type_operation = Operation.new('shared', {'reference': submodule_full_name})
                    subcell = ir_graph.add_node(shared_node_name, shared_type_operation)
                    subcell.python_name = shared_node_python_name
                else:
                    # this module is processed for the first time, build cell for it
                    if subgraph is None:
                        # if we do not parse this module's graph, we create Node for this module
                        subcell = ir_graph.add_node(submodule_full_name, submodule_type_str, sub_m_attrs)
                        subcell.python_name = submodule_python_name
                        if isinstance(submodule_obj, MutationAnchor):
                            subcell.update_label(submodule_obj.label)
                        elif isinstance(submodule_obj, InputChoice):
                            subcell.update_label(sub_m_attrs['label'])
                    else:
                        # Graph already created, create Cell for it
                        new_cell = Cell(cell_name=submodule_full_name, parameters=sub_m_attrs)
                        subcell = ir_graph.add_node(submodule_full_name, new_cell)
                        subcell.python_name = submodule_python_name
                    shared_module_index[submodule_full_name] = subcell
                node_index[node] = subcell
                # connect the cell into graph
                self._add_edge(ir_graph, node, graph_inputs, node_index, subcell, output_remap, ignore_first=True)
            else:
                # handle normal member function
                assert hasattr(script_module, node.s('name'))
                # TODO: support non member functions
                assert node.inputsAt(0).debugName() == 'self'
                script_method = getattr(script_module, node.s('name'))  # <class 'torch._C.ScriptMethod'>

                # step #1: generate graph ir for this method
                method_ir_graph = Graph(model=ir_model, graph_id=-100, name='temp_graph', _internal=True)
                self.handle_graph_nodes(script_module, script_method.graph, module,
                                        module_name, module_python_name, ir_model, method_ir_graph, shared_module_index)
                self.refine_graph(method_ir_graph)

                # step #2: merge this graph to its module graph
                for h_node in method_ir_graph.hidden_nodes:
                    h_node.graph = ir_graph
                    ir_graph.hidden_nodes.append(h_node)
                for edge in method_ir_graph.edges:
                    edge.graph = ir_graph
                    if edge.head == method_ir_graph.input_node:
                        # this is a member method, 'self' is the first argument, thus +1
                        assert edge.head_slot is not None
                        _input = node.inputsAt(edge.head_slot + 1)
                        src_node, src_node_idx = self._add_edge_handle_source_node(_input, graph_inputs, ir_graph, output_remap, node_index)
                        edge.head = src_node
                        edge.head_slot = src_node_idx
                    if edge.tail == method_ir_graph.output_node:
                        # since the following nodes have not been created, skip this edge
                        # edge.head is the output node of this method
                        # TODO: check whether there could be multiple output nodes???
                        node_index[node] = edge.head
                        continue
                    ir_graph.edges.append(edge)

        # ===================handle each single node===================
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
            if node.kind() == 'prim::CallMethod':
                handle_function_callmethod(node)
            elif node.kind() == 'prim::CallFunction':
                func_type_str = self._remove_mangle(node.inputsAt(0).type().str())
                func = node.inputsAt(0).node()
                assert func.kind() == 'prim::Constant'
                assert func.hasAttribute('name')
                func_name = func.s('name')
                # create node for func
                self.global_seq += 1
                func_node = ir_graph.add_node(build_full_name(module_name, func_name, self.global_seq),
                                              '{}.{}'.format(func_type_str, func_name))
                func_python_name = build_python_name(module_python_name, func_name)
                func_node.python_name = func_python_name
                node_index[node] = func_node
                self._add_edge(ir_graph, node, graph_inputs, node_index, func_node, output_remap, ignore_first=True)
            elif node.kind() == 'prim::Constant':
                new_node = self.create_prim_constant_node(ir_graph, node, module_name)
                node_index[node] = new_node
            elif node.kind() in ['prim::ListConstruct', 'prim::ListUnpack', 'prim::TupleConstruct', 'prim::TupleUnpack']:
                self.global_seq += 1
                prim_op_name = node.kind().split('::')[-1]
                new_node = ir_graph.add_node(build_full_name(module_name, prim_op_name, self.global_seq), node.kind())
                node_index[node] = new_node
                self._add_edge(ir_graph, node, graph_inputs, node_index, new_node, output_remap)
            elif node.kind() == 'prim::GetAttr':
                node_type, attrs = self.handle_prim_attr_node(node, module)
                self.global_seq += 1
                new_node = ir_graph.add_node(build_full_name(module_name, OpTypeName.Attr, self.global_seq),
                                             node_type, attrs)
                node_index[node] = new_node
            elif node.kind() == 'prim::If':
                last_block_node = handle_if_node(node)
                # last_block_node is None means no node in the branch block
                node_index[node] = last_block_node
            elif node.kind() == 'prim::Loop':
                # refer to https://gist.github.com/liuzhe-lz/90c35d9dd6fd7f3f32544940151ab186
                raise RuntimeError('Loop has not been supported yet!')
            elif node.kind().startswith('prim::'):
                self.global_seq += 1
                prim_op_name = node.kind().replace('::', '__')
                prim_node = ir_graph.add_node(build_full_name(module_name, prim_op_name, self.global_seq), node.kind())
                node_index[node] = prim_node
                self._add_edge(ir_graph, node, graph_inputs, node_index, prim_node, output_remap)
            elif node.kind() == 'aten::append':
                self.global_seq += 1
                aten_op_name = node.kind().replace('::', '__')
                aten_node = ir_graph.add_node(build_full_name(module_name, aten_op_name, self.global_seq), node.kind())
                node_index[node] = aten_node
                self._add_edge(ir_graph, node, graph_inputs, node_index, aten_node, output_remap)
                output_remap[node.inputsAt(0)] = node
            elif node.kind().startswith('aten::'):
                # handle aten::XXX
                self.global_seq += 1
                aten_op_name = node.kind().replace('::', '__')
                aten_op_python_name = node.kind().replace('aten::', '')
                aten_node = ir_graph.add_node(build_full_name(module_name, aten_op_name, self.global_seq), node.kind())
                aten_python_name = build_python_name(module_python_name, aten_op_python_name)
                aten_node.python_name = aten_python_name
                node_index[node] = aten_node
                self._add_edge(ir_graph, node, graph_inputs, node_index, aten_node, output_remap)
            else:
                raise RuntimeError('Unsupported kind: {}'.format(node.kind()))

            return node_index[node]

        for node in sm_graph.nodes():
            handle_single_node(node)

        if node_index != {}:
            for _output in sm_graph.outputs():
                ir_graph._add_output(_convert_name(_output.debugName()))
                predecessor_node_outputs = [o for o in _output.node().outputs()]
                if len(predecessor_node_outputs) == 1:
                    src_node_idx = None
                else:
                    src_node_idx = predecessor_node_outputs.index(_output)

                ir_graph.add_edge(head=(node_index[_output.node()], src_node_idx),
                                  tail=(ir_graph.output_node, None))
        else:
            # here is an example that the ir_graph and node_index is empty
            # graph(%self : __torch__.torchmodels.googlenet.GoogLeNet,
            # %x.1 : Tensor): return (%x.1)
            # add an edge from head to tail to handle this situation
            ir_graph.add_edge(head=(ir_graph.input_node, 0), tail=(ir_graph.output_node, None))

    def merge_aten_slices(self, ir_graph):
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
                for edge in head_node.incoming_edges:
                    edge.tail = new_slice_node
                for edge in head_node.outgoing_edges:
                    edge.head = new_slice_node
                ir_graph.hidden_nodes.remove(head_node)
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

    def refine_graph(self, ir_graph):
        """
        Do the following process to simplify graph:
        1. remove unconnected constant node
        2. remove unconnected getattr node
        """
        # some constant is not used, for example, function name as prim::Constant
        self.remove_unconnected_nodes(ir_graph, targeted_type='prim::Constant')
        self.remove_unconnected_nodes(ir_graph, targeted_type='prim::GetAttr')
        self.merge_aten_slices(ir_graph)

    def _handle_inputchoice(self, module):
        return {
            'n_candidates': module.n_candidates,
            'n_chosen': module.n_chosen,
            'reduction': module.reduction,
            'label': module.label
        }

    def _handle_valuechoice(self, module):
        return {
            'candidates': module.candidates,
            'label': module.label,
        }

    def _convert_module(self, script_module, module, module_name, module_python_name, ir_model):
        # NOTE: have not supported nested LayerChoice, i.e., a candidate module
        # also has LayerChoice or InputChoice or ValueChoice
        original_type_name = script_module.original_name
        m_attrs = None
        if original_type_name == OpTypeName.LayerChoice:
            graph = Graph(ir_model, -100, module_name, _internal=True)  # graph_id is not used now
            graph.python_name = module_python_name
            candidate_name_list = []
            for cand_name in module.names:
                cand = module[cand_name]
                script_cand = script_module._modules[str(cand_name)]
                # FIXME: should use cand_name instead of cand_full_name
                cand_full_name = build_cand_name(str(cand_name), module.label)
                cand_python_name = build_python_name(module_python_name, str(cand_name))
                candidate_name_list.append(cand_full_name)
                subgraph, attrs = self._convert_module(script_cand, cand, cand_full_name, cand_python_name, ir_model)
                if subgraph is not None:
                    cand_node = graph.add_node(subgraph.name, Cell(cell_name=subgraph.name, parameters=attrs))
                    cand_node.python_name = cand_python_name
                else:
                    cand_type = '__torch__.' + get_importable_name(cand.__class__)
                    cand_node = graph.add_node(cand_full_name, cand_type, attrs)
                    cand_node.python_name = cand_python_name
            graph._register()
            return graph, {'mutation': 'layerchoice', 'label': module.label, 'candidates': candidate_name_list}
        elif original_type_name == OpTypeName.InputChoice:
            m_attrs = self._handle_inputchoice(module)
        elif original_type_name == OpTypeName.ValueChoice:
            m_attrs = self._handle_valuechoice(module)
        elif original_type_name == OpTypeName.MutationAnchor:
            m_attrs = get_init_parameters_or_fail(module)
        elif module.__class__.__module__.startswith('torch.nn') and \
            original_type_name in torch.nn.__dict__ and \
                original_type_name not in MODULE_EXCEPT_LIST:
            # this is a basic module from pytorch, no need to parse its graph
            m_attrs = get_init_parameters_or_fail(module)
        elif getattr(module, '_nni_basic_unit', False):
            # this module is marked as serialize, won't continue to parse
            m_attrs = get_init_parameters_or_fail(module)
        elif isinstance(module, MutableModule) and not isinstance(module, Repeat) and module.mutables:
            raise RuntimeError(f'Arbitrary add_mutable() is not supported in graph-based model space, but found in {module}')
        if m_attrs is not None:
            return None, m_attrs

        # handle TorchScript graph
        sm_graph = script_module.graph
        self.global_graph_id += 1
        ir_graph = Graph(model=ir_model, graph_id=self.global_graph_id, name=module_name, _internal=True)
        ir_graph.python_name = module_python_name

        # handle graph nodes
        self.handle_graph_nodes(script_module, sm_graph, module,
                                module_name, module_python_name, ir_model, ir_graph)
        self.refine_graph(ir_graph)

        ir_graph._register()

        # add mutation signal for special modules
        if original_type_name == OpTypeName.Repeat:
            attrs = {
                'mutation': 'repeat',
                'label': module.label,
                'depth': module.depth_choice,
                'max_depth': module.max_depth,
                'min_depth': module.min_depth,
            }
            return ir_graph, attrs

        return ir_graph, {}

    def convert_module(self, script_module, module, module_name, ir_model):
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
        return self._convert_module(script_module, module, module_name, None, ir_model)


class GraphConverterWithShape(GraphConverter):
    """
    Convert a pytorch model to nni ir along with input/output shape info.
    Based ir acquired through ``torch.jit.script``
    and shape info acquired through ``torch.jit.trace``.

    .. warning::

        Known issues:

        1. ``InputChoice`` and ``ValueChoice`` not supported yet.
        2. Currently random inputs are fed while tracing layerchoice.
           If forward path of candidates depends on input data, then wrong path will be traced.
           This will result in incomplete shape info.
    """

    def convert_module(self, script_module, module, module_name, ir_model, dummy_input):
        module.eval()

        ir_graph, attrs = self._convert_module(script_module, module, module_name, None, ir_model)
        self.remove_dummy_nodes(ir_model)
        self._initialize_parameters(ir_model)
        self._trace_module(module, module_name, ir_model, dummy_input)
        return ir_graph, attrs

    def _initialize_parameters(self, ir_model: GraphModelSpace):
        for ir_node in ir_model.get_nodes():
            if ir_node.operation.parameters is None:
                ir_node.operation.parameters = {}
            ir_node.operation.attributes.setdefault('input_shape', [])
            ir_node.operation.attributes.setdefault('output_shape', [])

    def _trace_module(self, module, module_name, ir_model: GraphModelSpace, dummy_input):
        # First, trace the whole graph
        tm_graph = self._trace(module, dummy_input)

        for node in tm_graph.nodes():
            shape_parameters, parameters = _extract_info_from_trace_node(node)
            # '__module.convpool/__module.convpool.1/__module.convpool.1.conv'
            ir_node = match_node(ir_model, node, module_name)
            if ir_node is not None:
                ir_node.operation.attributes.update(shape_parameters)
                if parameters:
                    ir_node.operation.parameters.update(parameters)

        self.propagate_shape(ir_model)

        # trace each layerchoice
        for name, submodule in module.named_modules():
            # TODO: support InputChoice and ValueChoice
            if isinstance(submodule, LayerChoice):
                full_name = get_full_name_by_scope_name(ir_model, name.split('.'), module_name)
                lc_node = ir_model.get_node_by_name(full_name)
                assert lc_node is not None, f'Cannot find a node with name {full_name}'

                for cand_name in submodule.names:
                    cand = submodule[cand_name]
                    cand_name = build_cand_name(str(cand_name), submodule.label)
                    # TODO: Feed the exact input tensor if user provides input,
                    # in case the path changes according to input data.
                    lc_inputs = [torch.randn(shape) for shape in lc_node.operation.attributes['input_shape']]
                    self._trace_module(cand, str(cand_name), ir_model, lc_inputs)

    def propagate_shape(self, ir_model: GraphModelSpace):

        def propagate_shape_for_graph(graph: 'Graph'):
            if graph == ir_model.root_graph:
                return

            graph_node = ir_model.get_node_by_name(graph.name)
            assert graph_node is not None, f'Cannot find a node with name {graph.name}'
            if not _without_shape_info(graph_node):
                return

            if is_layerchoice_node(graph_node):
                cand_name = graph_node.operation.parameters['candidates'][0]
                cand_node = ir_model.get_node_by_name(cand_name)
                assert cand_node is not None, f'Cannot find a node with name {cand_name}'
                if _without_shape_info(cand_node):
                    propagate_shape_for_graph(ir_model.graphs[cand_name])
                graph_node.operation.attributes['input_shape'] = cand_node.operation.attributes['input_shape']
                graph_node.operation.attributes['output_shape'] = cand_node.operation.attributes['output_shape']
            else:
                input_shape = [[]] * len(graph.input_node.operation.io_names or [])
                output_shape = [[]] * len(graph.output_node.operation.io_names or [])
                for edge in graph.input_node.outgoing_edges:
                    node = edge.tail
                    if _without_shape_info(node):
                        if node.name in ir_model.graphs:
                            propagate_shape_for_graph(ir_model.graphs[node.name])
                    if node.operation.attributes['input_shape']:
                        input_shape[edge.head_slot or 0] = node.operation.attributes['input_shape'][edge.tail_slot or 0]
                graph_node.operation.attributes['input_shape'] = input_shape
                for edge in graph.output_node.incoming_edges:
                    node = edge.head
                    if _without_shape_info(node):
                        if node.name in ir_model.graphs:
                            propagate_shape_for_graph(ir_model.graphs[node.name])
                    if node.operation.attributes['output_shape']:
                        output_shape[edge.tail_slot or 0] = node.operation.attributes['output_shape'][edge.head_slot or 0]
                graph_node.operation.attributes['output_shape'] = output_shape

            propagate_shape_for_graph(graph_node.graph)

        # propagate from node to graph
        for node in ir_model.get_nodes():
            propagate_shape_for_graph(node.graph)

    def _trace(self, module, dummy_input):
        traced_module = torch.jit.trace(module, dummy_input)
        torch._C._jit_pass_inline(traced_module.graph)
        return traced_module.graph

    def remove_dummy_nodes(self, ir_model: GraphModelSpace):
        # remove identity nodes
        for node in ir_model.get_nodes_by_type('noop_identity'):
            graph = node.graph
            for in_edge in node.incoming_edges:
                for out_edge in node.outgoing_edges:
                    if in_edge.tail_slot == out_edge.head_slot:
                        graph.add_edge(head=(in_edge.head, in_edge.head_slot), tail=(out_edge.tail, out_edge.tail_slot))
                        graph.del_edge(in_edge)
                        graph.del_edge(out_edge)
                    break
            node.remove()
