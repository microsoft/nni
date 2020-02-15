# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import queue
import re
import torch
from .compress_modules import replace_module
from .infer_shape import ModuleMasks, infer_from_mask, infer_from_inshape, infer_from_outshape

_logger = logging.getLogger(__name__)


def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name

    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module

    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module

class GNode:
    """
    It is used to represent a node in model graph, in this graph a module is a node,
    a function out of module (in ```forward``` function) could also be a node.
    """
    def __init__(self, node_name, node_type, op_type, inputs, outputs, nodes):
        """
        Parameters
        ----------
        node_name : str
            It is module name if the node is a module, it is ```scope_name.node_kind.seq``` if it is a func
        node_type : str
            It only has two options: `module` or `func`
        op_type : str
            The operation type of the module or func
        inputs : list of str
            All the inputs of this node, each element is debugName of one input
        outputs : list of str
            All the outputs of this node, each element is debugName of one output
        nodes : list of node
            All the trace graph nodes included in this module or func
        """
        self.name = node_name
        self.type = node_type
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        # store supplementary information for different op types
        # for example, for ```view``` it stores the shape of its input and output
        self.auxiliary = None

class ModelSpeedup:
    """
    This class is to speedup the model with provided weight mask
    """

    def __init__(self, model, dummy_input, masks_file):
        """
        Parameters
        ----------
        model : pytorch model
            The model user wants to speed up
        dummy_input : pytorch tensor
            The dummy input for ```jit.trace```, users should put it on right device before pass in
        masks_file : str
            The path of user provided mask file
        """
        self.bound_model = model
        self.dummy_input = dummy_input
        self.masks = torch.load(masks_file)
        self.is_training = model.training
        # to obtain forward graph, model should be in ```eval``` mode
        if self.is_training:
            model.eval()
        self.trace_graph = torch.jit.trace(model, dummy_input)
        if self.is_training:
            model.train()
        self.inferred_masks = dict() # key: module_name, value: ModuleMasks
        self.g_nodes = list()
        self.global_count = 0
        self.name_to_gnode, self.input_to_gnode, self.output_to_gnode = self._build_graph()

    def _build_index_for_gnodes(self, g_nodes):
        """
        Build indexes for quick search

        Parameters
        ----------
        g_nodes : list of GNode
            All the g_node in processed model graph

        Returns
        -------
        dict
            use name to index g_nodes, key: node name, value: g_node
        dict
            use input (its name) to index g_nodes,
            key: input, value: list of g_nodes that take this input
        dict
            use output (its name) to index g_nodes,
            key: output, value: g_node that generates this output
        """
        name_to_gnode = dict()
        input_to_gnode = dict()
        output_to_gnode = dict()
        for node in g_nodes:
            name_to_gnode[node.name] = node
            for _input in node.inputs:
                if _input in input_to_gnode:
                    input_to_gnode[_input].append(node)
                else:
                    input_to_gnode[_input] = [node]
            for output in node.outputs:
                assert not output in output_to_gnode, \
                    "One output cannot be generated by multiple nodes"
                output_to_gnode[output] = node
        return name_to_gnode, input_to_gnode, output_to_gnode

    def _expand_non_prim_node(self, node, nodes, input_to_node, output_to_node):
        """
        For trace graph nodes, some nodes are not in modules, these nodes are usually generated by
        the functions directly called in module ```forward```. For such nodes, some of them are
        trivial op which are label by ```prim::```, some of them are not such ops which is call
        non-prim ops. This function is to merge neighbor prim ops to a non-prim op, to construct
        a GNode.

        Parameters
        ----------
        node : trace graph node
            The non-prim node to expand
        nodes : list of trace graph node
            All the trace graph nodes within the same scope as the non-prim node
        input_to_node : dict
            key: input name, value: a node that uses this input
        output_to_node : dict
            key: output name, value: a node that generates this output

        Returns
        -------
        GNode
            the expanded non-prim node in GNode format
        """
        # TODO: scope name could be empty
        node_name = '.'.join([node.scopeName(), node.kind(), str(self.global_count)])
        _logger.debug("expand non-prim node, node name: %s", node_name)
        self.global_count += 1
        op_type = node.kind()

        node_group = [node]
        inputs = list()
        outputs = list()
        node_queue = queue.Queue()
        node_queue.put(node)
        while not node_queue.empty():
            curr_node = node_queue.get()
            for _input in curr_node.inputs():
                input_name = _input.debugName()
                if input_name in output_to_node and output_to_node[input_name] in nodes:
                        predecessor_node = output_to_node[input_name]
                        if predecessor_node.kind().startswith('prim::'):
                            node_group.append(predecessor_node)
                            node_queue.put(predecessor_node)
                        else:
                            inputs.append(input_name)
                else:
                    inputs.append(input_name)
        for output in node.outputs():
            outputs.append(output.debugName())
        g_node = GNode(node_name, 'func', op_type, inputs, outputs, node_group)
        return g_node

    def _extract_shape_info(self, node):
        """
        Extract the shape information of ```aten::view``` node

        Parameters
        ----------
        node : trace graph node
            It should be ```aten::view``` node

        Returns
        -------
        dict
            Include shape of input tensor and shape of output tensor
        """
        t_input = None
        for _input in node.inputs():
            t_input = _input
            break
        t_output = node.output()
        assert isinstance(t_input.type(), torch._C.TensorType)
        assert isinstance(t_output.type(), torch._C.TensorType)
        in_shape = t_input.type().sizes()
        out_shape = t_output.type().sizes()
        return {'in_shape': in_shape, 'out_shape': out_shape}

    def _build_graph(self):
        """
        Build graph using our defined format from jit trace.
        There are basically three steps: first, construct necessary information (data structures),
        second, extract all the modules to convert to GNode, Third, extract all functions to convert
        to GNode.

        Returns
        -------
        dict
            use name to index g_nodes, key: node name, value: g_node
        dict
            use input (its name) to index g_nodes,
            key: input, value: list of g_nodes that take this input
        dict
            use output (its name) to index g_nodes,
            key: output, value: g_node that generates this output
        """
        graph = self.trace_graph.graph
        # if torch 1.4.0 is used, consider run torch._C._jit_pass_inline(graph) here
        #_logger.debug(graph)
        # build output mapping, from output debugName to its node
        output_to_node = dict()
        # build input mapping, from input debugName to its node
        input_to_node = dict()
        # build module mapping, from module name to all nodes (as list) under this module scope
        module_to_nodes = dict()
        # module name to its type
        module_to_type = dict()
        # the mapping of function (non-module in forward) to nodes, key is scope name
        func_to_nodes = dict()

        graph_inputs = list()
        graph_outputs = list()
        for _input in graph.inputs():
            graph_inputs.append(_input.debugName())
        for output in graph.outputs():
            graph_outputs.append(output.debugName())

        for node in graph.nodes():
            # populate output_to_node and input_to_node
            for output in node.outputs():
                output_name = output.debugName()
                output_to_node[output_name] = node
            for _input in node.inputs():
                input_name = _input.debugName()
                input_to_node[input_name] = node
            scope_name = node.scopeName() # example: scope_name, 'MyCell/Linear[linear]'
            module_name_slices = re.findall(r'\[(.*?)\]', scope_name)
            module_name = '.'.join(module_name_slices)
            # if module_name is empty, it is not a module
            if module_name == '':
                if scope_name == '':
                    continue
                else:
                    if scope_name in func_to_nodes:
                        func_to_nodes[scope_name].append(node)
                    else:
                        func_to_nodes[scope_name] = [node]
            else:
                scope_slice = scope_name.split('/')[-1]
                module_type = scope_slice.split('[')[0]
                module_to_type[module_name] = module_type
                if module_name in module_to_nodes:
                    module_to_nodes[module_name].append(node)
                else:
                    module_to_nodes[module_name] = [node]

        # construct GNode from module
        for module_name, nodes in module_to_nodes.items():
            inputs = set()
            outputs = set()
            for node in nodes:
                for output in node.outputs():
                    outputs.add(output.debugName())
                for _input in node.inputs():
                    inputs.add(_input.debugName())
            m_inputs = list()
            m_outputs = list()
            for output in outputs:
                # TODO: one input could be the input of multiple nodes
                if not output in input_to_node and output in graph_outputs:
                    m_outputs.append(output)
                elif not input_to_node[output] in nodes:
                    m_outputs.append(output)
            for _input in inputs:
                if not _input in output_to_node and _input in graph_inputs:
                    m_inputs.append(_input)
                elif not output_to_node[_input] in nodes:
                    m_inputs.append(_input)
            if module_name == '':
                _logger.warning("module_name is empty string")
            g_node = GNode(module_name, 'module', module_to_type[module_name], m_inputs, m_outputs, nodes)
            self.g_nodes.append(g_node)

        # each scope_name may have multiple funcs, we split them and create GNode for each of them
        for scope_name, nodes in func_to_nodes.items():
            # extract non prim:: nodes
            non_prim_nodes = list()
            for node in nodes:
                if not node.kind().startswith('prim::'):
                    non_prim_nodes.append(node)
            # for each non prim node, expand it has a GNode
            for node in non_prim_nodes:
                g_node = self._expand_non_prim_node(node, nodes, input_to_node, output_to_node)
                self.g_nodes.append(g_node)
                # get shape infor for view (aten::view) func
                if g_node.op_type == 'aten::view':
                    g_node.auxiliary = self._extract_shape_info(node)

        # build index for g_nodes
        name_to_gnode, input_to_gnode, output_to_gnode = self._build_index_for_gnodes(self.g_nodes)

        return name_to_gnode, input_to_gnode, output_to_gnode

    def _find_predecessors(self, module_name):
        """
        Find predecessor GNode of the given GNode

        Parameters
        ----------
        module_name : str
            The name of the GNode

        Returns
        -------
        list
            a list of GNodes who are the given GNode's predecessor
        """
        predecessors = []
        for _input in self.name_to_gnode[module_name].inputs:
            if not _input in self.output_to_gnode:
                _logger.debug("cannot find gnode with %s as its output", _input)
            else:
                g_node = self.output_to_gnode[_input]
                predecessors.append(g_node.name)
        return predecessors

    def _find_successors(self, module_name):
        """
        Find successor GNodes of the given GNode

        Parameters
        ----------
        module_name : str
            The name of the GNode

        Returns
        -------
        list
            a list of GNodes who are the given GNode's successor
        """
        successors = []
        for output in self.name_to_gnode[module_name].outputs:
            assert output in self.input_to_gnode, "No gnode with input {}".format(output)
            g_nodes = self.input_to_gnode[output]
            for g_node in g_nodes:
                successors.append(g_node.name)
        return successors

    def infer_module_mask(self, module_name, mask=None, in_shape=None, out_shape=None):
        """
        Infer input shape / output shape based on the module's weight mask / input shape / output shape.

        For a module:
            Infer its input and output shape from its weight mask
            Infer its output shape from its input shape
            Infer its input shape from its output shape

        If its input shape is changed, continue infering its predecessors
        If its output shape is changed, continue infering its successors

        Parameters
        ----------
        module_name : str
            The name of the GNode
        mask : tensor of mask or ModuleMasks
            Mask of the weights in this GNode (i.e., module)
        in_shape : ModuleMasks
            Input shape of this GNode
        out_shape : ModuleMasks
            Output shape of this GNode
        """
        input_cmask = output_cmask = None
        if module_name in self.inferred_masks:
            module_masks = self.inferred_masks[module_name]
        else:
            module_masks = ModuleMasks(module_name)
            self.inferred_masks[module_name] = module_masks

        m_type = self.name_to_gnode[module_name].op_type
        _logger.debug("infer mask of module %s with op_type %s", module_name, m_type)
        if mask is not None:
            _logger.debug("mask is not None")
            if not m_type in infer_from_mask:
                raise RuntimeError("Has not supported infering \
                    input/output shape from mask for module/function: `{}`".format(m_type))
            input_cmask, output_cmask = infer_from_mask[m_type](module_masks, mask)
        if in_shape is not None:
            _logger.debug("in_shape is not None")
            if not m_type in infer_from_inshape:
                raise RuntimeError("Has not supported infering \
                    output shape from input shape for module/function: `{}`".format(m_type))
            if m_type == 'aten::view':
                output_cmask = infer_from_inshape[m_type](module_masks,
                                                          in_shape,
                                                          self.name_to_gnode[module_name].auxiliary)
            else:
                output_cmask = infer_from_inshape[m_type](module_masks, in_shape)
        if out_shape is not None:
            _logger.debug("out_shape is not None")
            if not m_type in infer_from_outshape:
                raise RuntimeError("Has not supported infering \
                    input shape from output shape for module/function: `{}`".format(m_type))
            input_cmask = infer_from_outshape[m_type](module_masks, out_shape)

        if input_cmask:
            predecessors = self._find_predecessors(module_name)
            for _module_name in predecessors:
                self.infer_module_mask(_module_name, out_shape=input_cmask)
        if output_cmask:
            successors = self._find_successors(module_name)
            for _module_name in successors:
                self.infer_module_mask(_module_name, in_shape=output_cmask)

    def infer_modules_masks(self):
        """
        Do shape inference of involved modules, including the shape of weights, inputs, output
        """
        for module_name, mask in self.masks.items():
            self.infer_module_mask(module_name, mask=mask)

    def replace_compressed_modules(self):
        """
        Replace all the modules that have changed (weights/inputs/output) shape.
        The new module is created using the same arguments of the to-be-replaced module,
        and correctly inherits its weights.

        NOTE: ```func``` type cannot be replaced as it is not a module, thus, one limitation
        is that ```func``` should be not required to be replaced.
        """
        for module_name in self.inferred_masks:
            g_node = self.name_to_gnode[module_name]
            _logger.debug("replace %s, in %s type, with op_type %s",
                          module_name, g_node.type, g_node.op_type)
            if g_node.type == 'module':
                super_module, leaf_module = get_module_by_name(self.bound_model, module_name)
                m_type = g_node.op_type
                if not m_type in replace_module:
                    raise RuntimeError("Has not supported replacing the module: `{}`".format(m_type))
                compressed_module = replace_module[m_type](leaf_module, self.inferred_masks[module_name])
                setattr(super_module, module_name.split('.')[-1], compressed_module)
            elif g_node.type == 'func':
                _logger.info("Warning: cannot replace (name: %s, op_type: %s) which is func type",
                             module_name, g_node.op_type)
            else:
                raise RuntimeError("Unsupported GNode type: {}".format(g_node.type))

    def speedup_model(self):
        """
        There are basically two steps:
        first, do mask/shape inference,
        second, replace modules
        """
        _logger.info("start to speed up the model")
        _logger.info("infer module masks...")
        self.infer_modules_masks()
        _logger.info("replace compressed modules...")
        self.replace_compressed_modules()
        _logger.info("speedup done")
        # resume the model mode to that before the model is speed up
        if self.is_training:
            self.bound_model.train()
        else:
            self.bound_model.eval()