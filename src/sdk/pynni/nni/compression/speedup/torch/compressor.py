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
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module

class GNode:
    def __init__(self, node_name, node_type, op_type, inputs, outputs, nodes):
        self.name = node_name # module name if is module, scope name + seq if is func
        self.type = node_type # module or func
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes

class ModelSpeedup:
    """
    Abstract base PyTorch ModelSpeedup
    """

    def __init__(self, model, dummy_input, masks_file):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        masks : dict
            the generated masks for modules,
            key is module name,
            value is a dict including key `weight`, or also key `bias`
        onnx_graph : xxx
            it is used to parse dependencies between modules
        """
        self.bound_model = model
        self.dummy_input = dummy_input
        self.masks = torch.load(masks_file)
        #ori_masks = torch.load(masks_file)
        #self.masks = {'feature.1': ori_masks['feature.1']}
        self.is_training = model.training
        if self.is_training:
            model.eval()
        self.trace_graph = torch.jit.trace(model, dummy_input)
        if self.is_training:
            model.train()
        #print("masks: ", self.masks)
        #print(self.trace_graph)
        #print(self.trace_graph.graph)
        self.inferred_masks = dict() # key: module_name, value: ModuleMasks
        self.g_nodes = list()
        self.global_count = 0
        self.name_to_gnode, self.input_to_gnode, self.output_to_gnode = self._build_graph()
        #self.replaced_modules = dict()

    def _build_index_for_gnodes(self, g_nodes):
        """
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
                if output in output_to_gnode:
                    print("output: ", output)
                    print("gnode: ", output_to_gnode[output].name)
                assert not output in output_to_gnode, "One output cannot be generated by multiple nodes"
                output_to_gnode[output] = node
        return name_to_gnode, input_to_gnode, output_to_gnode

    def _expand_non_prim_node(self, node, nodes, input_to_node, output_to_node):
        """
        """
        #print('^=' * 30)
        #for n in nodes:
        #    print(n)
        #print('v=' * 30)
        # TODO: scope name could be empty
        node_name = '.'.join([node.scopeName(), node.kind(), str(self.global_count)])
        print('node_name: ', node_name)
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
                print('_input: ', _input)
                input_name = _input.debugName()
                if input_name in output_to_node and output_to_node[input_name] in nodes:
                        predecessor_node = output_to_node[input_name]
                        print("predecessor_node: ", predecessor_node)
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
        print('^' * 30)
        for n in g_node.nodes:
            print(n)
        print('v' * 30)
        return g_node

    def _build_graph(self):
        """
        """
        graph = self.trace_graph.graph
        #torch._C._jit_pass_inline(graph)
        print(graph)
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

        #print("graph_inputs: ", graph_inputs)
        #print("graph_outputs: ", graph_outputs)

        for node in graph.nodes():
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
                    # TODO: there might be more than one funcs in scope_name
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

        print('xx' * 30)
        for k in output_to_node:
            print(k)
        print('yy' * 30)

        # for each module, find its inputs and outputs
        # build module mapping, from module name to its inputs debugName and outputs debugName,
        #module_to_inputs = dict()
        #module_to_outputs = dict()
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
            #module_to_inputs[module_name] = m_inputs
            #module_to_outputs[module_name] = m_outputs
            print("module node_name: ", module_name)
            if module_name == '':
                for n in nodes:
                    print(n)
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

        # build index for g_nodes
        name_to_gnode, input_to_gnode, output_to_gnode = self._build_index_for_gnodes(self.g_nodes)

        return name_to_gnode, input_to_gnode, output_to_gnode #output_to_node, input_to_node

    '''def _do_module_replace(self, module_name, mask=None, in_shape=None, out_shape=None):
        """
        """
        assert not module_name in self.replaced_modules
        input_cmask = output_cmask = None
        assert module_name in self.module_inputs, "module does not exist in trace graph"
        if mask is not None:
            assert in_shape is None and out_shape is None
            super_module, leaf_module = get_module_by_name(self.bound_model, module_name)
            m_type = self.module_to_type[module_name]
            compressed_module, input_cmask, output_cmask = cms[m_type](leaf_module, mask)
            setattr(super_module, module_name, compressed_module)

        if in_shape is not None:
            assert not module_name in self.masks
            super_module, leaf_module = get_module_by_name(self.bound_model, module_name)
            m_type = self.module_to_type[module_name]
            compressed_module, input_cmask, output_cmask = cms_input[m_type](leaf_module, in_shape)

        if out_shape is not None:
            assert not module_name in self.masks
            #...
        return input_cmask, output_cmask'''

    def _find_predecessors(self, module_name):
        """
        """
        predecessors = []
        for _input in self.name_to_gnode[module_name].inputs:
            if not _input in self.output_to_gnode:
                print(_input)
            if not _input in self.output_to_gnode:
                # TODO: check _input which does not have node
                print("output with no gnode: ", _input)
            else:
                g_node = self.output_to_gnode[_input]
                predecessors.append(g_node.name)
        return predecessors

    def _find_successors(self, module_name):
        """
        """
        successors = []
        for output in self.name_to_gnode[module_name].outputs:
            if not output in self.input_to_gnode:
                print(output)
            assert output in self.input_to_gnode
            g_nodes = self.input_to_gnode[output]
            for g_node in g_nodes:
                successors.append(g_node.name)
        return successors

    def infer_module_mask(self, module_name, mask=None, in_shape=None, out_shape=None):
        """
        """
        input_cmask = output_cmask = None
        if module_name in self.inferred_masks:
            module_masks = self.inferred_masks[module_name]
        else:
            module_masks = ModuleMasks(module_name)
            self.inferred_masks[module_name] = module_masks

        m_type = self.name_to_gnode[module_name].op_type
        if m_type == 'VGG':
            print("VGG module name: ", module_name)
            for node in self.name_to_gnode[module_name].nodes:
                print(node)
        print("infer_module_mask: {}, module type: {}".format(module_name, m_type))
        if mask is not None:
            print("mask is not None")
            input_cmask, output_cmask = infer_from_mask[m_type](module_masks, mask)
        if in_shape is not None:
            print("in_shape is not None")
            output_cmask = infer_from_inshape[m_type](module_masks, in_shape)
        if out_shape is not None:
            print("out_shape is not None")
            input_cmask = infer_from_outshape[m_type](module_masks, out_shape)

        if input_cmask:
            print("input_cmask is not None")
            predecessors = self._find_predecessors(module_name)
            for _module_name in predecessors:
                print("input_cmask, module_name: ", _module_name)
                self.infer_module_mask(_module_name, out_shape=input_cmask)
        if output_cmask:
            print("output_cmask is not None")
            successors = self._find_successors(module_name)
            for _module_name in successors:
                print("output_cmask, module_name: ", _module_name)
                self.infer_module_mask(_module_name, in_shape=output_cmask)

    def infer_modules_masks(self):
        """
        """
        for module_name, mask in self.masks.items():
            self.infer_module_mask(module_name, mask=mask)

    def replace_compressed_modules(self):
        """
        """
        print('*' * 30)
        for module_name in self.inferred_masks:
            #module_masks = self.inferred_masks[module_name]
            #print(module_masks.param_masks)
            #print(module_masks.input_mask)
            #print(module_masks.output_mask)
            g_node = self.name_to_gnode[module_name]
            print(module_name, g_node.op_type)
            if g_node.type == 'module':
                super_module, leaf_module = get_module_by_name(self.bound_model, module_name)
                m_type = g_node.op_type
                compressed_module = replace_module[m_type](leaf_module, self.inferred_masks[module_name])
                setattr(super_module, module_name.split('.')[-1], compressed_module)
            elif g_node.type == 'func':
                print("Cannot replace func...")
            else:
                raise RuntimeError("Unsupported GNode type: {}".format(g_node.type))

    def speedup_model(self):
        """
        """
        #self.bound_model(self.dummy_input)
        print("start to compress")
        self.infer_modules_masks()
        self.replace_compressed_modules()
        print("finished compressing")
        #for name, module in self.bound_model.named_modules():
        #    print(name, module)
        #self.bound_model(self.dummy_input)
        if self.is_training:
            self.bound_model.train()
        else:
            self.bound_model.eval()