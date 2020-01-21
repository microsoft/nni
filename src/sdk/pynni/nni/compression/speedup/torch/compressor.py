# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import re
import torch
from .compress_modules import compress_modules as cms
from .infer_shape import ModuleMasks, infer_from_mask, infer_from_inshape, infer_from_outshape

_logger = logging.getLogger(__name__)


def get_module_by_name(model, module_name):
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module

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
        self.masks = torch.load(masks_file)
        self.trace_graph = torch.jit.trace(model, dummy_input)
        self.output_to_node, self.input_to_node, self.module_to_inputs, self.module_to_outputs, self.module_to_type = self._build_graph()
        
        #self.replaced_modules = dict()
        self.inferred_masks = dict() # key: module_name, value: ModuleMasks

    def _build_graph(self):
        """
        """
        graph = self.trace_graph.graph
        print(graph)
        # build output mapping, from output debugName to its node
        output_to_node = dict()
        # build input mapping, from input debugName to its node
        input_to_node = dict()
        #build module mapping, from module name to all nodes (as list) under this module scope
        module_to_nodes = dict()
        # module name to its type
        module_to_type = dict()

        graph_inputs = list()
        graph_outputs = list()
        for _input in graph.inputs():
            graph_inputs.append(_input.debugName())
        for output in graph.outputs():
            graph_outputs.append(output.debugName())

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
                continue
            scope_slice = scope_name.split('/')[-1]
            module_type = scope_slice.split('[')[0]
            module_to_type[module_name] = module_type
            if module_name in module_to_nodes:
                module_to_nodes[module_name].append(node)
            else:
                module_to_nodes[module_name] = [node]
        # for each module, find its inputs and outputs
        # build module mapping, from module name to its inputs debugName and outputs debugName,
        module_to_inputs = dict()
        module_to_outputs = dict()
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
            module_to_inputs[module_name] = m_inputs
            module_to_outputs[module_name] = m_outputs
        return output_to_node, input_to_node, module_to_inputs, module_to_outputs, module_to_type

    def _do_module_replace(self, module_name, mask=None, in_shape=None, out_shape=None):
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
        return input_cmask, output_cmask

    def _find_predecessors(self, module_name):
        """
        """
        predecessors = []
        for _input in self.module_to_inputs[module_name]:
            assert _input in self.output_to_node
            node = self.output_to_node[_input]
            #print("node: ", node)
            scope_name = node.scopeName() # example: scope_name, 'MyCell/Linear[linear]'
            #print("scope name: ", scope_name)
            if scope_name == '':
                continue
            module_name_slices = re.findall(r'\[(.*?)\]', scope_name)
            module_name = '.'.join(module_name_slices)
            if module_name == '':
                raise RuntimeError("_find_predecessors: cannot handle non-module node!")
            else:
                predecessors.append(module_name)
        return predecessors

    def _find_successors(self, module_name):
        """
        """
        successors = []
        for output in self.module_to_outputs[module_name]:
            assert output in self.input_to_node
            node = self.input_to_node[output]
            scope_name = node.scopeName()
            if scope_name == '':
                continue
            module_name_slices = re.findall(r'\[(.*?)\]', scope_name)
            module_name = '.'.join(module_name_slices)
            if module_name == '':
                raise RuntimeError("_find_successors: cannot handle non-module node!")
            else:
                successors.append(module_name)
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

        m_type = self.module_to_type[module_name]
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
