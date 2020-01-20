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
        # build output mapping, from output debugName to its node
        output_to_node = dict()
        # build input mapping, from input debugName to its node
        input_to_node = dict()
        #build module mapping, from module name to all nodes (as list) under this module scope
        module_to_nodes = dict()
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
            # TODO: check module_name is not empty
            if module_name in module_to_nodes:
                module_to_nodes[module_name].append(node)
            else:
                module_to_nodes[module_name] = [node]
        # for each module, find its inputs and outputs
        # build module mapping, from module name to its inputs debugName and outputs debugName,
        module_to_inputs = dict()
        module_to_outputs = dict()
        # TODO: fullfill modules_type
        module_to_type = dict()
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
                if not input_to_node[output] in nodes:
                    m_outputs.append(output)
            for _input in inputs:
                if not output_to_node[_input] in nodes:
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
            assert _input in self.input_to_node
            node = self.input_to_node[_input]
            scope_name = node.scopeName() # example: scope_name, 'MyCell/Linear[linear]'
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
        if mask is not None:
            input_cmask, output_cmask = infer_from_mask[m_type](module_masks, mask)
        if in_shape is not None:
            infer_from_inshape[m_type](module_masks, in_shape)
        if out_shape is not None:
            infer_from_outshape[m_type](module_masks, out_shape)

        if input_cmask:
            predecessors = self._find_predecessors(module_name)
            for module_name in predecessors:
                self.infer_module_mask(module_name, out_shape=input_cmask)
        if output_cmask:
            successors = self._find_successors(module_name)
            for module_name in successors:
                self.infer_module_mask(module_name, in_shape=output_cmask)

    def infer_modules_masks(self):
        """
        """
        for module_name, mask in self.masks:
            self.infer_module_mask(module_name, mask=mask)
