# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import re
import torch
from . import default_layers

_logger = logging.getLogger(__name__)


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
        self.output_to_node, self.input_to_node, self.module_inputs, self.module_outputs = self._build_graph()

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
            if module_name in module_to_nodes:
                module_to_nodes[module_name].append(node)
            else:
                module_to_nodes[module_name] = [node]
        # for each module, find its inputs and outputs
        # build module mapping, from module name to its inputs debugName and outputs debugName,
        module_inputs = dict()
        module_outputs = dict()
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
            module_inputs[module_name] = m_inputs
            module_outputs[module_name] = m_outputs
        return output_to_node, input_to_node, module_inputs, module_outputs

    def _do_module_replace(self, module_name, mask=None, in_shape=None, out_shape=None):
        """
        """
        changed_in_shape = changed_out_shape = None
        assert module_name in self.module_inputs, "module does not exist in trace graph"
        if mask is not None:
            assert in_shape is None and out_shape is None
        # fine-grained tensor sparse
        #...
        # coarse-grained shape sparse
        #...
        if in_shape is not None:
            #...
        if out_shape is not None:
            #...
        return changed_in_shape, changed_out_shape

    def _find_predecessors(self):
        """
        """

    def _find_successors(self):
        """
        """

    def replace_module(self, module_name, mask=None, in_shape=None, out_shape=None):
        """
        """
        changed_in_shape, changed_out_shape = self._do_module_replace(module_name, mask, in_shape, out_shape)
        if changed_in_shape:
            predecessors = self._find_predecessors()
            for module_name in predecessors:
                self.replace_module(module_name, out_shape=changed_in_shape)
        if changed_out_shape:
            successors = self._find_successors()
            for module_name in successors:
                self.replace_module(module_name, in_shape=changed_out_shape)

    def speedup_model(self):
        """
        """
        for name, mask in self.masks:
            self.replace_module(name, mask=mask)
