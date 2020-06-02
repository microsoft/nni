# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from nni._graph_utils import build_module_graph
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

class ModelSpeedup:
    """
    This class is to speedup the model with provided weight mask
    """

    def __init__(self, model, dummy_input, masks_file, map_location=None):
        """
        Parameters
        ----------
        model : pytorch model
            The model user wants to speed up
        dummy_input : pytorch tensor
            The dummy input for ```jit.trace```, users should put it on right device before pass in
        masks_file : str
            The path of user provided mask file
        map_location : str
            the device on which masks are placed, same to map_location in ```torch.load```
        """
        self.bound_model = model
        self.masks = torch.load(masks_file, map_location)
        self.inferred_masks = dict() # key: module_name, value: ModuleMasks
        self.torch_graph = build_module_graph(model, dummy_input)

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
            The name of the node
        mask : tensor of mask or ModuleMasks
            Mask of the weights in this node (i.e., module)
        in_shape : ModuleMasks
            Input shape of this node
        out_shape : ModuleMasks
            Output shape of this node
        """
        input_cmask = output_cmask = None
        if module_name in self.inferred_masks:
            module_masks = self.inferred_masks[module_name]
        else:
            module_masks = ModuleMasks(module_name)
            self.inferred_masks[module_name] = module_masks

        m_type = self.torch_graph.name_to_node[module_name].op_type
        _logger.debug("infer mask of module %s with op_type %s", module_name, m_type)
        if mask is not None:
            _logger.debug("mask is not None")
            if not m_type in infer_from_mask:
                raise RuntimeError(
                    "Has not supported infering input/output shape from mask for module/function: `{}`, {}"
                    .format(m_type, module_name))
            input_cmask, output_cmask = infer_from_mask[m_type](module_masks, mask)
        if in_shape is not None:
            _logger.debug("in_shape is not None")
            if not m_type in infer_from_inshape:
                raise RuntimeError(
                    "Has not supported infering output shape from input shape for module/function: `{}`, {}"
                    .format(m_type, module_name))
            if m_type in ['aten::view', 'aten::flatten']:
                output_cmask = infer_from_inshape[m_type](module_masks,
                                                          in_shape,
                                                          self.torch_graph.name_to_node[module_name].auxiliary)
            else:
                output_cmask = infer_from_inshape[m_type](module_masks, in_shape)
        if out_shape is not None:
            _logger.debug("out_shape is not None")
            if not m_type in infer_from_outshape:
                raise RuntimeError(
                    "Has not supported infering input shape from output shape for module/function: `{}`, {}"
                    .format(m_type, module_name))
            input_cmask = infer_from_outshape[m_type](module_masks, out_shape)

        if input_cmask:
            predecessors = self.torch_graph.find_predecessors(module_name)
            for _module_name in predecessors:
                self.infer_module_mask(_module_name, out_shape=input_cmask)
        if output_cmask:
            successors = self.torch_graph.find_successors(module_name)
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
            g_node = self.torch_graph.name_to_node[module_name]
            _logger.debug("replace %s, in %s type, with op_type %s",
                          module_name, g_node.type, g_node.op_type)
            if g_node.type == 'module':
                super_module, leaf_module = get_module_by_name(self.bound_model, module_name)
                m_type = g_node.op_type
                if not m_type in replace_module:
                    raise RuntimeError("Has not supported replacing the module: `{}`".format(m_type))
                _logger.info("replace module (name: %s, op_type: %s)", module_name, m_type)
                compressed_module = replace_module[m_type](leaf_module, self.inferred_masks[module_name])
                setattr(super_module, module_name.split('.')[-1], compressed_module)
            elif g_node.type == 'func':
                _logger.info("Warning: cannot replace (name: %s, op_type: %s) which is func type",
                             module_name, g_node.op_type)
            else:
                raise RuntimeError("Unsupported node type: {}".format(g_node.type))

    def speedup_model(self):
        """
        There are basically two steps:
        first, do mask/shape inference,
        second, replace modules
        """
        training = self.bound_model.training
        _logger.info("start to speed up the model")
        _logger.info("infer module masks...")
        self.infer_modules_masks()
        _logger.info("replace compressed modules...")
        self.replace_compressed_modules()
        self.bound_model.train(training)
        _logger.info("speedup done")
