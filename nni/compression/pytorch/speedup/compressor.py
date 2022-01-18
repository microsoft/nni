# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
from pathlib import Path
import queue

import torch
import torch.nn as nn

from nni.common.graph_utils import build_module_graph
from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict
from nni.compression.pytorch.utils.utils import get_module_by_name
from .compress_modules import replace_module
from .infer_mask import AutoMaskInference
from .jit_translate import jit_to_python_function
from ..utils import rand_like_with_shape


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class ModelSpeedup:
    """
    This class is to speedup the model with provided weight mask.

    Parameters
    ----------
    model : pytorch model
        The model user wants to speed up
    dummy_input : pytorch tensor, tuple of tensor, list of tensor
        Note: The first dimension of the dummy_input should be the batchsize.
        The dummy input for ```jit.trace```, users should put it on the right
        device.
    masks_file : str/dict
        The path of user provided mask file, or the mask object
    map_location : str
        the device on which masks are placed, same to map_location in ```torch.load```
    batch_dim : int
        the index of batch dimension in the dummy_input
    confidence: the confidence coefficient of the sparsity inference. This value is
        actually used as the batchsize of the dummy_input.
    """

    def __init__(self, model, dummy_input, masks_file, map_location=None,
                 batch_dim=0, confidence=8):
        assert confidence > 1
        # The auto inference will change the values of the parameters in the model
        # so we need make a copy before the mask inference
        self.ori_state_dict = copy.deepcopy(model.state_dict())
        self.bound_model = model
        self.inferred_masks = dict()  # key: module_name, value: ModuleMasks
        self.batch_dim = batch_dim
        self.dummy_input, self.device = self._random_model_input(dummy_input, confidence, batch_dim)
        self.torch_graph = build_module_graph(model, self.dummy_input)
        # dict object to save the auto inferences objects of the submodules
        self.auto_inferences = {}
        # the index dict to find the corresponding torch._C.Value object
        # according to the debug name
        # we need the dummy_input to infer the mask automaticlly, so we save
        # the indexes from tensor's debugname to the torch._C.Value object.
        self.debugname_to_value = {}
        # load the mask tensor to the same device with the dummy_input
        # self.masks save the mask tensors pruned by the user and the infered
        # masks of the others modules
        if isinstance(masks_file, (str, Path)) and Path(masks_file).exists():
            self.masks = torch.load(
                masks_file, map_location if map_location is not None else str(self.device))
        elif isinstance(masks_file, dict):
            self.masks = masks_file
        else:
            raise Exception('Please provide the mask or the path of the mask file')
        self.constant = {}
        # self.internal_result save the internal output of the submodules
        self.internal_result = {}

    def _random_model_input(self, dummy_input, confidence, batch_dim):
        """
        Get the new random dummy input accordint to the original dummy_input
        and confidence, batch_dim.

        Parameters
        ----------
        dummy_input: Tensor or list/dict of Tensors
            The dummy_input given by the user.
        confidence: int
            The new batch size of the generated dummy_input.
        batch_dim: int
            The index of the batch dimension.

        Returns
        ------
        new_dummy_input: Tensor or list/dict of Tensors
            The generated dummy_input for mask inference.
        device: torch.device
            The device of the generated dummy_inputs
        """
        input_errmsg = 'Only support the tensor, list/tuple/dict of tensors as input'
        # Some model may use list of tensors as input, for example transformers
        new_dummy_input, device = None, None
        if isinstance(dummy_input, torch.Tensor):
            input_shape = list(dummy_input.size())
            # set the batchsize to the confidence ratio
            input_shape[batch_dim] = confidence
            new_dummy_input = rand_like_with_shape(input_shape, dummy_input)
            device = dummy_input.device
        elif isinstance(dummy_input, (tuple, list)):
            # else if the dummy input is list/tuple
            new_dummy_input = []
            old_batchsize = dummy_input[0].size(0)
            device = dummy_input[0].device
            for _, t_input in enumerate(dummy_input):
                assert isinstance(t_input, torch.Tensor), input_errmsg
                assert t_input.size(0) == old_batchsize, 'The first dimension should be batchsize\
                    and the batchsize of all inputs should be the same!'
                input_shape = list(t_input.size())
                input_shape[batch_dim] = confidence
                # rand_func = torch.randint if t_input.dtype
                new_dummy_input.append(
                    rand_like_with_shape(input_shape, t_input))
        elif isinstance(dummy_input, dict):
            new_dummy_input = {}
            tmp_key = list(dummy_input.keys())[0]
            old_batchsize = dummy_input[tmp_key].size(0)
            device = dummy_input[tmp_key].device
            for in_name, t_input in dummy_input.items():
                assert isinstance(t_input, torch.Tensor), input_errmsg
                assert old_batchsize == t_input.size(0), 'The first dimension should be batchsize\
                and the batchsize of all inputs should be the same!'
                input_shape = list(t_input.size())
                input_shape[batch_dim] = confidence
                new_dummy_input[in_name] = rand_like_with_shape(
                    input_shape, t_input)
        else:
            raise TypeError(input_errmsg)
        return new_dummy_input, device

    def _prepare_dummy_input(self, node):
        """
        Prepare the dummy_input for the auto mask inference.

        Parameters
        ----------
        node: NodePyGroup

        Returns
        -------
        dummy_input: list
            List of tensors that will be used as input for the target node.
        debugnames: list of strs
            Debugnames of the dummy_inputs.
        """
        _logger.debug('Prepare auto mask inference for node: %s',
                      node.unique_name)

        # prepare the inputs and outputs mask for this node,
        # if there is already a mask in self.masks, then use
        # the original mask tensor, else create a new one.
        inputs_name = node.inputs
        # build the dummy_input, in_masks the target node
        dummy_input = []
        debugnames = []
        for _input in inputs_name:
            if _input not in self.internal_result:
                # if the input debug name is not in self.internal_result,
                # then this node isn't a output tensor of any predecessor
                # nodes. This node is a attribute of the submodule, such as
                # weight or bias, etc. We will skip these tensors.
                # If we don't want this specific judgement here, we can merge
                # the `prim::GetAttr` node of the weight/bias tensor into the key
                # node, such as `conv`.
                # This is caused by the `meage_module_node` function in the
                # _graph_utils.py, because it doesn't merge the prim::GetAttr
                # node into the key node. In current version of _graph_utils.py,
                # we will only merge the nodes that have same scope name, however,
                # the scope name of the correponding prim::GetAttr node of `weight` tensor
                # is None.
                continue
            # The detach operation here is for the in-place operation. We cannot
            # directly can the backward on the output tensor of an in-place operator.
            dummy_input.append(self.internal_result[_input].detach())
            debugnames.append(_input)

        return dummy_input, debugnames

    def update_direct_sparsity(self, node):
        """
        Update the direct sparsity for the target node. Here the direct sparsity
        means that the sparsity in the output tensor that caused by the sparsity
        in the input tensors/weight tensors.
        """
        # this name is consistent with the name returned by named_modules()
        module_name = node.name
        _logger.info('Update mask for %s', module_name)
        unique_name = node.unique_name
        dummy_input, input_debugname = self._prepare_dummy_input(node)
        # get the input mask from self.masks
        # Note: the input mask of the successor nodes are
        # already created by the predecessor node
        in_masks = [self.masks[debugname] for debugname in input_debugname]
        in_constants = [self.constant[debugname]
                        for debugname in input_debugname]
        if node.type == 'func':
            # we cannot get the runable function directly from the jit traced
            # graph, so we translate it back to python function, Note: the function
            # is appliable to both cpu/gpu devices, the output tensors will be on the
            # same device of the input tensors
            func = jit_to_python_function(node, self)
            if func is None:
                # no need to infer the sparsity for this node
                self.auto_inferences[unique_name] = None
                return
            # function doesn't have weights
            _auto_infer = AutoMaskInference(
                func, dummy_input, in_masks, in_constants=in_constants, batch_dim=self.batch_dim)
        else:
            weight_mask = None
            if module_name in self.masks:
                weight_mask = self.masks[module_name]
            _, module = get_module_by_name(self.bound_model, module_name)
            _auto_infer = AutoMaskInference(
                module, dummy_input, in_masks, weight_mask, in_constants=in_constants,
                state_dict=copy.deepcopy(module.state_dict()), batch_dim=self.batch_dim)
        self.auto_inferences[unique_name] = _auto_infer
        _auto_infer.name = node.unique_name

        _auto_infer.update_direct_sparsity()
        # also save the input debug names into the auto_infer
        _auto_infer.input_debugname = input_debugname
        # update the mask tensor and the internal output of the submodules
        # after manually unpack the tuple/list of tensors, the number of the outputs
        # of each node should always be one(Except for the TupleUnpack node at the end
        # of the whole model)
        assert len(
            node.outputs) == 1, 'The number of the output should be one after the Tuple unpacked manually'

        out_debugname = node.outputs[0]
        # update the output mask into self.masks
        self.masks[out_debugname] = _auto_infer.output_mask
        self.constant[out_debugname] = _auto_infer.out_constant
        # update the output result into self.internal_result, so that
        # the successor nodes can take these output tensors as inputs.
        self.internal_result[out_debugname] = _auto_infer.output
        # update the parameter mask of the node

        self.masks[module_name] = _auto_infer.weight_mask

    def update_indirect_sparsity(self, node):
        """
        This function will update the indirect sparsity. To explain what's
        indirect sparsity, for example, there is two tensors TA and TB, and
        we perform the calculation: TC = TA x TB in which TC is also a tensor.
        Once some values in TA are masked to zeros, then the corresponding
        positions in TB are also potential sparsities, because these have no
        effect of the final output(the gradient of these positions in TB equal
        to 0 all the time). This function it to fine the potential sparsity caused
        by other sparsity(we call it indirect sparsity here). Basically we can find
        these potential sparsity through gradient.

        Parameters
        ---------
        node: the NodePy
            The target node to update the indirect sparsity
        """
        unique_name = node.unique_name
        if unique_name in self.auto_inferences and self.auto_inferences[unique_name] is not None:
            # if the auto inference object already in self.auto_inference, then
            # directly update the previous one
            # self.auto_inferences[unique_name].update()
            _logger.info(
                'Update the indirect sparsity for the %s', unique_name)
            auto_infer = self.auto_inferences[unique_name]
            auto_infer.update_indirect_sparsity()
            # pass the gradient to the predecessor nodes
            for in_id, tin in enumerate(auto_infer.dummy_input):
                debug_name = auto_infer.input_debugname[in_id]
                last_output = self.internal_result[debug_name]
                # if isinstance(last_output, torch.Tensor):
                # TODO what if last output is tuple/list of tensor
                if last_output.grad is not None and tin.grad is not None:
                    last_output.grad.data += tin.grad.data
                else:
                    last_output.grad = tin.grad
        else:
            _logger.warning('Note: %s does not have corresponding mask inference object', node.name)

    def _vnode_to_value(self, c_node):
        """
        translate the C Value node into the values/tensors.
        """
        errmsg = "Only support the torch._C.Value type"
        assert isinstance(c_node, torch._C.Value), errmsg
        if isinstance(c_node.type(), torch._C.TensorType):
            shape = tuple(c_node.type().sizes())
            dtype = c_node.type().scalarType()
            # TODO should use a more general way to get the input
            if dtype.startswith('Float') or dtype.startswith('Double'):
                return torch.rand(shape).to(self.device)
            else:
                # This small range is due to the ·ReLU6·, we will add
                # the manual specific mask inference rule for several
                # ops in the future, so that we can remove the constraint.
                return torch.randint(0, 10, shape, device=self.device)
        else:
            value = c_node.toIValue()
            # TODO support more kinds of value node
            errmsg = "Doesn't support convert %s to values", str(c_node.type())
            # currently only support the tensors and constant values
            assert value is not None, errmsg
            return value

    def infer_modules_masks(self):
        """
        Infer the mask for all layers in the module, this function can be divided into
        two steps: first, forward inference of the the masks. Second, backward inference
        of the mask. We keep repeating these two steps until the masks of the model doesn't
        change.
        """
        # unpack the tensor tuple/list before the mask inference
        self.torch_graph.unpack_manually()
        # find the input/ouput tensor of the whole graph
        graph_input = []
        graph_output = []
        for name, nodeio in self.torch_graph.nodes_py.nodes_io.items():
            if nodeio.input_or_output == 'input':
                graph_input.append((name, nodeio))
                # also put the graph input tensor into the internal_result
                # TODO if we can find the corresponding relation between the value node
                # and the dummy_inputs, we can use the inputs value in the dummy_input
                value = self._vnode_to_value(self.debugname_to_value[name])
                self.internal_result[name] = value
                # create the mask tensor for the input value
                if isinstance(self.internal_result[name], torch.Tensor):
                    self.masks[name] = torch.ones_like(value)
                    self.constant[name] = torch.zeros_like(value)
            elif nodeio.input_or_output == 'output':
                graph_output.append((name, nodeio))
        # count the degree for the node in the graph
        in_degree = {}
        out_degree = {}
        visit_queue = queue.Queue()
        for node in self.torch_graph.nodes_py.nodes_op:
            successors = self.torch_graph.find_successors(node.unique_name)
            out_degree[node.unique_name] = len(successors)
            predecessors = self.torch_graph.find_predecessors(node.unique_name)
            in_degree[node.unique_name] = len(predecessors)
            if in_degree[node.unique_name] == 0:
                visit_queue.put(node)
        # Forward mask inference
        while not visit_queue.empty():
            curnode = visit_queue.get()
            # forward mask inference for curnode
            self.update_direct_sparsity(curnode)
            successors = self.torch_graph.find_successors(curnode.unique_name)
            for successor in successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    visit_queue.put(self.torch_graph.name_to_node[successor])
        # backward mask inference
        for unique_name in out_degree:
            if out_degree[unique_name] == 0:
                visit_queue.put(self.torch_graph.name_to_node[unique_name])
        while not visit_queue.empty():
            curnode = visit_queue.get()
            self.update_indirect_sparsity(curnode)
            predecessors = self.torch_graph.find_predecessors(
                curnode.unique_name)
            for predecessor in predecessors:
                out_degree[predecessor] -= 1
                if out_degree[predecessor] == 0:
                    visit_queue.put(self.torch_graph.name_to_node[predecessor])

    def replace_compressed_modules(self):
        """
        Replace all the modules that have changed (weights/inputs/output) shape.
        The new module is created using the same arguments of the to-be-replaced module,
        and correctly inherits its weights.

        NOTE: ```func``` type cannot be replaced as it is not a module, thus, one limitation
        is that ```func``` should be not required to be replaced.
        """
        with torch.no_grad():
            for unique_name in self.auto_inferences:
                self.replace_submodule(unique_name)

    def replace_submodule(self, unique_name, reindex_dim=None, reindex=None):
        """
        Replace the submodule according to the inferred sparsity.
        unique_name: str
            The unique_name of the submodule to replace.
        reindex_dim: int
            The dimension of the re-index operation.
        reindex: Reindex
            The index tensor. Normally this variable is None. If we want to reindex the
            output of this submodule, we can pass the index by this parameter.
        """
        class ReindexModule(nn.Module):
            """
            ReindexModule is used to resolve the mask conflict when replace the submodule.
            Basically, we can use two ways to resolve the mask conflict: (1) unmask some
            values(will introduce more computation overhead) (2) reindex and padd the output
            tensor of the target op(introduce more memory access overhad). Currently this
            method is shutdown, in the future, we will merge these two methods into a graph
            pass which is used to resolve the mask conflict.
            """
            def __init__(self, ori_module, reindex_dim, reindex):
                super(ReindexModule, self).__init__()
                self.ori_module = ori_module
                self.reindex_dim = reindex_dim
                self.reindex = reindex
                tmp_index = [slice(None, None) for i in range(reindex_dim+1)]
                # the index for the tensor
                tmp_index[reindex_dim] = reindex
                self.t_index = tuple(tmp_index)

            def forward(self, x):
                tmpout = self.ori_module(x)
                shape = list(tmpout.size())
                shape[self.reindex_dim] = self.reindex.size(0)
                out = torch.zeros(tuple(shape), device=tmpout.device,
                                  requires_grad=tmpout.requires_grad)
                out[self.t_index] = tmpout
                return out

        assert unique_name in self.auto_inferences
        g_node = self.torch_graph.name_to_node[unique_name]
        _logger.debug("replace %s, in %s type, with op_type %s",
                      unique_name, g_node.type, g_node.op_type)
        auto_infer = self.auto_inferences[unique_name]
        if g_node.type == 'module':
            if g_node.unique_name in self.torch_graph.reused_module:
                if reindex_dim is not None:
                    _logger.warning(
                        'Cannot replace a reused module with padding operator!!')
                    return None
            super_module, leaf_module = get_module_by_name(
                self.bound_model, g_node.name)
            m_type = g_node.op_type
            if not m_type in replace_module:
                raise RuntimeError(
                    "Has not supported replacing the module: `{}`".format(m_type))
            _logger.info("replace module (name: %s, op_type: %s)",
                         g_node.name, m_type)
            compressed_module = replace_module[m_type](
                leaf_module, auto_infer.get_masks())
            new_submodule = compressed_module
            if reindex_dim is None:
                setattr(super_module, g_node.name.split(
                    '.')[-1], compressed_module)
            elif reindex_dim is not None and reindex is not None:
                # reindex the output of this submodule and replace the orginal module
                new_submodule = ReindexModule(
                    compressed_module, reindex_dim, reindex)
                setattr(super_module, g_node.name.split(
                    '.')[-1], new_submodule)
            return new_submodule
        elif g_node.type == 'func':
            _logger.info("Warning: cannot replace (name: %s, op_type: %s) which is func type",
                         unique_name, g_node.op_type)
            return None
        else:
            raise RuntimeError("Unsupported node type: {}".format(g_node.type))

    def initialize_speedup(self):
        """
        Do some initial work for speedup.
        """
        # initialize the self.debugname_to_value
        # build a mapping table from the debug name of the tensor
        # to its value node in the graph
        traced_graph = self.torch_graph.trace.graph
        for node in traced_graph.nodes():
            for _input in node.inputs():
                debug_name = _input.debugName()
                if debug_name not in self.debugname_to_value:
                    self.debugname_to_value[debug_name] = _input
            for _output in node.outputs():
                debug_name = _output.debugName()
                if debug_name not in self.debugname_to_value:
                    self.debugname_to_value[debug_name] = _output
        # put the model itself into internel_result to perform the
        # value inference for the 'prim::GetAttr', the first ClassType
        # of the whole graph is the model class

        for graph_input in traced_graph.inputs():
            if graph_input.type().kind() == 'ClassType':
                self.internal_result[graph_input.debugName()
                                     ] = self.bound_model
                break

    def speedup_model(self):
        """
        There are basically two steps: first, do mask/shape inference,
        second, replace modules.
        """

        _logger.info("start to speed up the model")
        self.initialize_speedup()
        training = self.bound_model.training
        # set to the evaluation mode
        self.bound_model.train(False)
        # TODO suppose to fix the conflict after the sparsity propagation
        # which is more elegent
        fix_mask_conflict(self.masks, self.bound_model, self.dummy_input)

        _logger.info("infer module masks...")
        self.infer_modules_masks()
        _logger.info('resolve the mask conflict')

        # load the original stat dict before replace the model
        self.bound_model.load_state_dict(self.ori_state_dict)
        _logger.info("replace compressed modules...")
        # the mask conflict should be already resolved
        self.replace_compressed_modules()
        self.bound_model.train(training)
        _logger.info("speedup done")
