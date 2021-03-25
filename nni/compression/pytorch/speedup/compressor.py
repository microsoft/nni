# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import queue
import logging
import torch
import torch.nn as nn
import numpy as np
import copy
from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict
from nni.compression.pytorch.utils.utils import get_module_by_name
from .compress_modules import replace_module
from .infer_mask import AutoMaskInference
from .jit_translate import jit_to_python_function
from ..utils.shape_dependency import ADD_TYPES, CAT_TYPE, MUL_TYPES
from ..utils import rand_like_with_shape
from .sparsity_conflicts import calc_unmask, calc_padding
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


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

    def __init__(self, model, dummy_input, masks_file, map_location=None, batch_dim=0, confidence=8, fold_bias=False, enable_compile=False):
        """
        Parameters
        ----------
        model : pytorch model
            The model user wants to speed up
        dummy_input : pytorch tensor
            Note: The first dimension of the dummy_input should be the batchsize.
            The dummy input for ```jit.trace```, users should put it on the right
            device before pass in
        masks_file : str
            The path of user provided mask file
        map_location : str
            the device on which masks are placed, same to map_location in ```torch.load```
            confidence: the confidence coefficient of the sparsity inference. This value is
            actually used as the batchsize of the dummy_input.
        confidence: int
            The number of examples used to infer the mask.
        enable_compile: bool
            If this flag is enabled, we will modify the network architecture to resolve
            the sparsity conflict.
        """
        assert confidence > 1
        from nni.common.graph_utils import build_module_graph
        # The auto inference will change the values of the parameters in the model
        # so we need make a copy before the mask inference
        self.ori_state_dict = copy.deepcopy(model.state_dict())
        self.bound_model = model
        self.inferred_masks = dict()  # key: module_name, value: ModuleMasks
        self.batch_dim = batch_dim
        self._random_model_input(dummy_input, confidence, batch_dim)
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
        self.masks = torch.load(masks_file, map_location if map_location is not None else str(self.device))
        
        self.constant = {}
        # self.internal_result save the internal output of the submodules
        self.internal_result = {}
        # if we enable the compilation of the sparsity, then we will modify the network
        # architecture to resolve the sparsity conflict.
        self.fold_bias = fold_bias
        self.enable_compile = enable_compile
    



    def _random_model_input(self, dummy_input, confidence, batch_dim):
        input_errmsg = 'Only support the tensor, list/tuple/dict of tensors as input'
        # Some model may use list of tensors as input, for example transformers
        if isinstance(dummy_input, torch.Tensor):
            input_shape = list(dummy_input.size())
            # set the batchsize to the confidence ratio
            input_shape[batch_dim] = confidence
            self.dummy_input = rand_like_with_shape(input_shape, dummy_input)
            self.device = dummy_input.device
        elif isinstance(dummy_input, (tuple, list)):
            # else if the dummy input is list/tuple
            self.dummy_input = []
            old_batchsize = dummy_input[0].size(0)
            self.device = dummy_input[0].device
            for _, t_input in enumerate(dummy_input):
                assert isinstance(t_input, torch.Tensor), input_errmsg
                assert t_input.size(0) == old_batchsize, 'The first dimension should be batchsize\
                    and the batchsize of all inputs should be the same!'
                input_shape = list(t_input.size())
                input_shape[batch_dim] = confidence
                # rand_func = torch.randint if t_input.dtype 
                self.dummy_input.append(rand_like_with_shape(input_shape, t_input))
        elif isinstance(dummy_input, dict):
            self.dummy_input = {}
            tmp_key = list(dummy_input.keys())[0]
            old_batchsize = dummy_input[tmp_key].size(0)
            self.device = dummy_input[tmp_key].device
            for in_name, t_input  in dummy_input.items():
                assert isinstance(t_input, torch.Tensor), input_errmsg
                assert old_batchsize == t_input.size(0), 'The first dimension should be batchsize\
                and the batchsize of all inputs should be the same!'      
                input_shape = list(t_input.size())
                input_shape[batch_dim] = confidence
                self.dummy_input[in_name] = rand_like_with_shape(input_shape, t_input)
        else:
            raise TypeError(input_errmsg)


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
            # TODO why detach??
            # TODO what if a list/tuple of tensor
            dummy_input.append(self.internal_result[_input].detach())
            debugnames.append(_input)
            # v_node = self.debugname_to_value[_input]
            # if isinstance(v_node.type(), torch._C.TensorType) and \
            #         'prim::GetAttr' not in v_node.node().kind():
            #     # Filter the value nodes created by the prim::GetAttr, such as
            #     # weight and bias tensor should be skipped

            #     # print(v_node.type().sizes())
            #     # print(v_node)
            #     # print(v_node.node())
            #     shape = tuple(v_node.type().sizes())
            #     # Note: cannot support the value-dependent models
            #     dummy_input.append((torch.rand(shape).to(self.device), _input))
            #     if _input not in self.masks:
            #         # if the input tensor doesn't have masks, then create one
            #         self.masks[_input] = torch.ones(shape).to(self.device)

        return dummy_input, debugnames

    def update_direct_sparsity(self, node):
        """
        Update the mask for the target node.
        """
        AutoMaskInferenceClass = AutoMaskInference
        # print("Creating auto inference for")
        # print(node.unique_name)
        # print(node.op_type)
        # print(AutoMaskInferenceClass)
        # this name is consistent with the name returned by named_modules()
        module_name = node.name
        _logger.info('Update mask for %s', module_name)
        unique_name = node.unique_name
        # if it is the first visit to this node, then we create a corresponding auto
        # mask inference object for this node
        dummy_input, input_debugname = self._prepare_dummy_input(node)
        # get the input mask from self.masks
        # Note: the input mask of the successor nodes are
        # already created by the predecessor node
        in_masks = [self.masks[debugname] for debugname in input_debugname]
        in_constants = [self.constant[debugname]
                        for debugname in input_debugname]
        if node.type == 'func':
            # we cannot get the runable function directly from the jit traced
            # graph, so we translate it back to python function
            func = jit_to_python_function(node, self)
            if func is None:
                # no need to infer the sparsity for this node
                return
            # function doesn't have weights
            _auto_infer = AutoMaskInferenceClass(
                func, dummy_input, in_masks, in_constants=in_constants, batch_dim = self.batch_dim)
        else:
            # node.type == 'module'
            weight_mask = None
            if module_name in self.masks:
                weight_mask = self.masks[module_name]
            _, module = get_module_by_name(self.bound_model, module_name)
            _auto_infer = AutoMaskInferenceClass(
                module, dummy_input, in_masks, weight_mask, in_constants=in_constants, \
                state_dict=copy.deepcopy(module.state_dict()), batch_dim=self.batch_dim)
        self.auto_inferences[unique_name] = _auto_infer
        _auto_infer.name = node.unique_name
        _auto_infer.fold_bias = self.fold_bias
        # _auto_infer.update()
        _auto_infer.update_direct_sparsity()
        # also save the input debug names into the auto_infer
        _auto_infer.input_debugname = input_debugname
        # update the mask tensor and the internal output of the submodules
        # after manually unpack the tuple/list of tensors, the number of the outputs
        # of each node should always be one

        assert len(
            node.outputs) == 1, "The number of the outputs of %s is not 1" % module_name
        out_debugname = node.outputs[0]
        # update the output mask into self.masks
        self.masks[out_debugname] = _auto_infer.output_mask
        self.constant[out_debugname] = _auto_infer.out_constant
        # update the output result into self.internal_result, so that
        # the successor nodes can take these output tensors as inputs.
        self.internal_result[out_debugname] = _auto_infer.output
        # update the parameter mask of the node
        # print(self.masks.keys())
        self.masks[module_name] = _auto_infer.weight_mask

    def update_indirect_sparsity(self, node):
        """
        update the indirect sparisty for the target node.
        """
        module_name = node.name
        _logger.info('Update indirect sparsity for %s', module_name)
        unique_name = node.unique_name
        if unique_name in self.auto_inferences:
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
            # TODO ugly code here
            if dtype.startswith('Float') or dtype.startswith('Double'):
                return torch.rand(shape).to(self.device)
            else:
                return torch.randint(0, 10, shape, device=self.device)
        else:
            value = c_node.toIValue()
            # TODO support more kinds of value node
            errmsg = "Doesn't support convert %s to values", str(cnode.type())
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

        # Backwards mask inference
        # for module_name, mask in self.masks.items():
        #     _logger.debug('Start mask inference from %s', module_name)
        #     if module_name not in self.torch_graph.name_to_node:
        #         # this module is not traced in the torch_graph,
        #         # jit.trace only correctly records functions and
        #         # modules which are not data dependent (e.g., do
        #         # not have conditionals on data in tensors)
        #         # so, if a node is not traced, we just skip it.
        #         _logger.warning('%s has mask, but not found in the traced graph, just skip it.', module_name)
        #         continue
        #     self.infer_module_mask(module_name, None, mask=mask)

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
                # print(unique_name)
                tmpout = self.ori_module(x)
                shape = list(tmpout.size())
                shape[self.reindex_dim] = self.reindex.size(0)
                out = torch.zeros(tuple(shape), device=tmpout.device,
                                  requires_grad=tmpout.requires_grad)
                # print(self.t_index)
                # print('Output shape')
                # print(shape)
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
            compressed_module = replace_module[m_type](leaf_module, auto_infer)
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

    def compile_sparse_modules(self):
        """
        Reconstruct the model according to the inferred sparsity, compared to modifying the mask to
        resolve the conflict, we will try to utilize as more sparsity as possible to speedup the whole
        model.
        Note: we may change the network architecture in this function. If the user prefer to keep
        the original network architecture, please set the `enable_compile` flag to false.
        In addtion, currently the compiling engine cannot support all possible models (for example,
        a tensor is taken as input by more than one ADD nodes), when this funtion fails, you can
        still speedup the model with enable_compile set to false.
        """

        with torch.no_grad():
            # build the out_degree table for the nodes in the model
            out_degree = {}
            visit_queue = queue.Queue()
            # Store the padding and unmask tensors
            padding_map = {}
            unmask_map = {}
            for node in self.torch_graph.nodes_py.nodes_op:
                successors = self.torch_graph.find_successors(node.unique_name)
                out_degree[node.unique_name] = len(successors)
                if out_degree[node.unique_name] == 0:
                    # if this node doesn't have any successor nodes
                    visit_queue.put(node)
            # backward traverse the model graph and find the operators that have shape
            # dependencies
            while not visit_queue.empty():
                # remain_padding is the unmask tensors passed by the successor nodes
                # for example, the relu node is a function and we cannot replace
                # the function node in the model, so the unmask tensor which should be
                # handled by the relu node can only be passed to the predecessor nodes.
                cur_node = visit_queue.get()
                # Get the padding tensors from the padding_map
                if cur_node.unique_name in padding_map:
                    remain_padding = padding_map[cur_node.unique_name]
                    remain_unmask = unmask_map[cur_node.unique_name]
                else:
                    remain_padding = None
                    remain_unmask = None
                # if this not is not replaced yet
                _logger.debug('Sparisty Compiling %s', cur_node.unique_name)
                # calculate the unmask tensor for this node

                _auto_infer = self.auto_inferences[cur_node.unique_name]
                assert isinstance(_auto_infer.output_mask, torch.Tensor)
                _remain_resolved = None
                if remain_padding is not None:
                    # since we change the output of this node by padding zeros, so
                    # we also need to unmask the correponding values in the _auto_infer.output_mask
                    # so that, the successor node will take the right shape. And this will interfere
                    # with the replacement of the current node (because the output_mask has been changed)
                    # So, we need to replace the modules that need the padding operators before
                    # those nodes who don't. We have to update the output_mask to make sure
                    # that the successor has the right input shape,
                    _remain_resolved = self.replace_submodule(
                        cur_node.unique_name, 1, remain_padding == False)

                if _remain_resolved is None:
                    new_padding, new_unmask = self.calc_paddingindex(
                        cur_node, remain_unmask)
                    if remain_padding is not None:
                        pos = remain_unmask > 0
                        _auto_infer.output_mask[pos] = 1
                else:
                    # we can resovle the conflict by reindex the output of this module,
                    # so we don't need to pass the remained padding zeros to the predecessor
                    # nodes
                    new_padding, new_unmask = self.calc_paddingindex(
                        cur_node, None)
                    if remain_padding is not None:
                        pos = remain_unmask > 0
                        _auto_infer.output_mask[pos] = 1
                    remain_padding = remain_unmask = None
                if new_padding is not None:
                    for i, tensor in enumerate(new_padding):
                        if tensor is not None:
                            # The reason why we use the input_debugname in the _auto_infer
                            # rather the cur_node.inputs is that the cur_node.inputs have
                            # the parameters of the modules (weight, bias), and this is caused by
                            # the merging rules when we build the TorchModuleGraph. In TorchModuleGraph
                            # we merge the node based on its scope name and the 'prim::GetAttr' node of
                            # weight tensor has no scope name.
                            debugname = _auto_infer.input_debugname[i]
                            predecessor = self.torch_graph.output_to_node[debugname]
                            out_degree[predecessor.unique_name] -= 1

                            if predecessor.unique_name not in padding_map:
                                padding_map[predecessor.unique_name] = new_padding[i]
                                unmask_map[predecessor.unique_name] = new_unmask[i]
                            else:
                                # NOTE: Currently, compiling cannot handle the situation that a tensor is broadcast
                                # to several add Ops(fix_mask_conflict can handle this scenario). We cannot decide the
                                # unmask/reindex tensor from only one Add operator. Fortunately, there is no such structure
                                # in common networks.
                                # TODO May support the secenario mentioned above in the future
                                assert all(
                                    new_padding[1] == padding_map[predecessor.unique_name])
                            if out_degree[predecessor.unique_name] == 0:
                                visit_queue.put(predecessor)
                else:
                    # No conflict found here
                    predecessors = self.torch_graph.find_predecessors(
                        cur_node.unique_name)
                    for predecessor in predecessors:
                        out_degree[predecessor] -= 1
                        if remain_padding is not None:
                            if predecessor not in padding_map:
                                padding_map[predecessor] = remain_padding
                                unmask_map[predecessor] = remain_unmask
                            else:
                                # Currently the compiling cannot handle the scenario that a op is the input of two add
                                # operators.
                                errmsg = "Compiling engine cannot compile the sparsity for this model, Please set \
                                     enable_compile to False and try again!"
                                assert all(remain_padding ==
                                           padding_map[predecessor]), errmsg
                        if out_degree[predecessor] == 0:
                            visit_queue.put(
                                self.torch_graph.name_to_node[predecessor])
            # for node in padding_map:
                # print('Pruned channel', node, torch.sum(padding_map[node]))
            # replace the submodule that don't need the padding operators
            # according the inferred sparsity
            # If there are some values that need be unmasked
            for unique_name in self.auto_inferences:
                if unique_name not in padding_map:
                    self.replace_submodule(unique_name)

    def calc_paddingindex(self, node, remain_unmask):
        """
        Calculate the reindex tensor for this node. If this node has sparsity
        conflict then we will return a reindex tensor for each of its input, else
        this function will just return None.
        Parameters
        ----------
        node: Node NodePyGroup
            The target node to caculate the reindex tenosr for its output.
        Returns
        -------
        padding: list of tensor
            List of tensor, each tensor indicates the channel-wise index of a input
            to padding the zeros. Note: this tensor is calculated based on the shape
            after the pruning.
        unmask: list of tensor

        """
        if node.op_type not in ADD_TYPES and node.op_type not in MUL_TYPES \
                and node.op_type != CAT_TYPE:
            return None, None
        unique_name = node.unique_name
        auto_infer = self.auto_inferences[unique_name]
        input_masks = auto_infer.in_masks
        output_mask = auto_infer.output_mask
        # The difference between the padding and the unmask is that
        # the padding is calculated based on the shape that after pruning,
        # and the unmask is calculated based on the shape that befer the actual
        # pruning.
        if remain_unmask is not None:
            padding = calc_padding(
                node, input_masks, output_mask + remain_unmask)
            unmask = calc_unmask(
                node, input_masks, output_mask + remain_unmask)
        else:
            padding = calc_padding(node, input_masks, output_mask)
            unmask = calc_unmask(node, input_masks, output_mask)
        return padding, unmask

    def need_to_unmask(self, node):
        """
        Check if this node has shape/sparsity conflict. If not, then
        return None, if so, return the values that need to be unmasked.

        Parameters
        ----------
        node: NodePyGroup
            The target node to check if need unmask some values.
        Returns
        -------
        unmask: list
            List of the values that need to be unmasked. In the list, each element
            is a tuple which contains the debugName of the tensor and the correponding
            values that need to be unmask in this tensor. For example, [(1, tensor[0, 1])],
            in this example, we need unmask the sencond value of the tensor 1.
        """
        if node.op_type not in ADD_TYPES and node.op_type not in MUL_TYPES \
                and node.op_type != CAT_TYPE:
            # only abobe operators may invovle shape dependencies
            return None
        unique_name = node.unique_name
        auto_infer = self.auto_inferences[unique_name]
        input_masks = auto_infer.in_masks
        output_mask = auto_infer.output_mask
        unmask = calc_unmask(node, input_masks, output_mask)
        # Reduce the mask of the whole tensor into the channel mask
        c_unmask = []
        for t_unmask in unmask:
            if t_unmask is not None:
                shape = list(t_unmask.size())
                dim_list = list(range(len(shape)))
                dim_list.remove(1)
                _count = np.prod(shape) / shape[1]
                # reduce the element-wise unmask tensor to the channel-wise unmask tensor
                sum_unmask = torch.sum(t_unmask, dim_list)
                c_unmask.append(sum_unmask == _count)
        return c_unmask

    def unmask_chain(self, debugname, t_unmask):
        """
        Unmask the values in the tensor specified by debugname.
        This function will also unmask the related dependent values in the
        predecessor nodes/tensors.
        Parameters
        ---------
        debugname: string
            The debugname of the target tensor.
        unmask: torch.Tensor
            This tensor indicates the values that need to be unmasked in the
            target tensor. This tensor only contains 0 and 1, 1-> need to be unmasked, 0
            -> leave it.
        """
        if debugname not in self.torch_graph.output_to_node:
            # already reach the dummy_inputs of the graph
            unmask_pos = t_unmask > 0
            self.masks[debugname][unmask_pos] = 1
            return
        # find corresponding auto inference object
        node = self.torch_graph.output_to_node[debugname]
        unique_name = node.unique_name
        # _logger.debug('Unmask the tensor %s of %s', debugname, unique_name)

        auto_infer = self.auto_inferences[unique_name]
        debugnames, unmasks = auto_infer.unmask(t_unmask)
        # print("UNmasking  ", unique_name)
        # print(type(auto_infer))
        # print('Input unmask tensor')
        # print(t_unmask)
        # print('NEW tensors that need to be unmasked')
        # print(debugnames)
        # print(unmasks)
        # print('!!!!!!!!')
        for dname, _unmask in zip(debugnames, unmasks):
            # print(dname, _unmask)
            self.unmask_chain(dname, _unmask)

    def resolve_conflicts(self):
        """
        Resolve the channel and
        """
        self.resolve_channel_conflicts()
        # self.resolve_group_conflict()

    def resolve_channel_conflicts(self):
        """
        Resolve the shape/mask conflict for the model. Some operators may have shape constraints.
        For example, `add`, the add operation need the input tensors have exactly the same shape.
        If the two input tensors of the add opeartor mask difference values/channels, we need to
        unmask some values/channels and padde zeros to make the shapes of the input tensors are the
        same.
        """

        # build the out_degree table for the nodes in the model
        out_degree = {}
        visit_queue = queue.Queue()
        for node in self.torch_graph.nodes_py.nodes_op:
            successors = self.torch_graph.find_successors(node.unique_name)
            out_degree[node.unique_name] = len(successors)
            if out_degree[node.unique_name] == 0:
                # if this node doesn't have any successor nodes
                visit_queue.put(node)
        # backward traverse the model graph and find the operators that have shape
        # dependencies
        while not visit_queue.empty():
            cur_node = visit_queue.get()
            _auto_infer = self.auto_inferences[cur_node.unique_name]
            _logger.debug('Resolve conflict for %s', cur_node.unique_name)
            unmask = self.calc_reindex(cur_node)
            if unmask is not None:
                for i, tensor in enumerate(unmask):
                    if tensor is not None:
                        # The reason why we use the input_debugname in the _auto_infer
                        # rather the cur_node.inputs is that the cur_node.inputs have
                        # the parameters of the modules (weight, bias), and this is caused by
                        # the merging rules when we build the TorchModuleGraph. In TorchModuleGraph
                        # we merge the node based on its scope name and the 'prim::GetAttr' node of
                        # weight tensor has no scope name.
                        debugname = _auto_infer.input_debugname[i]
                        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                        # print('Find conflict at ', cur_node.name)
                        self.unmask_chain(debugname, tensor)
            predecessors = self.torch_graph.find_predecessors(
                cur_node.unique_name)
            for predecessor in predecessors:
                out_degree[predecessor] -= 1
                if out_degree[predecessor] == 0:
                    visit_queue.put(self.torch_graph.name_to_node[predecessor])

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
        # TODO suppose to fix the conflict after the sparsity inference
        if not self.enable_compile:
            # if we cannot modify the network sparsity, then we should resolve
            # the sparsity conflict by unmask some sparse values.
            fix_mask_conflict(self.masks, self.bound_model, self.dummy_input)
            # exit(-1)
        _logger.info("infer module masks...")
        self.infer_modules_masks()
        _logger.info('resolve the mask conflict')
        # self.resolve_conflicts()
        # load the original stat dict before replace the model
        self.bound_model.load_state_dict(self.ori_state_dict)
        _logger.info("replace compressed modules...")
        if not self.enable_compile:
            # the mask conflict should be already resolved
            self.replace_compressed_modules()
        else:
            self.compile_sparse_modules()
        self.bound_model.train(training)
        _logger.info("speedup done")
