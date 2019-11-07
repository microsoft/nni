import logging
from contextlib import contextmanager

import torch
import torch.nn as nn

from nni.nas.pytorch.mutables import PyTorchMutable
from nni.nas.utils import to_snake_case

logger = logging.getLogger(__name__)


class PyTorchMutator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.before_build(model)
        self.parse_search_space(model)
        self.after_build(model)

    def before_build(self, model):
        pass

    def after_build(self, model):
        pass

    def named_mutables(self, model):
        # if distinct is true, the method will filter out those with duplicated keys
        key2module = dict()
        for name, module in model.named_modules():
            if isinstance(module, PyTorchMutable):
                distinct = False
                if module.key in key2module:
                    assert key2module[module.key].similar(module), \
                        "Mutable \"{}\" that share the same key must be similar to each other".format(module.key)
                else:
                    distinct = True
                    key2module[module.key] = module
                yield name, module, distinct

    def __setattr__(self, key, value):
        if key in ["model", "net", "network"]:
            logger.warning("Think twice if you are including the network into mutator.")
        return super().__setattr__(key, value)

    def parse_search_space(self, model):
        for name, mutable, distinct in self.named_mutables(model):
            mutable.name = name
            mutable.set_mutator(self)
            if not distinct:
                continue
            init_method_name = "on_init_{}".format(to_snake_case(mutable.__class__.__name__))
            if hasattr(self, init_method_name) and callable(getattr(self, init_method_name)):
                getattr(self, init_method_name)(mutable)
            else:
                # fallback to general init
                self.on_init_general(mutable)

    def on_init_general(self, mutable):
        pass

    @contextmanager
    def forward_pass(self):
        self.before_pass()
        try:
            yield self
        finally:
            self.after_pass()

    def before_pass(self):
        self._in_forward_pass = True
        self._cache = dict()

    def after_pass(self):
        self._in_forward_pass = False

    def enter_mutable_scope(self, mutable_scope):
        pass

    def exit_mutable_scope(self, mutable_scope):
        pass

    def on_finish_mutable_cell(self):
        pass

    def forward(self, *inputs):
        raise NotImplementedError("Mutator is not forward-able")

    def on_forward(self, mutable, *inputs):
        """Callback on forwarding a mutable"""
        if not hasattr(self, "_in_forward_pass") or not self._in_forward_pass:
            raise ValueError("Not in forward pass. Did you forget to call mutator.forward_pass(), or forget to call "
                             "super().before_pass() and after_pass() in your override method?")
        forward_method_name = "on_forward_{}".format(to_snake_case(mutable.__class__.__name__))
        if hasattr(self, forward_method_name) and callable(getattr(self, forward_method_name)):
            return getattr(self, forward_method_name)(mutable, *inputs)
        else:
            # fallback to general forward
            return self.on_forward_general(mutable, *inputs)

    def on_forward_general(self, mutable, *inputs):
        raise NotImplementedError("Forward has to be implemented")

    def on_forward_layer_choice(self, mutable, *inputs):
        def _map_fn(op, *inputs):
            return op(*inputs)
        mask = self._cache.setdefault(mutable.key, self.on_calc_layer_choice_mask(mutable))
        out = self._select_with_mask(_map_fn, [(choice, *inputs) for choice in mutable.choices], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_forward_input_choice(self, mutable, tensor_list, semantic_labels):
        mask = self._cache.setdefault(mutable.key, self.on_calc_input_choice_mask(mutable, semantic_labels))
        out = self._select_with_mask(lambda x: x, [(t, ) for t in tensor_list], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_calc_layer_choice_mask(self, mutable):
        """
        Recommend to override mask method. Only override on_forward method when you are confident enough.

        Parameters
        ----------
        mutable

        Returns
        -------
        torch.Tensor
        """
        raise NotImplementedError("Layer choice mask calculation must be implemented")

    def on_calc_input_choice_mask(self, mutable, semantic_labels):
        """

        Parameters
        ----------
        mutable
        semantic_labels: list of string
            The name of labels of input tensors given by user.

        Returns
        -------

        """
        raise NotImplementedError("Input choice mask calculation must be implemented")

    def _select_with_mask(self, map_fn, candidates, mask):
        if isinstance(mask, torch.BoolTensor):
            # print(candidates[0], len(mask))
            out = [map_fn(*cand) for cand, m in zip(candidates, mask) if m]
        elif isinstance(mask, torch.FloatTensor):
            out = [map_fn(*cand) * m for cand, m in zip(candidates, mask)]
        else:
            raise ValueError("Unrecognized mask")
        return out

    def _tensor_reduction(self, reduction_type, tensor_list):
        if tensor_list == "none":
            return tensor_list
        if not tensor_list:
            return None  # empty. return None for now
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == "sum":
            return sum(tensor_list)
        if reduction_type == "mean":
            return sum(tensor_list) / len(tensor_list)
        if reduction_type == "concat":
            return torch.cat(tensor_list, dim=1)
        raise ValueError("Unrecognized reduction policy: \"{}\"".format(reduction_type))
