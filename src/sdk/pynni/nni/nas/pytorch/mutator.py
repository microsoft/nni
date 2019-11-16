from contextlib import contextmanager

import torch
import torch.nn as nn

from nni.nas.pytorch.base_mutator import BaseMutator


class Mutator(BaseMutator, nn.Module):

    def export(self):
        if self._in_forward_pass:
            raise RuntimeError("Still in forward pass. Exporting might induce incompleteness.")
        if not self._cache:
            raise RuntimeError("No running history found. You need to call your model at least once before exporting. "
                               "You might also want to check if there are no valid mutables in your model.")
        return self._cache

    @contextmanager
    def forward_pass(self):
        self._in_forward_pass = True
        self._cache = dict()
        self.before_pass()
        try:
            yield self
        finally:
            self.after_pass()
            self._in_forward_pass = False

    def before_pass(self):
        pass

    def after_pass(self):
        pass

    def _check_in_forward_pass(self):
        if not hasattr(self, "_in_forward_pass") or not self._in_forward_pass:
            raise ValueError("Not in forward pass. Did you forget to call mutator.forward_pass(), or forget to call "
                             "super().before_pass() and after_pass() in your override method?")

    def on_forward_layer_choice(self, mutable, *inputs):
        """
        Callback of layer choice forward. Override if you are an advanced user.
        On default, this method calls :meth:`on_calc_layer_choice_mask` to get a mask on how to choose between layers
        (either by switch or by weights), then it will reduce the list of all tensor outputs with the policy specified
        in `mutable.reduction`. It will also cache the mask with corresponding `mutable.key`.

        Parameters
        ----------
        mutable: LayerChoice
        inputs: list of torch.Tensor

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
        """
        self._check_in_forward_pass()

        def _map_fn(op, *inputs):
            return op(*inputs)

        mask = self._cache.setdefault(mutable.key, self.on_calc_layer_choice_mask(mutable))
        out = self._select_with_mask(_map_fn, [(choice, *inputs) for choice in mutable.choices], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_forward_input_choice(self, mutable, tensor_list, tags):
        """
        Callback of input choice forward. Override if you are an advanced user.
        On default, this method calls :meth:`on_calc_input_choice_mask` with `tags`
        to get a mask on how to choose between inputs (either by switch or by weights), then it will reduce
        the list of all tensor outputs with the policy specified in `mutable.reduction`. It will also cache the
        mask with corresponding `mutable.key`.

        Parameters
        ----------
        mutable: InputChoice
        tensor_list: list of torch.Tensor
        tags: list of string

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
        """
        self._check_in_forward_pass()
        mask = self._cache.setdefault(mutable.key, self.on_calc_input_choice_mask(mutable, tags))
        out = self._select_with_mask(lambda x: x, [(t,) for t in tensor_list], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_calc_layer_choice_mask(self, mutable):
        """
        Recommended to override. Calculate a mask tensor for a layer choice.

        Parameters
        ----------
        mutable: LayerChoice
            Corresponding layer choice object.

        Returns
        -------
        torch.Tensor
            Should be a 1D tensor, either float or bool. If float, the numbers are treated as weights. If bool,
            the numbers are treated as switch.
        """
        raise NotImplementedError("Layer choice mask calculation must be implemented")

    def on_calc_input_choice_mask(self, mutable, tags):
        """
        Recommended to override. Calculate a mask tensor for a input choice.

        Parameters
        ----------
        mutable: InputChoice
            Corresponding input choice object.
        tags: list of string
            The name of labels of input tensors given by user. Usually it's a
            :class:`~nni.nas.pytorch.mutables.MutableScope` marked by user.

        Returns
        -------
        torch.Tensor
            Should be a 1D tensor, either float or bool. If float, the numbers are treated as weights. If bool,
            the numbers are treated as switch.
        """
        raise NotImplementedError("Input choice mask calculation must be implemented")

    def _select_with_mask(self, map_fn, candidates, mask):
        if "BoolTensor" in mask.type():
            out = [map_fn(*cand) for cand, m in zip(candidates, mask) if m]
        elif "FloatTensor" in mask.type():
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
