import torch

from nni.nas.pytorch.base_mutator import BaseMutator


class Mutator(BaseMutator):

    def __init__(self, model, controller):
        super().__init__(model)
        self.controller = controller
        self.controller.build(self._structured_mutables)
        self._cache = dict()

    def reset(self):
        """
        Reset the mutator by call the controller to resample (for search).

        Returns
        -------
        None
        """
        self._cache = self.controller.sample_search(self._structured_mutables)

    def export(self):
        """
        Call the controller to resample (for final) and return the results from the controller.

        Returns
        -------
        dict
        """
        return self.controller.sample_final(self._structured_mutables)

    def get_decision(self, mutable):
        """
        By default, this method checks whether `mutable.key` is already in the decision cache, and returns the result
        without double-check.

        Parameters
        ----------
        mutable: Mutable

        Returns
        -------
        any
        """
        if mutable.key not in self._cache:
            raise ValueError("\"{}\" not found in decision cache.".format(mutable.key))
        return self._cache[mutable.key]

    def on_forward_layer_choice(self, mutable, *inputs):
        """
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

        def _map_fn(op, *inputs):
            return op(*inputs)

        mask = self.get_decision(mutable)
        assert len(mask) == len(mutable.choices)
        out = self._select_with_mask(_map_fn, [(choice, *inputs) for choice in mutable.choices], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_forward_input_choice(self, mutable, tensor_list):
        """
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
        mask = self.get_decision(mutable)
        assert len(mask) == mutable.n_candidates
        out = self._select_with_mask(lambda x: x, [(t,) for t in tensor_list], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

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
