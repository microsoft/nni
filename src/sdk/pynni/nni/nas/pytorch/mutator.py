# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import defaultdict

import numpy as np
import torch

from nni.nas.pytorch.base_mutator import BaseMutator

logger = logging.getLogger(__name__)


class Mutator(BaseMutator):

    def __init__(self, model):
        super().__init__(model)
        self._cache = dict()
        self._connect_all = False

    def sample_search(self):
        """
        Override to implement this method to iterate over mutables and make decisions.

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        raise NotImplementedError

    def sample_final(self):
        """
        Override to implement this method to iterate over mutables and make decisions that is final
        for export and retraining.

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the mutator by call the `sample_search` to resample (for search). Stores the result in a local
        variable so that `on_forward_layer_choice` and `on_forward_input_choice` can use the decision directly.
        """
        self._cache = self.sample_search()

    def export(self):
        """
        Resample (for final) and return results.

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        return self.sample_final()

    def status(self):
        """
        Return current selection status of mutator.

        Returns
        -------
        dict
            A mapping from key of mutables to decisions. All weights (boolean type and float type)
            are converted into real number values. Numpy arrays and tensors are converted into list.
        """
        data = dict()
        for k, v in self._cache.items():
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray):
                v = v.astype(np.float32).tolist()
            data[k] = v
        return data

    def graph(self, inputs):
        """
        Return model supernet graph.

        Parameters
        ----------
        inputs: tuple of tensor
            Inputs that will be feeded into the network.

        Returns
        -------
        dict
            Containing ``node``, in Tensorboard GraphDef format.
            Additional key ``mutable`` is a map from key to list of modules.
        """
        if not torch.__version__.startswith("1.4"):
            logger.warning("Graph is only tested with PyTorch 1.4. Other versions might not work.")
        from ._graph_utils import graph
        from google.protobuf import json_format
        # protobuf should be installed as long as tensorboard is installed
        try:
            self._connect_all = True
            graph_def, _ = graph(self.model, inputs, verbose=False)
            result = json_format.MessageToDict(graph_def)
        finally:
            self._connect_all = False

        # `mutable` is to map the keys to a list of corresponding modules.
        # A key can be linked to multiple modules, use `dedup=False` to find them all.
        result["mutable"] = defaultdict(list)
        for mutable in self.mutables.traverse(deduplicate=False):
            # A module will be represent in the format of
            # [{"type": "Net", "name": ""}, {"type": "Cell", "name": "cell1"}, {"type": "Conv2d": "name": "conv"}]
            # which will be concatenated into Net/Cell[cell1]/Conv2d[conv] in frontend.
            # This format is aligned with the scope name jit gives.
            modules = mutable.name.split(".")
            path = [
                {"type": self.model.__class__.__name__, "name": ""}
            ]
            m = self.model
            for module in modules:
                m = getattr(m, module)
                path.append({
                    "type": m.__class__.__name__,
                    "name": module
                })
            result["mutable"][mutable.key].append(path)
        return result

    def on_forward_layer_choice(self, mutable, *inputs):
        """
        On default, this method retrieves the decision obtained previously, and select certain operations.
        Only operations with non-zero weight will be executed. The results will be added to a list.
        Then it will reduce the list of all tensor outputs with the policy specified in `mutable.reduction`.

        Parameters
        ----------
        mutable : LayerChoice
            Layer choice module.
        inputs : list of torch.Tensor
            Inputs

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            Output and mask.
        """
        if self._connect_all:
            return self._all_connect_tensor_reduction(mutable.reduction,
                                                      [op(*inputs) for op in mutable.choices]), \
                torch.ones(mutable.length)

        def _map_fn(op, *inputs):
            return op(*inputs)

        mask = self._get_decision(mutable)
        assert len(mask) == len(mutable.choices), \
            "Invalid mask, expected {} to be of length {}.".format(mask, len(mutable.choices))
        out = self._select_with_mask(_map_fn, [(choice, *inputs) for choice in mutable.choices], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_forward_input_choice(self, mutable, tensor_list):
        """
        On default, this method retrieves the decision obtained previously, and select certain tensors.
        Then it will reduce the list of all tensor outputs with the policy specified in `mutable.reduction`.

        Parameters
        ----------
        mutable : InputChoice
            Input choice module.
        tensor_list : list of torch.Tensor
            Tensor list to apply the decision on.

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            Output and mask.
        """
        if self._connect_all:
            return self._all_connect_tensor_reduction(mutable.reduction, tensor_list), \
                torch.ones(mutable.n_candidates)
        mask = self._get_decision(mutable)
        assert len(mask) == mutable.n_candidates, \
            "Invalid mask, expected {} to be of length {}.".format(mask, mutable.n_candidates)
        out = self._select_with_mask(lambda x: x, [(t,) for t in tensor_list], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def _select_with_mask(self, map_fn, candidates, mask):
        if "BoolTensor" in mask.type():
            out = [map_fn(*cand) for cand, m in zip(candidates, mask) if m]
        elif "FloatTensor" in mask.type():
            out = [map_fn(*cand) * m for cand, m in zip(candidates, mask) if m]
        else:
            raise ValueError("Unrecognized mask")
        return out

    def _tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == "none":
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

    def _all_connect_tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == "none":
            return tensor_list
        if reduction_type == "concat":
            return torch.cat(tensor_list, dim=1)
        return torch.stack(tensor_list).sum(0)

    def _get_decision(self, mutable):
        """
        By default, this method checks whether `mutable.key` is already in the decision cache,
        and returns the result without double-check.

        Parameters
        ----------
        mutable : Mutable

        Returns
        -------
        object
        """
        if mutable.key not in self._cache:
            raise ValueError("\"{}\" not found in decision cache.".format(mutable.key))
        result = self._cache[mutable.key]
        logger.debug("Decision %s: %s", mutable.key, result)
        return result
