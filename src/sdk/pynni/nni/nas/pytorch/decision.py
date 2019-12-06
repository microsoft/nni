import logging

import numpy as np
import torch

_logger = logging.getLogger(__name__)


class Decision:
    def __init__(self):
        raise NotImplementedError("You should never use init to initialize a general decision.")

    @classmethod
    def from_nni_protocol_format(cls, candidate, search_space=None):
        assert "_idx" in candidate and "_val" in candidate, "A candidate must have '_idx' and '_val' in its fields."
        assert type(candidate["_idx"]) == type(candidate["_val"]), "Indices and values must have the same type."
        if search_space is not None:
            search_space_values = search_space["_values"]
            if isinstance(candidate["_idx"], list):
                assert len(candidate["_idx"]) == len(candidate["_val"]), \
                    "Number of indices must be equal of number of values."
                for idx, val in zip(candidate["_idx"], candidate["_val"]):
                    assert 0 <= idx < len(search_space_values) and search_space_values[idx] == val, \
                        "Index '{}' in search space '{}' is not '{}'".format(idx, search_space_values, val)
            elif isinstance(candidate["_idx"], int):
                idx, val = candidate["_idx"], candidate["_val"]
                assert 0 <= idx < len(search_space_values) and search_space_values[idx] == val, \
                    "Index '{}' in search space '{}' is not '{}'".format(idx, search_space_values, val)
            else:
                raise ValueError("Index of unrecognized type: {}".format(candidate["_idx"]))
            return cls.from_indices(candidate["_idx"], len(search_space_values))
        return cls.from_indices(candidate["_idx"])

    @classmethod
    def from_indices(cls, indices, n_candidates=None):
        """
        Construct a decision from indices.

        Parameters
        ----------
        indices : int or list of int
        n_candidates : int

        Returns
        -------
        RelaxedDecision
        """
        return RelaxedDecision(indices, n_candidates)

    @classmethod
    def deserialize(cls, obj):
        if obj is None:
            return EmptyDecision()
        if isinstance(obj, dict) and "_idx" in obj:
            return cls.from_nni_protocol_format(obj)
        if isinstance(obj, int):
            return cls.from_indices(obj)
        obj_type = cls._list_type(obj)
        if obj_type == int:
            # list of indices
            return cls.from_indices(obj)
        if obj_type == float:
            # list of weights
            return ContinuousDecision(obj)
        if obj_type == bool:
            # one/multi-hot tensor
            return RelaxedDecision.from_multi_hot_iterable(obj)

    @staticmethod
    def _list_type(lst):
        # get the element type of a list / tensor

        def _print_all_01_warning():
            if all_01:
                _logger.warning("All elements in %s are 0 and 1, but type is not bool.", lst)

        all_01 = all(map(lambda x: x in [0., 1.], lst))
        if torch.is_tensor(lst):
            type_lower = lst.type().lower()
            if "bool" in type_lower:
                return bool
            _print_all_01_warning()
            if "float" in type_lower:
                return float
            raise ValueError("Unsupported tensor type: {}".format(type_lower))
        if all(map(lambda x: isinstance(x, bool), lst)):
            return bool
        _print_all_01_warning()
        for t in (int, float):
            if all(map(lambda x: isinstance(x, t), lst)):
                return t

    def serialize(self):
        raise NotImplementedError


class EmptyDecision(Decision):
    def serialize(self):
        return None


class RelaxedDecision(Decision):
    def __init__(self, indices, n_candidates=None):
        if isinstance(indices, int):
            self.indices = [indices]
        elif isinstance(indices, list):
            self.indices = indices
        assert len(set(self.indices)) == len(self.indices), "Indices must be unique"
        self.n_candidates = n_candidates
        if n_candidates is not None:
            assert all(map(lambda x: 0 <= x < n_candidates, self.indices)), \
                "Indices must be in range [0, n_candidates)."

    @classmethod
    def from_multi_hot_iterable(cls, iterable):
        indices, total = [], 0
        for i, t in enumerate(iterable):
            if t:
                indices.append(i)
            total += 1
        return cls(indices, total)

    def serialize(self):
        if len(self.indices) == 1:
            return self.index
        return self.indices

    @property
    def index(self):
        if len(self.indices) > 1:
            raise ValueError("More than one indices. Index doesn't work.")
        return self.indices[0]


class ContinuousDecision:
    def __init__(self, weights):
        self.weights = weights

    def serialize(self):
        if torch.is_tensor(self.weights):
            return self.weights.detach().numpy().tolist()
        if isinstance(self.weights, np.ndarray):
            return self.weights.tolist()
        return self.weights
