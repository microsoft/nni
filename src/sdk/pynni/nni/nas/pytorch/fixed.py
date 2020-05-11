# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from .mutables import InputChoice, LayerChoice, MutableScope
from .mutator import Mutator
from .utils import to_list


class FixedArchitecture(Mutator):
    """
    Fixed architecture mutator that always selects a certain graph.

    Parameters
    ----------
    model : nn.Module
        A mutable network.
    fixed_arc : dict
        Preloaded architecture object.
    strict : bool
        Force everything that appears in ``fixed_arc`` to be used at least once.
    """

    def __init__(self, model, fixed_arc, strict=True):
        super().__init__(model)
        self._fixed_arc = fixed_arc

        mutable_keys = set([mutable.key for mutable in self.mutables if not isinstance(mutable, MutableScope)])
        fixed_arc_keys = set(self._fixed_arc.keys())
        if fixed_arc_keys - mutable_keys:
            raise RuntimeError("Unexpected keys found in fixed architecture: {}.".format(fixed_arc_keys - mutable_keys))
        if mutable_keys - fixed_arc_keys:
            raise RuntimeError("Missing keys in fixed architecture: {}.".format(mutable_keys - fixed_arc_keys))
        self._fixed_arc = self._from_human_readable_architecture(self._fixed_arc)

    def _from_human_readable_architecture(self, human_arc):
        # convert from an exported architecture
        result_arc = {k: to_list(v) for k, v in human_arc.items()}  # there could be tensors, numpy arrays, etc.
        # First, convert non-list to list, because there could be {"op1": 0} or {"op1": "conv"},
        # which means {"op1": [0, ]} ir {"op1": ["conv", ]}
        result_arc = {k: v if isinstance(v, list) else [v] for k, v in result_arc.items()}
        # Second, infer which ones are multi-hot arrays and which ones are in human-readable format.
        # This is non-trivial, since if an array in [0, 1], we cannot know for sure it means [false, true] or [true, true].
        # Here, we assume an multihot array has to be a boolean array or a float array and matches the length.
        for mutable in self.mutables:
            if mutable.key not in result_arc:
                continue  # skip silently
            choice_arr = result_arc[mutable.key]
            if all(isinstance(v, bool) for v in choice_arr) or all(isinstance(v, float) for v in choice_arr):
                if (isinstance(mutable, LayerChoice) and len(mutable) == len(choice_arr)) or \
                        (isinstance(mutable, InputChoice) and mutable.n_candidates == len(choice_arr)):
                    # multihot, do nothing
                    continue
            if isinstance(mutable, LayerChoice):
                choice_arr = [mutable.names.index(val) if isinstance(val, str) else val for val in choice_arr]
                choice_arr = [i in choice_arr for i in range(len(mutable))]
            elif isinstance(mutable, InputChoice):
                choice_arr = [mutable.choose_from.index(val) if isinstance(val, str) else val for val in choice_arr]
                choice_arr = [i in choice_arr for i in range(mutable.n_candidates)]
            result_arc[mutable.key] = choice_arr
        return result_arc

    def sample_search(self):
        """
        Always returns the fixed architecture.
        """
        return self._fixed_arc

    def sample_final(self):
        """
        Always returns the fixed architecture.
        """
        return self._fixed_arc


def apply_fixed_architecture(model, fixed_arc):
    """
    Load architecture from `fixed_arc` and apply to model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with mutables.
    fixed_arc : str or dict
        Path to the JSON that stores the architecture, or dict that stores the exported architecture.

    Returns
    -------
    FixedArchitecture
        Mutator that is responsible for fixes the graph.
    """

    if isinstance(fixed_arc, str):
        with open(fixed_arc) as f:
            fixed_arc = json.load(f)
    architecture = FixedArchitecture(model, fixed_arc)
    architecture.reset()
    return architecture
