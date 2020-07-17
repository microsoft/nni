# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import tensorflow as tf

from .base_mutator import BaseMutator
from . import graph_utils
from .mutable import InputChoice, LayerChoice


_logger = logging.getLogger(__name__)


class Mutator(BaseMutator):
    def __init__(self, model):
        super().__init__(model)
        self._cache = {}

    def sample_search(self):
        raise NotImplementedError('Method `sample_search` must be overridden')

    def sample_final(self):
        raise NotImplementedError('Method `sample_final` must be overriden for exporting')

    def reset(self):
        self._cache = self.sample_search()

        # TensorFlow cannot detect unused `Mutable` branches.
        # We need to fool the graph tracer for reasonable performance.
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice) and mutable._built:
                mutable._backup_call = mutable.__call__
                mask = self._get_decision(mutable)
                active = [(op, op_mask) for op, op_mask in zip(mutable.choices, mask) if op_mask]

                def _fixed_call(*inputs):
                    out = self._select_with_mask(lambda op: op(*inputs), active[0], active[1])
                    return (out, mask) if mutable.return_mask else mask

                # TODO: only retracing when the connection changes
                mutable.__call__ = graph_utils.function(_fixed_call)

            elif isinstance(mutable, InputChoice):
                # Since ENAS always calculate all candidate inputs, leave this unchange for now
                pass



    def export(self):
        return self.sample_final()

    # TODO: status
    # TODO: graph

    def on_forward_layer_choice(self, mutable, *inputs):
        mask = self._get_decision(mutable)
        assert len(mask) == len(mutable), \
                'Invalid mask, expected {} to be of length {}.'.format(mask, len(mutable))
        out = self._select_with_mask(lambda choice: choice(*inputs), mutable.choices, mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_forward_input_choice(self, mutable, tensor_list):
        mask = self._get_decision(mutable)
        assert len(mask) == mutable.n_candidates, \
                'Invalid mask, expected {} to be of length {}.'.format(mask, mutable.n_candidates)
        out = self._select_with_mask(lambda tensor: tensor, tensor_list, mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def _select_with_mask(self, map_fn, candidates, mask):
        if mask.dtype.is_bool:
            out = [map_fn(cand) for cand, m in zip(candidates, mask) if m]
        elif mask.dtype.is_floating:
            out = [map_fn(cand) * m for cand, m in zip(candidates, mask) if m]
        else:
            raise ValueError('Unrecognized mask, dtype is {}'.format(mask.dtype.name))
        return out

    def _tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == 'none':
            return tensor_list
        if not tensor_list:
            return None
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == 'sum':
            return sum(tensor_list)
        if reduction_type == 'mean':
            return sum(tensor_list) / len(tensor_list)
        if reduction_type == 'concat':
            return tf.concat(tensor_list, axis=0)
        raise ValueError('Unrecognized reduction policy: "{}'.format(reduction_type))

    def _get_decision(self, mutable):
        if mutable.key not in self._cache:
            raise ValueError('"{}" not found in decision cache.'.format(mutable.key))
        result = self._cache[mutable.key]
        _logger.debug('Decision %s: %s', mutable.key, result)
        return result
