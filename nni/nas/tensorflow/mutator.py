# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import tensorflow as tf

from .base_mutator import BaseMutator


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
            image_data_format = tf.keras.backend.image_data_format()
            if image_data_format == "channels_first":
                axis = 0
            else:
                axis = -1
            return tf.concat(tensor_list, axis=axis)  # pylint: disable=E1120,E1123
            # pylint issue #3613
        raise ValueError('Unrecognized reduction policy: "{}'.format(reduction_type))

    def _get_decision(self, mutable):
        if mutable.key not in self._cache:
            raise ValueError('"{}" not found in decision cache.'.format(mutable.key))
        result = self._cache[mutable.key]
        _logger.debug('Decision %s: %s', mutable.key, result)
        return result
