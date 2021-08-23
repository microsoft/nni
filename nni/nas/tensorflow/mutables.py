# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import OrderedDict

from tensorflow.keras import Model

from .utils import global_mutable_counting


_logger = logging.getLogger(__name__)


class Mutable(Model):
    def __init__(self, key=None):
        super().__init__()
        if key is None:
            self._key = '{}_{}'.format(type(self).__name__, global_mutable_counting())
        elif isinstance(key, str):
            self._key = key
        else:
            self._key = str(key)
            _logger.warning('Key "%s" is not string, converted to string.', key)
        self.init_hook = None
        self.forward_hook = None

    def __deepcopy__(self, memodict=None):
        raise NotImplementedError("Deep copy doesn't work for mutables.")

    def set_mutator(self, mutator):
        if hasattr(self, 'mutator'):
            raise RuntimeError('`set_mutator is called more than once. '
                               'Did you parse the search space multiple times? '
                               'Or did you apply multiple fixed architectures?')
        self.mutator = mutator

    def call(self, *inputs):
        raise NotImplementedError('Method `call` of Mutable must be overridden')

    def build(self, input_shape):
        self._check_built()

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._name if hasattr(self, '_name') else self._key

    @name.setter
    def name(self, name):
        self._name = name

    def _check_built(self):
        if not hasattr(self, 'mutator'):
            raise ValueError(
                "Mutator not set for {}. You might have forgotten to initialize and apply your mutator. "
                "Or did you initialize a mutable on the fly in forward pass? Move to `__init__` "
                "so that trainer can locate all your mutables. See NNI docs for more details.".format(self))

    def __repr__(self):
        return '{} ({})'.format(self.name, self.key)


class MutableScope(Mutable):
    def __call__(self, *args, **kwargs):
        try:
            self.mutator.enter_mutable_scope(self)
            return super().__call__(*args, **kwargs)
        finally:
            self.mutator.exit_mutable_scope(self)


class LayerChoice(Mutable):
    def __init__(self, op_candidates, reduction='sum', return_mask=False, key=None):
        super().__init__(key=key)
        self.names = []
        if isinstance(op_candidates, OrderedDict):
            for name in op_candidates:
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                self.names.append(name)
        elif isinstance(op_candidates, list):
            for i, _ in enumerate(op_candidates):
                self.names.append(str(i))
        else:
            raise TypeError("Unsupported op_candidates type: {}".format(type(op_candidates)))

        self.length = len(op_candidates)
        self.choices = op_candidates
        self.reduction = reduction
        self.return_mask = return_mask

    def call(self, *inputs):
        out, mask = self.mutator.on_forward_layer_choice(self, *inputs)
        if self.return_mask:
            return out, mask
        return out

    def build(self, input_shape):
        self._check_built()
        for op in self.choices:
            op.build(input_shape)

    def __len__(self):
        return len(self.choices)


class InputChoice(Mutable):
    NO_KEY = ''

    def __init__(self, n_candidates=None, choose_from=None, n_chosen=None, reduction='sum', return_mask=False, key=None):
        super().__init__(key=key)
        assert n_candidates is not None or choose_from is not None, \
                'At least one of `n_candidates` and `choose_from` must be not None.'
        if choose_from is not None and n_candidates is None:
            n_candidates = len(choose_from)
        elif choose_from is None and n_candidates is not None:
            choose_from = [self.NO_KEY] * n_candidates
        assert n_candidates == len(choose_from), 'Number of candidates must be equal to the length of `choose_from`.'
        assert n_candidates > 0, 'Number of candidates must be greater than 0.'
        assert n_chosen is None or 0 <= n_chosen <= n_candidates, \
                'Expected selected number must be None or no more than number of candidates.'

        self.n_candidates = n_candidates
        self.choose_from = choose_from.copy()
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.return_mask = return_mask

    def call(self, optional_inputs):
        optional_input_list = optional_inputs
        if isinstance(optional_inputs, dict):
            optional_input_list = [optional_inputs[tag] for tag in self.choose_from]
        assert isinstance(optional_input_list, list), \
                'Optional input list must be a list, not a {}.'.format(type(optional_input_list))
        assert len(optional_inputs) == self.n_candidates, \
                'Length of the input list must be equal to number of candidates.'
        out, mask = self.mutator.on_forward_input_choice(self, optional_input_list)
        if self.return_mask:
            return out, mask
        return out
