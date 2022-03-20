# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import itertools
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, List, Dict
from pathlib import Path


def import_(target: str, allow_none: bool = False) -> Any:
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)


def version_larger_equal(a: str, b: str) -> bool:
    # TODO: refactor later
    a = a.split('+')[0]
    b = b.split('+')[0]
    return tuple(map(int, a.split('.'))) >= tuple(map(int, b.split('.')))


_last_uid = defaultdict(int)

_DEFAULT_MODEL_NAMESPACE = 'model'


def uid(namespace: str = 'default') -> int:
    _last_uid[namespace] += 1
    return _last_uid[namespace]


def reset_uid(namespace: str = 'default') -> None:
    _last_uid[namespace] = 0


def get_module_name(cls_or_func):
    module_name = cls_or_func.__module__
    if module_name == '__main__':
        # infer the module name with inspect
        for frm in inspect.stack():
            if inspect.getmodule(frm[0]).__name__ == '__main__':
                # main module found
                main_file_path = Path(inspect.getsourcefile(frm[0]))
                if not Path().samefile(main_file_path.parent):
                    raise RuntimeError(f'You are using "{main_file_path}" to launch your experiment, '
                                       f'please launch the experiment under the directory where "{main_file_path.name}" is located.')
                module_name = main_file_path.stem
                break
    if module_name == '__main__':
        warnings.warn('Callstack exhausted but main module still not found. This will probably cause issues that the '
                      'function/class cannot be imported.')

    # NOTE: this is hacky. As torchscript retrieves LSTM's source code to do something.
    # to make LSTM's source code can be found, we should assign original LSTM's __module__ to
    # the wrapped LSTM's __module__
    # TODO: find out all the modules that have the same requirement as LSTM
    if f'{cls_or_func.__module__}.{cls_or_func.__name__}' == 'torch.nn.modules.rnn.LSTM':
        module_name = cls_or_func.__module__

    return module_name


def get_importable_name(cls, relocate_module=False):
    module_name = get_module_name(cls) if relocate_module else cls.__module__
    return module_name + '.' + cls.__name__


class NoContextError(Exception):
    pass


class ContextStack:
    """
    This is to maintain a globally-accessible context envinronment that is visible to everywhere.

    Use ``with ContextStack(namespace, value):`` to initiate, and use ``get_current_context(namespace)`` to
    get the corresponding value in the namespace.

    Note that this is not multi-processing safe. Also, the values will get cleared for a new process.
    """

    _stack: Dict[str, List[Any]] = defaultdict(list)

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    def __enter__(self):
        self.push(self.key, self.value)
        return self

    def __exit__(self, *args, **kwargs):
        self.pop(self.key)

    @classmethod
    def push(cls, key: str, value: Any):
        cls._stack[key].append(value)

    @classmethod
    def pop(cls, key: str) -> None:
        cls._stack[key].pop()

    @classmethod
    def top(cls, key: str) -> Any:
        if not cls._stack[key]:
            raise NoContextError('Context is empty.')
        return cls._stack[key][-1]


class ModelNamespace:
    """
    To create an individual namespace for models to enable automatic numbering.
    """

    def __init__(self, key: str = _DEFAULT_MODEL_NAMESPACE):
        # for example, key: "model_wrapper"
        self.key = key

    def __enter__(self):
        # For example, currently the top of stack is [1, 2, 2], and [1, 2, 2, 3] is used,
        # the next thing up is [1, 2, 2, 4].
        # `reset_uid` to count from zero for "model_wrapper_1_2_2_4"
        try:
            current_context = ContextStack.top(self.key)
            next_uid = uid(self._simple_name(self.key, current_context))
            ContextStack.push(self.key, current_context + [next_uid])
            reset_uid(self._simple_name(self.key, current_context + [next_uid]))
        except NoContextError:
            ContextStack.push(self.key, [])
            reset_uid(self._simple_name(self.key, []))

    def __exit__(self, *args, **kwargs):
        ContextStack.pop(self.key)

    @staticmethod
    def next_label(key: str = _DEFAULT_MODEL_NAMESPACE) -> str:
        try:
            current_context = ContextStack.top(key)
        except NoContextError:
            # fallback to use "default" namespace
            return ModelNamespace._simple_name('default', [uid()])

        next_uid = uid(ModelNamespace._simple_name(key, current_context))
        return ModelNamespace._simple_name(key, current_context + [next_uid])

    @staticmethod
    def _simple_name(key: str, lst: List[Any]) -> str:
        return key + ''.join(['_' + str(k) for k in lst])


def get_current_context(key: str) -> Any:
    return ContextStack.top(key)


# map variables to prefix in the state dict
# e.g., {'upsample': 'mynet.module.deconv2.upsample_layer'}
STATE_DICT_PY_MAPPING = '_mapping_'

# map variables to `prefix`.`value` in the state dict
# e.g., {'upsample': 'choice3.upsample_layer'},
# which actually means {'upsample': 'mynet.module.choice3.upsample_layer'},
# and 'upsample' is also in `mynet.module`.
STATE_DICT_PY_MAPPING_PARTIAL = '_mapping_partial_'


@contextmanager
def original_state_dict_hooks(model: Any):
    """
    Use this patch if you want to save/load state dict in the original state dict hierarchy.

    For example, when you already have a state dict for the base model / search space (which often
    happens when you have trained a supernet with one-shot strategies), the state dict isn't organized
    in the same way as when a sub-model is sampled from the search space. This patch will help
    the modules in the sub-model find the corresponding module in the base model.

    The code looks like,

    .. code-block:: python

        with original_state_dict_hooks(model):
            model.load_state_dict(state_dict_from_supernet, strict=False)  # supernet has extra keys

    Or vice-versa,

    .. code-block:: python

        with original_state_dict_hooks(model):
            supernet_style_state_dict = model.state_dict()
    """

    import torch.nn as nn
    assert isinstance(model, nn.Module), 'PyTorch is the only supported framework for now.'

    # the following are written for pytorch only

    # first get the full mapping
    full_mapping = {}

    def full_mapping_in_module(src_prefix, tar_prefix, module):
        if hasattr(module, STATE_DICT_PY_MAPPING):
            # only values are complete
            local_map = getattr(module, STATE_DICT_PY_MAPPING)
        elif hasattr(module, STATE_DICT_PY_MAPPING_PARTIAL):
            # keys and values are both incomplete
            local_map = getattr(module, STATE_DICT_PY_MAPPING_PARTIAL)
            local_map = {k: tar_prefix + v for k, v in local_map.items()}
        else:
            # no mapping
            local_map = {}

        if '__self__' in local_map:
            # special case, overwrite prefix
            tar_prefix = local_map['__self__'] + '.'

        for key, value in local_map.items():
            if key != '' and key not in module._modules:  # not a sub-module, probably a parameter
                full_mapping[src_prefix + key] = value

        if src_prefix != tar_prefix:  # To deal with leaf nodes.
            for name, value in itertools.chain(module._parameters.items(), module._buffers.items()):  # direct children
                if value is None or name in module._non_persistent_buffers_set:
                    # it won't appear in state dict
                    continue
                if (src_prefix + name) not in full_mapping:
                    full_mapping[src_prefix + name] = tar_prefix + name

        for name, child in module.named_children():
            # sub-modules
            full_mapping_in_module(
                src_prefix + name + '.',
                local_map.get(name, tar_prefix + name) + '.',  # if mapping doesn't exist, respect the prefix
                child
            )

    full_mapping_in_module('', '', model)

    def load_state_dict_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        reverse_mapping = defaultdict(list)
        for src, tar in full_mapping.items():
            reverse_mapping[tar].append(src)

        transf_state_dict = {}
        for src, tar_keys in reverse_mapping.items():
            if src in state_dict:
                value = state_dict.pop(src)
                for tar in tar_keys:
                    transf_state_dict[tar] = value
            else:
                missing_keys.append(src)
        state_dict.update(transf_state_dict)

    def state_dict_hook(module, destination, prefix, local_metadata):
        result = {}
        for src, tar in full_mapping.items():
            if src in destination:
                result[tar] = destination.pop(src)
            else:
                raise KeyError(f'"{src}" not in state dict, but found in mapping.')
        destination.update(result)

    try:
        hooks = []
        hooks.append(model._register_load_state_dict_pre_hook(load_state_dict_hook))
        hooks.append(model._register_state_dict_hook(state_dict_hook))
        yield
    finally:
        for hook in hooks:
            hook.remove()
