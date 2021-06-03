# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import warnings
from collections import defaultdict
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
                if main_file_path.parents[0] != Path('.'):
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


class ContextStack:
    """
    This is to maintain a globally-accessible context envinronment that is visible to everywhere.

    Use ``with ContextStack(namespace, value):`` to initiate, and use ``get_current_context(namespace)`` to
    get the corresponding value in the namespace.
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
        assert cls._stack[key], 'Context is empty.'
        return cls._stack[key][-1]


def get_current_context(key: str) -> Any:
    return ContextStack.top(key)
