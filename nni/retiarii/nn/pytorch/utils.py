from typing import Any, Optional, Tuple, Union

import torch.nn as nn
from nni.retiarii.utils import NoContextError, ModelNamespace, get_current_context


class Mutable(nn.Module):
    """
    This is just an implementation trick for now.

    In future, this could be the base class for all PyTorch mutables including layer choice, input choice, etc.
    This is not considered as an interface, but rather as a base class consisting of commonly used class/instance methods.
    For API developers, it's not recommended to use ``isinstance(module, Mutable)`` to check for mutable modules either,
    before the design is finalized.
    """

    def __new__(cls, *args, **kwargs):
        if not args and not kwargs:
            # this can be the case of copy/deepcopy
            # attributes are assigned afterwards in __dict__
            return super().__new__(cls)

        try:
            return cls.create_fixed_module(*args, **kwargs)
        except NoContextError:
            return super().__new__(cls)

    @classmethod
    def create_fixed_module(cls, *args, **kwargs) -> Union[nn.Module, Any]:
        """
        Try to create a fixed module from fixed dict.
        If the code is running in a trial, this method would succeed, and a concrete module instead of a mutable will be created.
        Raises no context error if the creation failed.
        """
        raise NotImplementedError


def generate_new_label(label: Optional[str]):
    if label is None:
        return ModelNamespace.next_label()
    return label


def get_fixed_value(label: str) -> Any:
    ret = get_current_context('fixed')
    try:
        return ret[generate_new_label(label)]
    except KeyError:
        raise KeyError(f'Fixed context with {label} not found. Existing values are: {ret}')


def get_fixed_dict(label_prefix: str) -> Tuple[str, Any]:
    ret = get_current_context('fixed')
    try:
        label_prefix = generate_new_label(label_prefix)
        ret = {k: v for k, v in ret.items() if k.startswith(label_prefix + '/')}
        if not ret:
            raise KeyError
        return label_prefix, ret
    except KeyError:
        raise KeyError(f'Fixed context with prefix {label_prefix} not found. Existing values are: {ret}')
