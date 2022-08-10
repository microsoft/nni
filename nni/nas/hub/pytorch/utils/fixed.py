# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This file should be merged to nni/nas/fixed.py"""

from typing import Type

from nni.nas.utils import ContextStack


class FixedFactory:
    """Make a model space ready to create a fixed model.

    Examples
    --------
    >>> factory = FixedFactory(ModelSpaceClass, {"choice1": 3})
    >>> model = factory(channels=16, classes=10)
    """

    # TODO: mutations on ``init_args`` and ``init_kwargs`` themselves are not supported.

    def __init__(self, cls: Type, arch: dict):
        self.cls = cls
        self.arch = arch

    def __call__(self, *init_args, **init_kwargs):
        with ContextStack('fixed', self.arch):
            return self.cls(*init_args, **init_kwargs)

    def __repr__(self):
        return f'FixedFactory(class={self.cls}, arch={self.arch})'
