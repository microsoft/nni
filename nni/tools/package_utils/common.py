# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['AlgoMeta']

from typing import NamedTuple

from nni.typehint import Literal

class AlgoMeta(NamedTuple):
    name: str
    alias: str | None
    class_name: str | None
    accept_class_args: bool
    class_args: dict | None
    validator_class_name: str | None
    algo_type: Literal['tuner', 'assessor']
    is_advisor: bool
    is_builtin: bool
    nni_version: str | None

    @staticmethod
    def load(meta: dict, algo_type: Literal['tuner', 'assessor', 'advisor'] | None = None) -> AlgoMeta:
        if algo_type is None:
            algo_type = meta['algoType']  # type: ignore
        return AlgoMeta(
            name = meta['builtinName'],
            alias = meta.get('alias'),
            class_name = meta['className'],
            accept_class_args = meta.get('acceptClassArgs', True),
            class_args = meta.get('classArgs'),
            validator_class_name = meta.get('classArgsValidator'),
            algo_type = ('assessor' if algo_type == 'assessor' else 'tuner'),
            is_advisor = meta.get('isAdvisor', algo_type == 'advisor'),
            is_builtin = (meta.get('source') == 'nni'),
            nni_version = meta.get('nniVersion')
        )

    def dump(self) -> dict:
        ret = {}
        ret['builtinName'] = self.name
        if self.alias is not None:
            ret['alias'] = self.alias
        ret['className'] = self.class_name
        if not self.accept_class_args:
            ret['acceptClassArgs'] = False
        if self.class_args is not None:
            ret['classArgs'] = self.class_args
        if self.validator_class_name is not None:
            ret['classArgsValidator'] = self.validator_class_name
        if self.is_advisor:
            ret['isAdvisor'] = True
        ret['source'] = 'nni' if self.is_builtin else 'user'
        if self.nni_version is not None:
            ret['nniVersion'] = self.nni_version
        return ret
