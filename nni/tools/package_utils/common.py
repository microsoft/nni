# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = ['AlgoMeta']

from typing import Dict, NamedTuple, Optional

class AlgoMeta(NamedTuple):
    name: str
    class_name: Optional[str]
    accept_class_args: bool
    class_args: Optional[dict]
    validator_class_name: Optional[str]
    algo_type: str  # 'tuner' | 'assessor' | 'advisor'
    is_builtin: bool
    nni_version: Optional[str]

    @staticmethod
    def load(meta: Dict, algo_type: Optional[str] = None) -> 'AlgoMeta':
        if algo_type is None:
            algo_type = meta['algoType']
        return AlgoMeta(
            name=meta['builtinName'],
            class_name=meta['className'],
            accept_class_args=meta.get('acceptClassArgs', True),
            class_args=meta.get('classArgs'),
            validator_class_name=meta.get('classArgsValidator'),
            algo_type=algo_type,
            is_builtin=(meta.get('source') == 'nni'),
            nni_version=meta.get('nniVersion')
        )

    def dump(self) -> Dict:
        ret = {}
        ret['builtinName'] = self.name
        ret['className'] = self.class_name
        if not self.accept_class_args:
            ret['acceptClassArgs'] = False
        if self.class_args is not None:
            ret['classArgs'] = self.class_args
        if self.validator_class_name is not None:
            ret['classArgsValidator'] = self.validator_class_name
        ret['source'] = 'nni' if self.is_builtin else 'user'
        if self.nni_version is not None:
            ret['nniVersion'] = self.nni_version
        return ret
