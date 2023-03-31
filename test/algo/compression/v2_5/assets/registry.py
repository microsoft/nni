# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from nni.common.concrete_trace_utils import concrete_trace


@dataclass
class ModelInfo:
    need_run: bool = False
    need_auto_set_dependency: bool = False
    
    leaf_module: Tuple[torch.nn.Module] = ()
    fake_middle_class: Optional[Tuple[torch.nn.Module]] = None
    forward_function_name: str = 'forward'
    

class ModelZoo(dict):
    def __init__(self):
        super().__init__()
    
    def register(self, package: str, name: str, func: Callable, dummy_inputs: Callable, config_list: Callable, skip_reason: str = '', **kwargs):
        self[package] = self.get_(package, {})
        self[package][name] = mark_default(func, dummy_inputs, config_list, skip_reason, name, **kwargs)
        
    def get_(self, __key, __default=None):
        return super().get(__key, __default)
        
    def get(self, package: str, names: Union[str, Tuple[str]] = None, skip_reason: str = None):
        if names is None:
            ret = [fn for fn in self[package].values()]
        elif isinstance(names, str):
            ret = self[package][names]
        elif isinstance(names, tuple):
            ret = [self[package][name] for name in names]
        else:
            raise ValueError(f'names must be str or tuple, got {type(names)}')
        
        if skip_reason is not None:
            ret = filter(lambda fn: fn.skip_reason == skip_reason, ret)
        
        return ret
        
    def skip_summary(self, packages: Union[str, Tuple[str]] = None):
        try:
            import tabulate
        except:
            raise ImportError('Please install `tabulate` by "pip install tabulate"!')
        skip_summary = []
        if packages is None:
            packages = self.keys()
        
        if isinstance(packages, str):
            packages = (packages, )

        for package in packages:
            for name, fn in self[package].items():
                if fn.skip_reason == '':
                    continue
                entry = {
                    'package': package,
                    'model name': name,
                    'skip_reason': fn.skip_reason
                }
                skip_summary.append(entry)
        return tabulate.tabulate(skip_summary, headers='keys', tablefmt="github")

def mark_default(func, dummy_inputs, config_list, skip_reason: str = '', name: str = None, **extra_kwargs):
    def mod_fn(*args, **kwargs) -> torch.nn.Module:
        mod = func(*args, **kwargs)
        mod.dummy_inputs = dummy_inputs(mod)
        mod.config_list = config_list(mod)
        mod.extra_info = ModelInfo(**extra_kwargs)
        return mod
    if name is not None:
        mod_fn.__name__ = name
    else:
        mod_fn.__name__ = func.__name__
    mod_fn.skip_reason = skip_reason
    return mod_fn

model_zoo = ModelZoo()