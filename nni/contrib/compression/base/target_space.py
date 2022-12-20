# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import abc
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List

import torch
from torch import Tensor

from nni.compression.pytorch.utils.scaling import Scaling


class TargetType(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    PARAMETER = 'parameter'


class TargetSpace:
    def __init__(self, wrapper: torch.nn.Module, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
        assert target_type in TargetType
        self._wrapper = wrapper
        self._target_name = target_name
        self._target_type = target_type
        self._setting = setting if setting is not None else {}

        self._register_target()

    @property
    def setting(self) -> Dict[str, Any]:
        return deepcopy(self._setting)

    @property
    def target(self) -> Tensor | None:
        if self.type is TargetType.PARAMETER:
            return self._get_wrapper_attr(self._target_name)
        else:
            return None

    @property
    def type(self) -> TargetType:
        return self._target_type

    def _get_wrapper_attr(self, attr_name: str):
        assert hasattr(self._wrapper, attr_name), f'Wrapper {self._wrapper.name} do not have attribute {attr_name}.'
        return getattr(self._wrapper, attr_name)

    def _set_wrapper_attr(self, attr_name: str, val: Any):
        setattr(self._wrapper, attr_name, val)

    def _tensor_setter_helper(self, attr_name: str, val: Tensor | None):
        attr: Tensor | None = self._get_wrapper_attr(attr_name)
        if attr is None:
            self._set_wrapper_attr(attr_name, val)
        else:
            # here using inplace copy for ddp issue, and never set val to a torch.nn.Parameter.
            attr.copy_(val)

    def _register_target(self):
        if self._target_type is TargetType.PARAMETER and not hasattr(self._wrapper, self._target_name):
            assert hasattr(self._wrapper.module, self._target_name)
            target = getattr(self._wrapper.module, self._target_name)
            if isinstance(target, torch.nn.parameter.Parameter):
                self._wrapper.register_parameter(self._target_name, torch.nn.Parameter(target.detach().clone()))
            elif isinstance(target, torch.Tensor):
                self._wrapper.register_buffer(self._target_name, target.detach().clone())
            elif isinstance(target, None):
                self._wrapper.register_buffer(self._target_name, None)
            else:
                raise TypeError(f'Type of {self._target_name} is {type(target)}, can not register to {self._wrapper.name}.')


class PruningTargetSpace(TargetSpace):
    def __init__(self, wrapper: torch.nn.Module, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
        super().__init__(wrapper, target_name, target_type, setting)
        self._register_mask()
        self._scaler: Scaling | None = None

    def _register_mask(self):
        self._wrapper.register_buffer(self._mask_name, None)
        if isinstance(self.target, torch.Tensor):
            self.mask = torch.ones_like(self.target).detach()

    # don't support setter
    @property
    def _mask_name(self) -> str:
        return f'{self._target_name}_mask'

    @property
    def mask(self) -> Tensor | None:
        return self._get_wrapper_attr(self._mask_name)

    @mask.setter
    def mask(self, val: Tensor | None):
        self._tensor_setter_helper(self._mask_name, val)

    # don't support setter
    @property
    def apply_method(self) -> str:
        _method = self.setting.get('apply_method', None)
        _method = _method if _method else 'mul'
        assert _method in ['mul', 'add']
        return _method

    @property
    def sparse_ratio(self) -> float | None:
        return self.setting.get('sparse_ratio', None)

    @sparse_ratio.setter
    def sparse_ratio(self, val: float):
        assert isinstance(val, float)
        self.setting['sparse_ratio'] = val

    @property
    def sparse_threshold(self) -> float | None:
        return self.setting.get('sparse_threshold', None)

    @sparse_threshold.setter
    def sparse_threshold(self, val: float):
        assert isinstance(val, float)
        self.setting['sparse_threshold'] = val

    @property
    def max_sparse_ratio(self) -> float | None:
        return self.setting.get('max_sparse_ratio', None)

    @max_sparse_ratio.setter
    def max_sparse_ratio(self, val: float):
        assert isinstance(val, float)
        self.setting['max_sparse_ratio'] = val

    @property
    def min_sparse_ratio(self) -> float | None:
        return self.setting.get('min_sparse_ratio', None)

    @min_sparse_ratio.setter
    def min_sparse_ratio(self, val: float):
        assert isinstance(val, float)
        self.setting['min_sparse_ratio'] = val

    @property
    def sparse_granularity(self) -> List[int] | str | None:
        return self.setting.get('sparse_granularity', None)

    @sparse_granularity.setter
    def sparse_granularity(self, val: List[int] | str | None):
        assert isinstance(val, str) or val is None or (isinstance(val, abc.Sequence) and all(isinstance(v, int) for v in val))
        self.setting['sparse_granularity'] = val

    @property
    def global_group_id(self) -> int | str | None:
        return self.setting.get('global_group_id', None)

    @global_group_id.setter
    def global_group_id(self, val: int | str):
        assert isinstance(val, (int, str))
        self.setting['global_group_id'] = val

    @property
    def dependency_group_id(self) -> int | str | None:
        return self.setting.get('dependency_group_id', None)

    @dependency_group_id.setter
    def dependency_group_id(self, val: int | str):
        assert isinstance(val, (int, str))
        self.setting['dependency_group_id'] = val

    # don't support setter
    @property
    def internal_metric_block(self) -> int | List[int] | str | None:
        return self.setting.get('internal_metric_block', None)

    # don't support setter
    @property
    def align(self) -> Dict | None:
        return self.setting.get('align', None)


class QuantizationTargetSpace(TargetSpace):
    def __init__(self, wrapper: torch.nn.Module, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
        super().__init__(wrapper, target_name, target_type, setting)
        self._register_scale()

    def _register_scale(self):
        self._wrapper.register_buffer(self._scale_name, None)
        self._wrapper.register_buffer(self._zero_point_name, None)
        qmin, qmax = self._compute_qmin_qmax()
        setattr(self._wrapper, self._qmin_name, qmin)
        setattr(self._wrapper, self._qmax_name, qmax)
        if isinstance(self.target, torch.Tensor):
            self.scale = torch.ones_like(self.target)
            self.zero_point = torch.zeros_like(self.target)

    def _compute_qmin_qmax(self):
        quant_dtype = self.setting.get('quant_dtype', None)
        quant_dtype = quant_dtype if quant_dtype else 'int8'
        if quant_dtype.startswith('int'):
            quant_bit = int(quant_dtype.split('int', 1)[1])
            qmin, qmax = -2 ** (quant_bit - 1) + 1, 2 ** (quant_bit - 1) - 1
        elif quant_dtype.startswith('uint'):
            quant_bit = int(quant_dtype.split('uint', 1)[1])
            qmin, qmax = 0, 2 ** quant_bit - 1
        else:
            raise TypeError(f'Unsupported quant_dtype: {quant_dtype}')
        return qmin, qmax

    @property
    def _scale_name(self) -> str:
        return f'{self._target_name}_scale'

    @property
    def _zero_point_name(self) -> str:
        return f'{self._target_name}_zero_point'

    @property
    def _qmax_name(self) -> str:
        return f'{self._target_name}_qmax'

    @property
    def _qmin_name(self) -> str:
        return f'{self._target_name}_qmin'

    @property
    def scale(self) -> Tensor | None:
        return self._get_wrapper_attr(self._scale_name)

    @scale.setter
    def scale(self, val: Tensor | None):
        self._tensor_setter_helper(self._scale_name, val)

    @property
    def zero_point(self) -> Tensor | None:
        return self._get_wrapper_attr(self._zero_point_name)

    @zero_point.setter
    def zero_point(self, val: Tensor | None) -> Tensor | None:
        self._tensor_setter_helper(self._zero_point_name, val)

    @property
    def qmax(self) -> Tensor | None:
        return self._get_wrapper_attr(self._qmax_name)

    @property
    def qmin(self) -> Tensor | None:
        return self._get_wrapper_attr(self._qmin_name)

    @property
    def apply_method(self) -> str:
        _method = self.setting.get('apply_method', None)
        _method = _method if _method else 'clamp_round'
        assert _method in ['clamp_round']
        return _method


class DistillationTargetSpace(TargetSpace):
    def __init__(self, wrapper: torch.nn.Module, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
        assert target_type is TargetType.INPUT or target_type is TargetType.OUTPUT
        super().__init__(wrapper, target_name, target_type, setting)
        self._buffer = []

    def clean(self):
        self._buffer.clear()

    @property
    def hidden_states(self) -> List[Tensor]:
        return self._buffer

    @property
    def hidden_state(self) -> Tensor | None:
        if len(self._buffer) > 0:
            return self._buffer[-1]
        else:
            return None

    @hidden_state.setter
    def hidden_state(self, val: torch.Tensor):
        if not isinstance(val, torch.Tensor):
            raise TypeError('Only support saving tensor as distillation hidden_state.')
        self._buffer.append(val)

    @property
    def lambda_(self) -> float | None:
        return self.setting.get('lambda', None)

    @lambda_.setter
    def lambda_(self, val: float):
        assert isinstance(val, float)
        self.setting['lambda'] = val

    @property
    def link(self):
        return self.setting.get('link', None)

    @property
    def apply_method(self) -> str:
        _method = self.setting.get('apply_method', None)
        _method = _method if _method else 'mse'
        assert _method in ['mse', 'kl']
        return _method
