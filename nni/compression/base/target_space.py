# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import abc
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple

import torch
from torch import Tensor

from ..utils.scaling import Scaling


class TargetType(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    PARAMETER = 'parameter'


class TargetSpace:
    """
    TargetSpace is a container to handle all compression information related to a specific compression target,
    note that it has the ability to register information in the wrapper,
    this because in most compression cases need to register tensor (mask, scale, zero point) as buffer/parameter in the model.

    Parameters
    ----------
    wrapper
        The wrapper of a module.
    target_name
        If the target is a parameter, the name is the attr name in the module, i.e., 'weight', 'bias'.
        If the target is the module input/output, the name should have prefix '_input_'/'_output_',
        then follow a number or name, i.e., '_input_0', '_output_hidden_state',
        number is the position of this input/output in all inputs/outputs,
        name is the input argument name or outputs dict key.
    target_type
        'input' or 'output' or 'parameter'.
    setting
        The compression setting, i.e., {'sparse_ratio': 0.1}.
    """

    def __init__(self, wrapper: torch.nn.Module, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
        assert target_type in TargetType
        self._wrapper = wrapper
        self._target_name = target_name
        self._target_type = target_type
        self._setting = setting if setting is not None else {}

        self._register_target()
        self._shape = None

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
    def shape(self) -> List[int] | None:
        if self.type is TargetType.PARAMETER and self.target is not None:
            return [_ for _ in self.target.shape]
        else:
            return self._shape

    @shape.setter
    def shape(self, val: List[int] | None):
        assert self.type is not TargetType.PARAMETER, 'Can not set shape to a parameter target space.'
        assert val is None or all(isinstance(_, int) for _ in val)
        self._shape = val

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
        if attr is None or val is None:
            self._set_wrapper_attr(attr_name, val)
        else:
            # here using inplace copy for ddp issue, and never set val to a torch.nn.Parameter.
            attr.copy_(val)

    def _register_target(self):
        if self._target_type is TargetType.PARAMETER and not hasattr(self._wrapper, self._target_name):
            assert hasattr(self._wrapper.module, self._target_name)
            target = getattr(self._wrapper.module, self._target_name)
            if isinstance(target, torch.nn.parameter.Parameter):
                self._wrapper.register_parameter(self._target_name, torch.nn.Parameter(target.detach().clone(),
                                                                                       requires_grad=target.requires_grad))
            elif isinstance(target, torch.Tensor):
                self._wrapper.register_buffer(self._target_name, target.detach().clone())
            elif target is None:
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

    @property
    def apply_method(self) -> str:
        _method = self.setting.get('apply_method', None)
        _method = _method if _method else 'mul'
        return _method

    @apply_method.setter
    def apply_method(self, val: str):
        self._setting['apply_method'] = val

    @property
    def sparse_ratio(self) -> float | None:
        return self.setting.get('sparse_ratio', None)

    @sparse_ratio.setter
    def sparse_ratio(self, val: float):
        assert isinstance(val, float)
        self._setting['sparse_ratio'] = val

    @property
    def sparse_threshold(self) -> float | None:
        return self.setting.get('sparse_threshold', None)

    @sparse_threshold.setter
    def sparse_threshold(self, val: float):
        assert isinstance(val, float)
        self._setting['sparse_threshold'] = val

    @property
    def max_sparse_ratio(self) -> float | None:
        return self.setting.get('max_sparse_ratio', None)

    @max_sparse_ratio.setter
    def max_sparse_ratio(self, val: float):
        assert isinstance(val, float)
        self._setting['max_sparse_ratio'] = val

    @property
    def min_sparse_ratio(self) -> float | None:
        return self.setting.get('min_sparse_ratio', None)

    @min_sparse_ratio.setter
    def min_sparse_ratio(self, val: float):
        assert isinstance(val, float)
        self._setting['min_sparse_ratio'] = val

    @property
    def granularity(self) -> List[int] | Tuple[List[int], str, int] | str | None:
        return self.setting.get('granularity', None)

    @granularity.setter
    def granularity(self, val: List[int] | Tuple[List[int], str, int] | str | None):
        if isinstance(val, abc.Sequence) and not isinstance(val, str):
            assert all(isinstance(v, int) for v in val) or \
                   (all(isinstance(v, int) for v in val[0]) and  # type: ignore
                    isinstance(val[1], str) if len(val) > 1 else True and \
                    isinstance(val[2], int) if len(val) > 2 else True)
        else:
            assert isinstance(val, str) or val is None
        self._setting['granularity'] = val

    @property
    def global_group_id(self) -> int | str | None:
        return self.setting.get('global_group_id', None)

    @global_group_id.setter
    def global_group_id(self, val: int | str):
        assert isinstance(val, (int, str))
        self._setting['global_group_id'] = val

    @property
    def dependency_group_id(self) -> int | str | None:
        return self.setting.get('dependency_group_id', None)

    @dependency_group_id.setter
    def dependency_group_id(self, val: int | str):
        assert isinstance(val, (int, str))
        self._setting['dependency_group_id'] = val

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
        self._scaler: Scaling | None = None

    def _register_scale(self):
        self._wrapper.register_buffer(self._scale_name, None)
        self._wrapper.register_buffer(self._zero_point_name, None)
        self._wrapper.register_buffer(self._tracked_max_name, None)
        self._wrapper.register_buffer(self._tracked_min_name, None)
        qmin, qmax = self._compute_qmin_qmax()
        setattr(self._wrapper, self._qmin_name, qmin)
        setattr(self._wrapper, self._qmax_name, qmax)

    def _compute_qmin_qmax(self):
        quant_dtype = self.quant_dtype if self.quant_dtype else 'int8'
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
    def _tracked_min_name(self) -> str:
        return f'{self._target_name}_tracked_min'

    @property
    def _tracked_max_name(self) -> str:
        return f'{self._target_name}_tracked_max'

    @property
    def quant_bits(self) -> int:
        quant_dtype = self.quant_dtype if self.quant_dtype else 'int8'
        if quant_dtype.startswith('int'):
            quant_bit = int(quant_dtype.split('int', 1)[1])
        elif quant_dtype.startswith('uint'):
            quant_bit = int(quant_dtype.split('uint', 1)[1])
        else:
            raise TypeError(f'Unsupported quant_dtype: {quant_dtype}')
        return quant_bit

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
    def qmax(self) -> int:
        return self._get_wrapper_attr(self._qmax_name)

    @property
    def qmin(self) -> int:
        return self._get_wrapper_attr(self._qmin_name)

    @property
    def tracked_min(self) -> Tensor | None:
        return self._get_wrapper_attr(self._tracked_min_name)

    @tracked_min.setter
    def tracked_min(self, val: Tensor | None) -> Tensor | None:
        self._tensor_setter_helper(self._tracked_min_name, val)

    @property
    def tracked_max(self) -> Tensor | None:
        return self._get_wrapper_attr(self._tracked_max_name)

    @tracked_max.setter
    def tracked_max(self, val: Tensor | None) -> Tensor | None:
        self._tensor_setter_helper(self._tracked_max_name, val)

    @property
    def granularity(self) -> List[int] | Tuple[List[int], str, int] | str | None:
        return self.setting.get('granularity', None)

    @granularity.setter
    def granularity(self, val: List[int] | Tuple[List[int], str, int] | str | None):
        if isinstance(val, abc.Sequence) and not isinstance(val, str):
            assert all(isinstance(v, int) for v in val) or \
                   (all(isinstance(v, int) for v in val[0]) and  # type: ignore
                    isinstance(val[1], str) if len(val) > 1 else True and \
                    isinstance(val[2], int) if len(val) > 2 else True)
        else:
            assert isinstance(val, str) or val is None
        self._setting['granularity'] = val

    @property
    def quant_dtype(self) -> str:
        return self.setting.get('quant_dtype', None)

    @property
    def quant_scheme(self) -> Literal['affine', 'symmetric'] | None:
        return self.setting.get('quant_scheme', None)

    @property
    def apply_method(self) -> str:
        _method = self.setting.get('apply_method', None)
        _method = _method if _method else 'clamp_round'
        return _method

    @apply_method.setter
    def apply_method(self, val: str):
        self._setting['apply_method'] = val


class DistillationTargetSpace(TargetSpace):
    def __init__(self, wrapper: torch.nn.Module, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
        assert target_type is TargetType.INPUT or target_type is TargetType.OUTPUT
        super().__init__(wrapper, target_name, target_type, setting)
        self._buffer = []
        # 'append' will record each observed tensor and cost a lot memory, 'refresh' only record the latest tensor.
        self._buffer_mode = 'refresh'

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
        if self._buffer_mode == 'append':
            self._buffer.append(val)
        elif self._buffer_mode == 'refresh':
            self._buffer.clear()
            self._buffer.append(val)
        else:
            raise RuntimeError(f'Unsupported buffer mode: {self._buffer_mode}')

    @property
    def lambda_(self) -> float | None:
        return self.setting.get('lambda', None)

    @lambda_.setter
    def lambda_(self, val: float):
        assert isinstance(val, float)
        self._setting['lambda'] = val

    @property
    def link(self) -> str | List[str] | Tuple[str]:
        return self.setting.get('link', None)

    @link.setter
    def link(self, val: str | List[str] | Tuple[str]):
        assert isinstance(val, str) or all(isinstance(v, str) for v in val)
        self._setting['link'] = val

    @property
    def apply_method(self) -> str:
        _method = self.setting.get('apply_method', None)
        _method = _method if _method else 'mse'
        assert _method in ['mse', 'kl']
        return _method
