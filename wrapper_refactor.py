from __future__ import annotations

from copy import deepcopy
from enum import Enum
import inspect
from typing import Any, Dict, List, Tuple, Type, Union

import torch
from torch import Tensor

INPUT_PREFIX = '_input'
OUTPUT_PREFIX = '_output'
SMALL_MASK_VALUE = -1000
OUTPUT_FORMAT = Union[Tensor, Tuple[Tensor, Any], Dict[str, Union[Tensor, Any]]]


class TargetType(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    PARAMETER = 'parameter'


class TargetSpace:
    def __init__(self, wrapper: ModuleWrapper, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
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
            attr.copy_(val)

    def _register_target(self):
        if self._target_type is TargetType.PARAMETER and not hasattr(self._wrapper, self._target_name):
            assert hasattr(self._wrapper.module, self._target_name)
            target = getattr(self._wrapper.module, self._target_name)
            if isinstance(target, torch.nn.parameter.Parameter):
                self._wrapper.register_parameter(self._target_name, target.data.clone())
            elif isinstance(target, torch.Tensor):
                self._wrapper.register_buffer(self._target_name, target.data.clone())
            elif isinstance(target, None):
                self._wrapper.register_buffer(self._target_name, None)
            else:
                raise TypeError(f'Type of {self._target_name} is {type(target)}, can not register to {self._wrapper.name}.')


class PruningTargetSpace(TargetSpace):
    def __init__(self, wrapper: ModuleWrapper, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
        super().__init__(wrapper, target_name, target_type, setting)
        self._register_mask()

    def _register_mask(self):
        self._wrapper.register_buffer(self._mask_name, None)
        if isinstance(self.target, torch.Tensor):
            self.mask = torch.ones_like(self.target).detach()

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
        _method = self.setting.get('apply_method', 'mul')
        assert _method in ['mul', 'add']
        return _method


class QuantizationTargetSpace(TargetSpace):
    def __init__(self, wrapper: ModuleWrapper, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
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
        quant_dtype = self.setting.get('quant_dtype', 'int8')
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
        _method = self.setting.get('apply_method', 'clamp_round')
        assert _method in ['clamp_round']
        return _method


class DistillationTargetSpace(TargetSpace):
    def __init__(self, wrapper: ModuleWrapper, target_name: str, target_type: TargetType, setting: Dict[str, Any] | None = None):
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
    def hidden_state(self, val):
        self._buffer.append(val)

    @property
    def apply_method(self) -> str:
        _method = self.setting.get('apply_method', 'mse')
        assert _method in ['mse', 'kl']
        return _method


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, module_name: str, config: Dict[str, Any]):
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        # config information
        self.config = config

        # the arguments' name of self.module.forward
        self._input_args_names = inspect.getfullargspec(self.module.forward).args[1:]

        # create target spaces
        pruning_target_settings = self._canonicalize_target_settings(self.config.get('pruning', {}))
        quantization_target_settings = self._canonicalize_target_settings(self.config.get('quantization', {}))
        distillation_target_settings = self._canonicalize_target_settings(self.config.get('distillation', {}))

        self.pruning_target_spaces: Dict[str, PruningTargetSpace] = self._create_target_spaces(pruning_target_settings, PruningTargetSpace)
        self.quantization_target_spaces: Dict[str, QuantizationTargetSpace] = self._create_target_spaces(quantization_target_settings, QuantizationTargetSpace)
        self.distillation_target_spaces: Dict[str, DistillationTargetSpace] = self._create_target_spaces(distillation_target_settings, DistillationTargetSpace)

    def __getattr__(self, name: str) -> Union[Tensor, torch.Module]:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.module.__getattr__(name)

    def _canonicalize_target_settings(self, sub_config: Dict[str, Any]) -> Dict[str, Dict]:
        target_names: List[str] = deepcopy(sub_config.get('target_names', []))
        settings: Dict[str, Dict] = deepcopy(sub_config.get('settings', {}))
        c_settings = {}

        if INPUT_PREFIX in target_names:
            target_names.pop(INPUT_PREFIX)
            target_names.extend([f'{INPUT_PREFIX}_{name}' for name in self._input_args_names])
        if OUTPUT_PREFIX in target_names:
            target_names.pop(OUTPUT_PREFIX)
            # another way is compressing all outputs
            target_names.append(f'{OUTPUT_PREFIX}_0')

        # only inherit the first level key
        for target_name in target_names:
            default_setting = deepcopy(settings.get('_default', {}))
            if target_name.startswith(INPUT_PREFIX):
                default_setting.update((settings.get(INPUT_PREFIX, {})))
            if target_name.startswith(OUTPUT_PREFIX):
                default_setting.update((settings.get(OUTPUT_PREFIX, {})))
            default_setting.update(settings.get(target_name, {}))
            c_settings[target_name] = deepcopy(default_setting)

        return c_settings

    def _create_target_spaces(self, settings: Dict[str, Dict], target_space_cls: Type[TargetSpace]) -> Dict[str, TargetSpace]:
        target_spaces = {}
        for target_name, setting in settings.items():
            target_type = TargetType.INPUT if target_name.startswith(INPUT_PREFIX) else TargetType.OUTPUT if target_name.startswith(OUTPUT_PREFIX) else TargetType.PARAMETER
            target_space = target_space_cls(self, target_name, target_type, setting)
            target_spaces[target_name] = target_space
        return target_spaces

    def _transfer_input_name(self, input_name_or_idx: str | int, contx2idx: bool = True) -> str | int:
        if contx2idx:
            if isinstance(input_name_or_idx, int) or input_name_or_idx.isdigit():
                idx = int(input_name_or_idx)
                assert idx < len(self._input_args_names)
            else:
                assert input_name_or_idx in self._input_args_names
                idx = self._input_args_names.index(input_name_or_idx)
            return idx
        else:
            if isinstance(input_name_or_idx, int) or input_name_or_idx.isdigit():
                idx = int(input_name_or_idx)
                assert idx < len(self._input_args_names)
                contx = self._input_args_names[idx]
            else:
                contx = input_name_or_idx
                assert contx in self._input_args_names
            return contx

    def _apply_mask_helper(self, target: Tensor, target_space: PruningTargetSpace) -> Tensor:
        if target_space.mask is not None:
            if target_space.apply_method is 'mul':
                return torch.mul(target, target_space.mask)
            elif target_space.apply_method is 'add':
                trans_mask = torch.where(target_space.mask == 1, torch.zeros_like(target_space.mask), SMALL_MASK_VALUE)
                return torch.add(target, trans_mask)
            else:
                raise TypeError('Only `mul` and `add` are supported for mask `apply_method`.')
        else:
            return target

    def _apply_quant_helper(self, target: Tensor, target_space: QuantizationTargetSpace) -> Tensor:
        if target_space.scale is not None and target_space.zero_point is not None:
            if target_space.apply_method is 'clamp_round':
                transformed_target = target_space.zero_point + target / target_space.scale
                quantized_target = torch.round(torch.clamp(transformed_target, target_space.qmin, target_space.qmax))
                dequantized_target = (quantized_target - target_space.zero_point) * target_space.scale
            else:
                raise TypeError('Only `clamp_round` are supported for quantization `apply_method`.')
            return dequantized_target
        else:
            return target

    def _distil_observe_helper(self, target: Tensor, target_space: DistillationTargetSpace) -> Tensor:
        # NOTE: here will have a risk if target is not Tensor, we don't know if it can be deepcopy and if it will be changed.
        target_space.hidden_state = target.clone().detach() if isinstance(target, Tensor) else target
        return target

    def patch_helper(self, target_name: str, target: Tensor) -> Tensor:
        # apply quantize-dequantize -> apply pruning mask -> record state for distil
        if target_name in self.quantization_target_spaces:
            target = self._apply_quant_helper(target, self.quantization_target_spaces[target_name])
        if target_name in self.pruning_target_spaces:
            target = self._apply_mask_helper(target, self.pruning_target_spaces[target_name])
        if target_name in self.distillation_target_spaces:
            target = self._distil_observe_helper(target, self.distillation_target_spaces[target_name])
        return target

    def patch_inputs(self, *args, **kwargs) -> Dict[str, Tensor | Any]:
        # all inputs will convert in kwargs
        for arg_idx, arg_value in enumerate(args):
            arg_name = self._transfer_input_name(arg_idx, contx2idx=False)
            kwargs[arg_name] = arg_value

        new_kwargs = {}
        for arg_name, arg_value in kwargs.items():
            target_name = f'{INPUT_PREFIX}_{arg_name}'
            arg_value = self.patch_helper(target_name, arg_value)
            new_kwargs[arg_name] = arg_value
        return new_kwargs

    def patch_params(self, targets_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        new_target_dict = {}
        for target_name, target in targets_dict.items():
            target = self.patch_helper(target_name, target)
            new_target_dict[target_name] = target
        return new_target_dict

    def patch_outputs(self, outputs: OUTPUT_FORMAT) -> OUTPUT_FORMAT:
        if isinstance(outputs, Tensor):
            target_name = f'{OUTPUT_PREFIX}_0'
            outputs = self.patch_helper(target_name, outputs)
        elif isinstance(outputs, (list, tuple)):
            new_outputs = []
            for idx, target in enumerate(outputs):
                target_name = f'{OUTPUT_PREFIX}_{idx}'
                new_outputs.append(self.patch_helper(target_name, target))
            return new_outputs
        elif isinstance(outputs, dict):
            new_outputs = {}
            for output_name, target in outputs.items():
                target_name = f'{OUTPUT_PREFIX}_{output_name}'
                new_outputs[output_name] = self.patch_helper(target_name, target)
        else:
            raise TypeError(f'Only support return Tensor/list/dict, but got {type(outputs)}')

    def forward(self, *args, **kwargs):
        inputs = self.patch_inputs(*args, **kwargs)

        params_dict = {}
        params_dict.update({k: v.target for k, v in self.pruning_target_spaces.items() if v.type is TargetType.PARAMETER})
        params_dict.update({k: v.target for k, v in self.quantization_target_spaces.items() if v.type is TargetType.PARAMETER})
        params_dict.update({k: v.target for k, v in self.distillation_target_spaces.items() if v.type is TargetType.PARAMETER})
        params_dict = self.patch_params(params_dict)
        for target_name, wrapper_param in params_dict.items():
            module_param: Tensor = getattr(self.module, target_name)
            module_param.copy_(wrapper_param)

        outputs = self.module(**inputs)
        outputs = self.patch_outputs(outputs)
        return outputs


class WrapperHandler:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.named_modules()
