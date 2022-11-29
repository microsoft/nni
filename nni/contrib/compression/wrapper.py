from __future__ import annotations

import logging
import inspect
from typing import Any, Dict, List, Tuple, Type, Union, Literal

import torch
from torch import Tensor

from .settings import INPUT_PREFIX, OUTPUT_PREFIX
from .target_space import (
    TargetSpace,
    TargetType,
    PruningTargetSpace,
    QuantizationTargetSpace,
    DistillationTargetSpace
)
from .utils import select_modules, canonicalize_settings

_logger = logging.getLogger(__name__)
SMALL_MASK_VALUE = -1000
OUTPUT_FORMAT = Union[Tensor, Tuple[Tensor, Any], Dict[str, Union[Tensor, Any]]]


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, module_name: str, config: Dict[str, Dict[str, Any]] | None = None):
        """
        Two changes will be done during initialization. One is an attribute named ``_nni_wrapper`` will be set to original module,
        this attribute points to this wrapper in the original module.
        The other is the original ``module.forward`` will be replaced by ``module._nni_wrapper.forward``.

        The module can be unwrapped by ``module._nni_wrapper.unwrap()``.

        Parameters
        ----------
        module
            The torch.nn.Module to be wrapped.
        module_name
            The name of the module in the original model.
        config
            The config is a dict which contains keys (not required): ``pruning``, ``quantization``, ``distillation``.
        """
        super().__init__()

        # origin layer information
        assert isinstance(module, torch.nn.Module)
        object.__setattr__(self, 'module', module)
        self.module: torch.nn.Module
        self.module_forward = self.module.forward
        self.name = module_name
        self.config = config if config is not None else {}

        # the arguments' name of self.module.forward
        self._input_args_names = inspect.getfullargspec(self.module.forward).args[1:]

        # create target spaces
        self.pruning_target_spaces: Dict[str, PruningTargetSpace] = {}
        self.quantization_target_spaces: Dict[str, QuantizationTargetSpace] = {}
        self.distillation_target_spaces: Dict[str, DistillationTargetSpace] = {}

        if 'pruning' in config:
            self.extend_target_spaces(config.get('pruning'), 'pruning')
        if 'quantization' in config:
            self.extend_target_spaces(config.get('quantization'), 'quantization')
        if 'distillation' in config:
            self.extend_target_spaces(config.get('distillation'), 'distillation')

        self._frozen = False

    def extra_repr(self) -> str:
        return f'module={self.module.__class__.__name__}({self.module.extra_repr()}), module_name={self.name}'

    @property
    def is_frozen(self) -> bool:
        # if the wrapper is frozen, should not update any state of this wrapper, i.e., pruning masks or quant scale.
        return self._frozen

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def wrap(self):
        if hasattr(self.module, '_nni_wrapper') and getattr(self.module, '_nni_wrapper') == self:
            warn_msg = f'Wrapper of {self.name} is wrapped, no need to wrap again.'
            _logger.warning(warn_msg)
            return
        assert not hasattr(self.module, '_nni_wrapper'), f'{self.name} is already wrapped by another wrapper, can not wrap it again.'
        setattr(self.module, '_nni_wrapper', self)
        self.module.forward = self.forward

        for target_name, target_space in self.pruning_target_spaces.items():
            if target_space.type == TargetType.PARAMETER and isinstance(target_space.target, torch.nn.Parameter):
                delattr(self.module, target_name)
                self.module.register_buffer(target_name, target_space.target.data.clone())

        for target_name, target_space in self.quantization_target_spaces.items():
            if target_space.type == TargetType.PARAMETER and isinstance(target_space.target, torch.nn.Parameter):
                delattr(self.module, target_name)
                self.module.register_buffer(target_name, target_space.target.data.clone())

    def unwrap(self):
        if not hasattr(self.module, '_nni_wrapper'):
            warn_msg = f'{self.name} is not wrapped, no need to unwrap.'
            _logger.warning(warn_msg)

        for target_name, target_space in self.pruning_target_spaces.items():
            if target_space.type == TargetType.PARAMETER and isinstance(target_space.target, torch.nn.Parameter):
                delattr(self.module, target_name)
                self.module.register_parameter(target_name, torch.nn.Parameter(target_space.target.data.clone()))

        for target_name, target_space in self.quantization_target_spaces.items():
            if target_space.type == TargetType.PARAMETER and isinstance(target_space.target, torch.nn.Parameter):
                delattr(self.module, target_name)
                self.module.register_parameter(target_name, torch.nn.Parameter(target_space.target.data.clone()))

        self.module.forward = self.module_forward
        delattr(self.module, '_nni_wrapper')

    def extend_target_spaces(self, sub_config: Dict[str, Any], mode: Literal['pruning', 'quantization', 'distillation']):
        assert mode in ['pruning', 'quantization', 'distillation']
        # assert not hasattr(self.module, '_nni_wrapper'), \
        #     f'Module {self.name} is wrapped, please unwrap the module before extend target spaces.'

        if mode == 'pruning':
            target_spaces = self.pruning_target_spaces
        if mode == 'quantization':
            target_spaces = self.quantization_target_spaces
        if mode == 'distillation':
            target_spaces = self.distillation_target_spaces

        settings = canonicalize_settings(self.module, sub_config, mode)
        inter_sec = set(target_spaces.keys()).intersection(settings.keys())
        for name in inter_sec:
            # if need to update target space setting, should directly update it, not extend a repeat target.
            warn_msg = f'{name} have already configured, the new config will be ignored.'
            _logger.warning(warn_msg)
            settings.pop(name)
        target_spaces.update(self._create_target_spaces(settings, PruningTargetSpace))

    def update_masks(self, masks: Dict[str, torch.Tensor]):
        """
        Parameters
        ----------
        masks
            A masks dict, the key should be the target name in the ``self.pruning_target_spaces``,
            and the value is a Tensor contains 0 or 1.
        """
        if self.is_frozen:
            warn_msg = f'Can not update masks for frozen wrapper {self.name}, skip this update.'
            _logger.warning(warn_msg)
        for target_name, mask in masks.items():
            assert target_name in self.pruning_target_spaces, f'{target_name} is not set to a pruning target in {self.name}.'
            self.pruning_target_spaces[target_name].mask = mask

    def _create_target_spaces(self, settings: Dict[str, Dict], target_space_cls: Type[TargetSpace]) -> Dict[str, TargetSpace]:
        target_spaces = {}
        for target_name, setting in settings.items():
            target_type = TargetType.INPUT if target_name.startswith(INPUT_PREFIX) else TargetType.OUTPUT \
                if target_name.startswith(OUTPUT_PREFIX) else TargetType.PARAMETER
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
            if target_space.apply_method == 'mul':
                return torch.mul(target, target_space.mask)
            elif target_space.apply_method == 'add':
                trans_mask = torch.where(target_space.mask == 1, torch.zeros_like(target_space.mask), SMALL_MASK_VALUE)
                return torch.add(target, trans_mask)
            else:
                raise TypeError('Only `mul` and `add` are supported for mask `apply_method`.')
        else:
            return target

    def _apply_quant_helper(self, target: Tensor, target_space: QuantizationTargetSpace) -> Tensor:
        if target_space.scale is not None and target_space.zero_point is not None:
            if target_space.apply_method == 'clamp_round':
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
            new_outputs = self.patch_helper(target_name, outputs)
        elif isinstance(outputs, (list, tuple)):
            new_outputs = []
            for idx, target in enumerate(outputs):
                target_name = f'{OUTPUT_PREFIX}_{idx}'
                new_outputs.append(self.patch_helper(target_name, target))
        elif isinstance(outputs, dict):
            new_outputs = {}
            for output_name, target in outputs.items():
                target_name = f'{OUTPUT_PREFIX}_{output_name}'
                new_outputs[output_name] = self.patch_helper(target_name, target)
        else:
            raise TypeError(f'Only support return Tensor/list/dict, but got {type(outputs)}')
        return new_outputs

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

        outputs = self.module_forward(**inputs)
        outputs = self.patch_outputs(outputs)
        return outputs


def register_wrapper(model: torch.nn.Module, config_list: List[Dict[str, Any]],
                     mode: Literal['pruning', 'quantization', 'distillation'],
                     existed_wrappers: Dict[str, ModuleWrapper] | None = None) -> Dict[str, ModuleWrapper]:
    assert mode in ['pruning', 'quantization', 'distillation']
    existed_wrappers = existed_wrappers if existed_wrappers else {}
    module_wrappers = {k: v for k, v in existed_wrappers.items()}
    for _, wrapper in module_wrappers.items():
        wrapper.freeze()
    for config in config_list:
        modules, public_config = select_modules(model, config)
        for module_name, module in modules.items():
            if module_name in module_wrappers:
                wrapper = module_wrappers[module_name]
                wrapper.unfreeze()
                wrapper.extend_target_spaces(public_config, mode)
            else:
                wrapper = ModuleWrapper(module, module_name, {mode: public_config})
                module_wrappers[module_name] = wrapper
    return module_wrappers
