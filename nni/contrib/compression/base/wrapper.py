# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import inspect
from typing import Any, Callable, Dict, List, Tuple, Type, Union, Literal

import torch
from torch import Tensor

from .apply_method import pruning_apply_methods, quant_apply_methods
from .config import select_modules_by_config
from .setting import INPUT_PREFIX, OUTPUT_PREFIX, canonicalize_settings
from .target_space import (
    TargetSpace,
    TargetType,
    PruningTargetSpace,
    QuantizationTargetSpace,
    DistillationTargetSpace
)

_logger = logging.getLogger(__name__)
OUTPUT_FORMAT = Union[Tensor, Any, Tuple[Tensor, Any], Dict[str, Union[Tensor, Any]]]


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
        self._input_args_spec = inspect.getfullargspec(self.module.forward)

        # create target spaces
        self.pruning_target_spaces: Dict[str, PruningTargetSpace] = {}
        self.quantization_target_spaces: Dict[str, QuantizationTargetSpace] = {}
        self.distillation_target_spaces: Dict[str, DistillationTargetSpace] = {}

        if 'pruning' in self.config:
            self.extend_target_spaces(self.config['pruning'], 'pruning')
        if 'quantization' in self.config:
            self.extend_target_spaces(self.config['quantization'], 'quantization')
        if 'distillation' in self.config:
            self.extend_target_spaces(self.config['distillation'], 'distillation')

        self._frozen = False
        # By default, input/output shape will be track during forward,
        # more track functions can be registered by ``ModuleWrapper.register_track_info_func``.
        # An example please refer ``track_target_shape``.
        self._track_funcs: List[Callable[[ModuleWrapper, str, Tensor], None]] = [track_target_shape]

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
                self.module.register_buffer(target_name, target_space.target.detach().clone())

        for target_name, target_space in self.quantization_target_spaces.items():
            if target_space.type == TargetType.PARAMETER and isinstance(target_space.target, torch.nn.Parameter):
                delattr(self.module, target_name)
                self.module.register_buffer(target_name, target_space.target.detach().clone())

    def unwrap(self):
        if not hasattr(self.module, '_nni_wrapper'):
            warn_msg = f'{self.name} is not wrapped, no need to unwrap.'
            _logger.warning(warn_msg)

        for target_name, target_space in self.pruning_target_spaces.items():
            if target_space.type == TargetType.PARAMETER and isinstance(target_space.target, torch.nn.Parameter):
                delattr(self.module, target_name)
                self.module.register_parameter(target_name, torch.nn.Parameter(target_space.target.detach().clone()))

        for target_name, target_space in self.quantization_target_spaces.items():
            if target_space.type == TargetType.PARAMETER and isinstance(target_space.target, torch.nn.Parameter):
                delattr(self.module, target_name)
                self.module.register_parameter(target_name, torch.nn.Parameter(target_space.target.detach().clone()))

        self.module.forward = self.module_forward
        delattr(self.module, '_nni_wrapper')

    def extend_target_spaces(self, sub_config: Dict[str, Any], mode: Literal['pruning', 'quantization', 'distillation']):
        assert mode in ['pruning', 'quantization', 'distillation']

        if mode == 'pruning':
            target_spaces = self.pruning_target_spaces
            target_space_cls = PruningTargetSpace
        elif mode == 'quantization':
            target_spaces = self.quantization_target_spaces
            target_space_cls = QuantizationTargetSpace
        else:
            target_spaces = self.distillation_target_spaces
            target_space_cls = DistillationTargetSpace

        settings = canonicalize_settings(self.module, sub_config, mode)
        inter_sec = set(target_spaces.keys()).intersection(settings.keys())
        for name in inter_sec:
            # if need to update target space setting, should directly update it, not extend a repeat target.
            warn_msg = f'{name} have already configured, the new config will be ignored.'
            _logger.warning(warn_msg)
            settings.pop(name)
        new_target_spaces = self._create_target_spaces(settings, target_space_cls)
        target_spaces.update(new_target_spaces)  # type: ignore

        # return the new registered target spaces
        return new_target_spaces

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

    def update_calibration_config(self, calibration_config):
        # TODO: implement it
        raise NotImplementedError()

    def _create_target_spaces(self, settings: Dict[str, Dict], target_space_cls: Type[TargetSpace]) -> Dict[str, TargetSpace]:
        target_spaces = {}
        for target_name, setting in settings.items():
            target_type = TargetType.INPUT if target_name.startswith(INPUT_PREFIX) else TargetType.OUTPUT \
                if target_name.startswith(OUTPUT_PREFIX) else TargetType.PARAMETER
            target_space = target_space_cls(self, target_name, target_type, setting)
            target_spaces[target_name] = target_space
        return target_spaces

    def _transfer_input(self, *args, **kwargs) -> Tuple:
        # -1 because the first arg of forward is `self`, not in args
        pos_args_num = len(self._input_args_spec.args) - 1
        pos_args = args[:pos_args_num]
        if len(pos_args) < pos_args_num:
            pos_args += tuple(kwargs.pop(k) for k in self._input_args_spec.args[len(pos_args) + 1:])
        var_args = args[pos_args_num:]
        kwonly_args = {k: kwargs.pop(k) for k in self._input_args_spec.kwonlyargs}
        return pos_args, var_args, kwonly_args, kwargs

    def _transfer_args_name(self, input_name_or_idx: str | int, contx2idx: bool = True) -> str | int:
        if contx2idx:
            if isinstance(input_name_or_idx, int) or input_name_or_idx.isdigit():
                idx = int(input_name_or_idx)
                assert idx < len(self._input_args_spec.args)
            else:
                assert input_name_or_idx in self._input_args_spec.args
                idx = self._input_args_spec.args.index(input_name_or_idx)
            return idx
        else:
            if isinstance(input_name_or_idx, int) or input_name_or_idx.isdigit():
                idx = int(input_name_or_idx)
                assert idx < len(self._input_args_spec.args)
                contx = self._input_args_spec.args[idx]
            else:
                contx = input_name_or_idx
                assert contx in self._input_args_spec.args
            return contx

    def _apply_mask_helper(self, target: Tensor, target_space: PruningTargetSpace) -> Tensor:
        # NOTE: if mask is None, and is registered as buffer during training, will cause DDP sync problem.
        if target_space.mask is not None:
            if target_space.apply_method in pruning_apply_methods:
                return pruning_apply_methods[target_space.apply_method](target, target_space)
            else:
                raise TypeError(f'Only {list(pruning_apply_methods.keys())} are supported for mask `apply_method`.')
        elif target_space.type is TargetType.PARAMETER:
            # Prevent registering buffer as a parameter
            return target * 1.
        else:
            return target

    def _apply_quant_helper(self, target: Tensor, target_space: QuantizationTargetSpace) -> Tensor:
        # NOTE: if scale or zero point is None, and is registered as buffer during training, will cause DDP sync problem.
        if target_space.scale is not None and target_space.zero_point is not None:
            if target_space.apply_method in quant_apply_methods:
                dequantized_target: Tensor = quant_apply_methods[target_space.apply_method](target, target_space)
            else:
                raise TypeError(f'Only {list(quant_apply_methods.keys())} are supported for quantization `apply_method`.')
            return dequantized_target
        elif target_space.type is TargetType.PARAMETER:
            # Prevent registering buffer as a parameter
            return target * 1.
        else:
            return target

    def _distil_observe_helper(self, target: Tensor, target_space: DistillationTargetSpace) -> Tensor:
        # NOTE: here will have a risk, we don't know if target will be inplace changed in the following.
        target_space.hidden_state = target.clone().detach()
        return target

    def _track_info(self, target_name: str, target: Tensor):
        # this function will be called in path_helper at first.
        for track_func in self._track_funcs:
            track_func(self, target_name, target)

    def register_track_func(self, track_func: Callable[[ModuleWrapper, str, Tensor], None]):
        """
        Execute ``track_func`` sequentially according to the order of registration.

        Parameters
        ----------
        track_func
            The inputs of track_func are (wrapper, target_name, target).
            TODO: add a simple track_func example.
        """
        self._track_funcs.append(track_func)

    def patch_helper(self, target_name: str, target: Tensor | Any) -> Tensor | Any:
        self._track_info(target_name=target_name, target=target)
        # apply quantize-dequantize -> apply pruning mask -> record state for distil
        if target_name in self.quantization_target_spaces:
            target = self._apply_quant_helper(target, self.quantization_target_spaces[target_name])
        if target_name in self.pruning_target_spaces:
            target = self._apply_mask_helper(target, self.pruning_target_spaces[target_name])
        if target_name in self.distillation_target_spaces:
            target = self._distil_observe_helper(target, self.distillation_target_spaces[target_name])
        return target

    def patch_inputs(self, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        # NOTE: even here has an interface to compress `varargs`, `varkw`, but nni doesn't suppot compress them right now.
        pos_args, varargs, kwonly_args, varkw = self._transfer_input(*args, **kwargs)

        new_args = []
        for idx, arg_value in enumerate(pos_args):
            target_name = f'{INPUT_PREFIX}{idx}'
            new_args.append(self.patch_helper(target_name, arg_value))
        # NOTE: by default, we do not support varargs, if it is need, override the patch_helper
        new_args.extend(self.patch_helper(f'{INPUT_PREFIX}{self._input_args_spec.varargs}', varargs))

        new_kwargs = {}
        for key, value in kwonly_args.items():
            target_name = f'{INPUT_PREFIX}{key}'
            new_kwargs[key] = self.patch_helper(target_name, value)
        # NOTE: by default, we do not support varkw, if it is need, override the patch_helper
        new_kwargs.update(self.patch_helper(f'{INPUT_PREFIX}{self._input_args_spec.varkw}', varkw))  # type: ignore

        return new_args, new_kwargs

    def patch_params(self, targets_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        new_target_dict = {}
        for target_name, target in targets_dict.items():
            target = self.patch_helper(target_name, target)
            new_target_dict[target_name] = target
        return new_target_dict

    def patch_outputs(self, outputs: OUTPUT_FORMAT) -> OUTPUT_FORMAT:
        if isinstance(outputs, Tensor):
            target_name = f'{OUTPUT_PREFIX}0'
            new_outputs = self.patch_helper(target_name, outputs)
        elif isinstance(outputs, (list, tuple)):
            new_outputs = []
            for idx, target in enumerate(outputs):
                target_name = f'{OUTPUT_PREFIX}{idx}'
                new_outputs.append(self.patch_helper(target_name, target))
        elif isinstance(outputs, dict):
            new_outputs = {}
            for output_name, target in outputs.items():
                target_name = f'{OUTPUT_PREFIX}{output_name}'
                new_outputs[output_name] = self.patch_helper(target_name, target)
        else:
            raise TypeError(f'Only support return Tensor/list/dict, but got {type(outputs)}')
        return new_outputs

    def forward(self, *args, **kwargs):
        args, kwargs = self.patch_inputs(*args, **kwargs)

        params_dict = {}
        params_dict.update({k: v.target for k, v in self.pruning_target_spaces.items() if v.type is TargetType.PARAMETER})
        params_dict.update({k: v.target for k, v in self.quantization_target_spaces.items() if v.type is TargetType.PARAMETER})
        params_dict.update({k: v.target for k, v in self.distillation_target_spaces.items() if v.type is TargetType.PARAMETER})
        params_dict = self.patch_params(params_dict)
        for target_name, patched_param in params_dict.items():
            # NOTE: here using copy_ will cause `backward through the graph a second time` error, don't know why.
            # We want to use copy_ for buffers because in-place modification can be recorded in DP, or it will be lost.
            # Here we use setattr to workaround because we don't need to record the buffer value for these module fake targets.

            # module_param: Tensor = getattr(self.module, target_name)
            # module_param.copy_(patched_param)
            setattr(self.module, target_name, patched_param)

        outputs = self.module_forward(*args, **kwargs)
        outputs = self.patch_outputs(outputs)
        return outputs


def register_wrappers(model: torch.nn.Module, config_list: List[Dict[str, Any]],
                      mode: Literal['pruning', 'quantization', 'distillation'],
                      existed_wrappers: Dict[str, ModuleWrapper] | None = None,
                      ) -> Tuple[Dict[str, ModuleWrapper], Dict[str, Dict[str, TargetSpace]]]:
    assert mode in ['pruning', 'quantization', 'distillation']
    configured_target_spaces = {}
    existed_wrappers = existed_wrappers if existed_wrappers else {}
    module_wrappers = {k: v for k, v in existed_wrappers.items()}
    for config in config_list:
        modules, public_config = select_modules_by_config(model, config)
        for module_name, module in modules.items():
            if module_name in module_wrappers:
                wrapper = module_wrappers[module_name]
                wrapper.unfreeze()
                target_spaces = wrapper.extend_target_spaces(public_config, mode)
            else:
                wrapper = ModuleWrapper(module, module_name, {mode: public_config})
                module_wrappers[module_name] = wrapper
                if mode == 'pruning':
                    target_spaces = {k: v for k, v in wrapper.pruning_target_spaces.items()}
                elif mode == 'quantization':
                    target_spaces = {k: v for k, v in wrapper.quantization_target_spaces.items()}
                else:
                    target_spaces = {k: v for k, v in wrapper.distillation_target_spaces.items()}
            configured_target_spaces[module_name] = target_spaces

    return module_wrappers, configured_target_spaces


def track_target_shape(wrapper: ModuleWrapper, target_name: str, target: Tensor):
    """
    Track the input/output target shape and save the shape information to ``TargetSpace.shape``.
    """
    if not isinstance(target, Tensor):
        return
    if target_name in wrapper.quantization_target_spaces:
        if wrapper.quantization_target_spaces[target_name].type is not TargetType.PARAMETER:
            wrapper.quantization_target_spaces[target_name].shape = [_ for _ in target.shape]
    if target_name in wrapper.pruning_target_spaces:
        if wrapper.pruning_target_spaces[target_name].type is not TargetType.PARAMETER:
            wrapper.pruning_target_spaces[target_name].shape = [_ for _ in target.shape]
    if target_name in wrapper.distillation_target_spaces:
        if wrapper.distillation_target_spaces[target_name].type is not TargetType.PARAMETER:
            wrapper.distillation_target_spaces[target_name].shape = [_ for _ in target.shape]
