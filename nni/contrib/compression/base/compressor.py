# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import logging
from typing import Any, Dict, List, Literal, Callable

import torch
from torch.utils._pytree import tree_map

from .config import trans_legacy_config_list, default_config_schema
from .target_space import TargetSpace, TargetType, DistillationTargetSpace, PruningTargetSpace, QuantizationTargetSpace
from .wrapper import ModuleWrapper, register_wrappers
from ..utils import Evaluator, Scaling, _EVALUATOR_DOCSTRING

_logger = logging.getLogger(__name__)


_PRUNING_TARGET_SPACES = Dict[str, Dict[str, PruningTargetSpace]]
_QUANTIZATION_TARGET_SPACES = Dict[str, Dict[str, QuantizationTargetSpace]]
_DISTILLATION_TARGET_SPACES = Dict[str, Dict[str, DistillationTargetSpace]]


class Compressor:
    __doc__ = r"""Compressor base class.

    Parameters
    ----------
    model
        Model to be compressed.
    config_list
        Config list. Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.
    evaluator
        {evaluator_docstring}
    existed_wrappers
        Use by class method ``from_compressor`` to inherit another compressor's wrapper.
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    mode: Literal['pruning', 'quantization', 'distillation']

    def __init__(self, model: torch.nn.Module, config_list: List[Dict],
                 evaluator: Evaluator | None = None, existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        assert self.mode in ['pruning', 'quantization', 'distillation']
        self.bound_model = model
        self.config_list = trans_legacy_config_list(deepcopy(config_list))
        self._validate_config()
        self.evaluator = evaluator
        if self.evaluator is not None:
            assert isinstance(evaluator, Evaluator)
            if not evaluator._initialization_complete:
                evaluator._init_optimizer_helpers(self.bound_model)

        self._is_wrapped = False
        self._module_wrappers, self._target_spaces = register_wrappers(self.bound_model, self.config_list, self.mode, existed_wrappers)
        self.wrap_model()

        self.fused_compressors = [self]

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], *args, evaluator: Evaluator | None = None, **kwargs):
        """
        Inherited from the compressor exsited wrappers and evaluator to initialize a new compressor.

        Parameters
        ----------
        compressor
            The bound model, wrappers and evaluator in this compressor will be inherited to create a new compressor.
        new_config_list
            The config_list used to config the new compressor.
        evaluator
            Key-word only parameter. If the inherited compressor doesn't have an evaluator, then this evaluator will be used to
            initialize the new compressor. If the inherited compressor already has an evaluator, this parameter will be ignored.
        args
            Positional arguments. Will be directly passed to the ``__init__`` function.
        kwargs
            Keyword arguments. Will be directly passed to the ``__init__`` function.
        """
        if compressor._is_wrapped:
            compressor.unwrap_model()
        model = compressor.bound_model
        existed_wrappers = compressor._module_wrappers
        if compressor.evaluator is not None and evaluator is not None:
            _logger.warning('compessor already has evaluator, the new evaluator passed to this function will be ignored.')
        evaluator = compressor.evaluator if compressor.evaluator else evaluator

        new_compressor = cls(model=model, config_list=new_config_list, evaluator=evaluator, existed_wrappers=existed_wrappers,
                             *args, **kwargs)
        new_compressor.fused_compressors.extend(compressor.fused_compressors)
        return new_compressor

    def _validate_config(self):
        schema = default_config_schema(self.mode)
        for config in self.config_list:
            schema.validate(config)

    def wrap_model(self):
        """
        Traverse all wrappers and execute ModuleWrapper.wrap()
        """
        if self._is_wrapped is True:
            warn_msg = 'The bound model has been wrapped, no need to wrap again.'
            _logger.warning(warn_msg)
        for _, wrapper in self._module_wrappers.items():
            wrapper.wrap()
        self._is_wrapped = True

    def unwrap_model(self):
        """
        Traverse all wrappers and execute ModuleWrapper.unwrap()
        """
        if self._is_wrapped is False:
            warn_msg = 'The bound model is not wrapped, can not unwrap it.'
            _logger.warning(warn_msg)
        for _, wrapper in self._module_wrappers.items():
            wrapper.unwrap()
        self._is_wrapped = False

    def track_forward(self, *args, **kwargs):
        """
        Forward once to track information, such as the wrapped module input/output shape.
        Make sure the input has the same batch size and data distribution with the batch sampled from dataloader.
        The track logic can be found in ``ModuleWrapper._track_info``.

        Parameters
        ----------
        args
            Positional real input to the model.
        kwargs
            Keyword real input to the model.
        """
        assert self._is_wrapped, 'The bound model is not wrapped, can not track information with an unwrapped model.'
        with torch.no_grad():
            model_device = next(iter(self.bound_model.parameters())).device
            args = tree_map(lambda x: x.to(model_device) if isinstance(x, torch.Tensor) else x, args)
            kwargs = tree_map(lambda x: x.to(model_device) if isinstance(x, torch.Tensor) else x, kwargs)
            self.bound_model(*args, **kwargs)

    def _get_param_names_map(self) -> Dict[str, str]:
        """
        Returns
        -------
        Dict[str, str]
            {original_parameter_name: wrapped_parameter_name}.
            i.e., {'model.fc.weight': 'model.fc._nni_wrapper.weight'}.
        """
        param_names_map = {}
        if self._is_wrapped is True:
            for param_name, _ in self.bound_model.named_parameters():
                origin_param_name = ''.join(param_name.split('_nni_wrapper.')) if '_nni_wrapper' in param_name else param_name
                param_names_map[origin_param_name] = param_name
        else:
            raise RuntimeError('Only can get param_names_map when the model is wrapped.')
        return param_names_map

    def _single_compress(self, max_steps: int | None, max_epochs: int | None) -> None:
        raise NotImplementedError()

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        """
        (Experimental) Compressor can register compress logic into training/predict loop by this api before evaluator training.
        It is recommended to decide whether to execute compress according to the current optimization step.
        """
        raise NotImplementedError()

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        """
        (Experimental) Do some postprocess after evaluator training.
        """
        raise NotImplementedError()

    def _fusion_compress(self, max_steps: int | None, max_epochs: int | None) -> None:
        """
        (Experimental) Simultaneous execution of multiple compression logics.
        """
        assert self.evaluator is not None
        assert max_steps is not None or max_epochs is not None
        self.evaluator.bind_model(self.bound_model, self._get_param_names_map())
        for compressor in self.fused_compressors:
            compressor._fuse_preprocess(self.evaluator)
        self.evaluator.train(max_steps, max_epochs)
        for compressor in self.fused_compressors:
            compressor._fuse_postprocess(self.evaluator)
        self.evaluator.unbind_model()

    def compress(self, max_steps: int | None, max_epochs: int | None):
        if len(self.fused_compressors) <= 1:
            self._single_compress(max_steps, max_epochs)
        else:
            self._fusion_compress(max_steps, max_epochs)
        return self.bound_model


class Pruner(Compressor):
    mode = 'pruning'

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model=model, config_list=config_list, evaluator=evaluator, existed_wrappers=existed_wrappers)
        self._target_spaces: _PRUNING_TARGET_SPACES
        self._register_scalers()

    def _register_scalers(self):
        # scalers are used to support different sparse/quant granularity
        register_scalers(self._target_spaces, self._set_default_sparse_granularity)  # type: ignore

    def _set_default_sparse_granularity(self, target_space: PruningTargetSpace) -> List[int] | str | None:
        if target_space.type is TargetType.PARAMETER:
            return 'out_channel'
        if target_space.type in [TargetType.INPUT, TargetType.OUTPUT]:
            return 'per_channel'

    def get_masks(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get all masks registered in the wrappers.

        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]
            The masks' format is {module_name: {target_name: target_masks}}.
            For example, {'layer.0.conv1': {'weight': weight_mask_tensor, 'bias': bias_mask_tensor}}.
        """
        masks = defaultdict(dict)
        for module_name, wrapper in self._module_wrappers.items():
            for target_name, target_space in wrapper.pruning_target_spaces.items():
                masks[module_name][target_name] = target_space.mask.clone() if target_space.mask is not None else None
        return masks

    def update_masks(self, masks: Dict[str, Dict[str, torch.Tensor]]):
        """
        For given masks, update the masks in the wrappers.

        Parameters
        ----------
        masks
            The masks' format should be {module_name: {target_name: target_masks}}.
            If the module is not registered as a wrapper in the model or the target is not registered as a target space in the compressor,
            the related masks updating will be ignored.
        """
        for module_name, target_masks in masks.items():
            if module_name not in self._module_wrappers:
                _logger.warning(f'Wrapper {module_name} is not register in this compressor, the masks will be ignored.')
            wrapper = self._module_wrappers[module_name]
            for target_name, target_mask in target_masks.items():
                target_space = wrapper.pruning_target_spaces.get(target_name, None)
                if target_space is None:
                    _logger.warning(f'Target space `{target_name}` is not in wrapper `{wrapper.name}`, the mask of it will be ignored.')
                    continue
                if target_mask is None:
                    target_space.mask = None
                else:
                    assert target_name in wrapper.pruning_target_spaces, \
                        f'{module_name}.{target_name} is not a pruning target, can not update mask for it.'
                    # NOTE: try to auto get the device of the current module
                    try:
                        device = next(wrapper.parameters()).device
                    except StopIteration:
                        try:
                            device = next(wrapper.buffers()).device
                        except StopIteration:
                            if target_space.mask is not None:
                                device = target_space.mask.device
                            else:
                                # NOTE: this will have risk in model parallel
                                device = next(self.bound_model.parameters()).device
                    target_space.mask = target_mask.to(device)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def _generate_sparsity(self, metrics: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def generate_masks(self) -> Dict[str, Dict[str, torch.Tensor]]:
        data = self._collect_data()
        metrics = self._calculate_metrics(data)
        return self._generate_sparsity(metrics)

    def compress(self, max_steps: int | None, max_epochs: int | None):
        return super().compress(max_steps, max_epochs), self.get_masks()


class Quantizer(Compressor):
    mode = 'quantization'

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model=model, config_list=config_list, evaluator=evaluator, existed_wrappers=existed_wrappers)
        self._target_spaces: _QUANTIZATION_TARGET_SPACES
        self._register_scalers()

    def _register_scalers(self):
        # scalers are used to support different sparse/quant granularity
        register_scalers(self._target_spaces, self._set_default_sparse_granularity)  # type: ignore

    def check_target(self, wrapper: ModuleWrapper, target_name: str) -> bool:
        module_name = wrapper.name
        if module_name not in self._target_spaces:
            return False
        ts = self._target_spaces[module_name]
        if target_name not in ts:
            return False

        return True

    def _set_default_sparse_granularity(self, target_space: PruningTargetSpace) -> List[int] | str | None:
        return None

    def get_calibration_config(self) -> Dict[str, Dict[str, Dict[str, torch.Tensor | Any]]]:
        """
        Get all calibration information in the wrappers.

        Returns
        -------
        Dict[str, Dict[str, Dict[str, torch.Tensor | Any]]]
            The calibration config format is {module_name: {target_name: {info_key: val}}}.
            By default, info_key contains ['scale', 'zero_point', 'quant_dtype', 'quant_scheme', 'quant_bits'] and
            ['tracked_max', 'tracked_min'] if it has.

            NOTE: The internal key of the calibration config for the target is not stable,
            might be adjusted according to the needs of speedup.
        """
        calibration_config = defaultdict(dict)
        for module_name, wrapper in self._module_wrappers.items():
            for target_name, target_space in wrapper.quantization_target_spaces.items():
                calibration_config[module_name][target_name] = {
                    'scale': target_space.scale.cpu() if isinstance(target_space.scale, torch.Tensor) \
                        else target_space.scale,
                    'zero_point': target_space.zero_point.cpu() if isinstance(target_space.zero_point, torch.Tensor) \
                        else target_space.zero_point,
                    'quant_dtype': target_space.quant_dtype if target_space.quant_dtype else 'int8',
                    'quant_scheme': target_space.quant_scheme,
                    'quant_bits': target_space.quant_bits,
                }
                if target_space.tracked_max is not None:
                    calibration_config[module_name][target_name]['tracked_max'] = target_space.tracked_max.cpu()
                elif target_space.zero_point is not None and target_space.scale is not None:
                    tracked_max = target_space.scale * (target_space.qmax - target_space.zero_point)
                    calibration_config[module_name][target_name]['tracked_max'] = tracked_max.cpu()
                if target_space.tracked_min is not None:
                    calibration_config[module_name][target_name]['tracked_min'] = target_space.tracked_min.cpu()
                elif target_space.zero_point is not None and target_space.scale is not None:
                    tracked_min = target_space.scale * (target_space.qmin - target_space.zero_point)
                    calibration_config[module_name][target_name]['tracked_min'] = tracked_min.cpu()

        return calibration_config

    def update_calibration_config(self, calibration_config: Dict[str, Dict[str, Dict[str, torch.Tensor | Any]]]):
        for module_name, target_configs in calibration_config.items():
            for target_name, config in target_configs.items():
                assert module_name in self._module_wrappers and target_name in self._module_wrappers[module_name].quantization_target_spaces
                wrapper = self._module_wrappers[module_name]
                target_space = wrapper.quantization_target_spaces[target_name]
                # NOTE: try to auto get the device of the current module
                try:
                    device = next(wrapper.parameters()).device
                except StopIteration:
                    try:
                        device = next(wrapper.buffers()).device
                    except StopIteration:
                        if target_space.scale is not None:
                            device = target_space.scale.device
                        else:
                            # NOTE: this will have risk in model parallel
                            device = next(self.bound_model.parameters()).device
                config = tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, config)
                target_space.scale = config['scale']
                target_space.zero_point = config['zero_point']
                assert target_space.quant_bits == config['quant_bits']
                assert target_space.quant_dtype == config['quant_dtype']
                assert target_space.quant_scheme == config['quant_scheme']

    def patch_optimizer_param_group(self):
        module_name_param_dict = {}
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            if getattr(wrapper, "is_register_bias", False) and isinstance(wrapper.bias, torch.nn.parameter.Parameter):
                module_name_param_dict[module_name] = [wrapper.bias]

        return module_name_param_dict

    def compress(self, max_steps: int | None, max_epochs: int | None):
        return super().compress(max_steps, max_epochs), self.get_calibration_config()


class Distiller(Compressor):
    mode = 'distillation'

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model=model, config_list=config_list, evaluator=evaluator, existed_wrappers=existed_wrappers)
        self._target_spaces: _DISTILLATION_TARGET_SPACES


def register_scalers(target_spaces: _PRUNING_TARGET_SPACES | _QUANTIZATION_TARGET_SPACES,
                     set_default_granularity: Callable[[TargetSpace], Any]):
    """
    Create and register scaler to the target space according to the granularity of target space.
    Four string type granularity will be treated specially:

        * ``default`` will be treated by function ``set_default_granularity``.
        * ``in_channel`` means compressing on the 0-th dim of the paramter target.
        * ``out_channel`` means compressing on the 1-th dim of the paramter target.
        * ``per_channel`` means compressing on the 1-th dim of the input/output target (assume 0-th dim is batch dim).

    Parameters
    ----------
    target_spaces
        {module_name: {target_name: target_space}}.
    set_default_granularity
        A callable function, given a target space and return a scaler.
    """
    # scalers are used to support different sparse/quant granularity
    for _, ts in target_spaces.items():
        for _, target_space in ts.items():
            if target_space.granularity == 'default':
                target_space.granularity = set_default_granularity(target_space)
            if target_space.granularity is None:
                continue
            if target_space.granularity == 'out_channel':
                assert target_space._target_type is TargetType.PARAMETER
                target_space._scaler = Scaling([1], kernel_padding_mode='back', kernel_padding_val=-1)
            elif target_space.granularity == 'in_channel':
                assert target_space._target_type is TargetType.PARAMETER
                target_space._scaler = Scaling([-1, 1], kernel_padding_mode='back', kernel_padding_val=-1)
            elif target_space.granularity == 'per_channel':
                # NOTE: here assume dim 0 is batch, dim 1 is channel
                assert target_space._target_type in [TargetType.INPUT, TargetType.OUTPUT]
                target_space._scaler = Scaling([-1, 1], kernel_padding_mode='back', kernel_padding_val=-1)
            else:
                kernel_padding_mode = None
                kernel_padding_val = None
                if all(isinstance(_, int) for _ in target_space.granularity):
                    kernel_size = target_space.granularity
                elif len(target_space.granularity) == 1:
                    kernel_size = target_space.granularity[0]
                elif len(target_space.granularity) == 2:
                    kernel_size, kernel_padding_mode = target_space.granularity[0], target_space.granularity[1]
                else:
                    assert len(target_space.granularity) == 3
                    kernel_size = target_space.granularity[0]
                    kernel_padding_mode = target_space.granularity[1]
                    kernel_padding_val = target_space.granularity[2]
                kernel_padding_mode = kernel_padding_mode if kernel_padding_mode else 'front'
                kernel_padding_val = kernel_padding_val if kernel_padding_val else 1
                target_space._scaler = Scaling(kernel_size, kernel_padding_mode, kernel_padding_val)  # type: ignore
