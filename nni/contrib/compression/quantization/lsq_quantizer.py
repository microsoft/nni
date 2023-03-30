# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import List, Dict, overload

import torch
from torch import Tensor

from ..base.compressor import Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING


class LsqQuantizer(Quantizer):
    __doc__ = r'''
    LsqQuantizer, as defined in: `LEARNED STEP SIZE QUANTIZATION <https://arxiv.org/pdf/1902.08153.pdf>`__,
    authors Steven K. Esser and Jeffrey L. McKinstry provide an algorithm to train the scales with gradients.

    ..

        The authors introduce a novel means to estimate and scale the task loss gradient at each weight and activation
        layer's quantizer step size, such that it can be learned in conjunction with other network parameters.

    Parameters
    ----------
    model
        Model to be quantized.
    config_list
        A list of dict, each dict configure which module need to be quantized, and how to quantize.
        Please refer :doc:`Compression Config Specification </compression/compression_config_list>` for more information.
    evaluator
        TODO: {evaluator_docstring}

    Examples
    --------
        >>> from nni.contrib.compression.quantization import LsqQuantizer
        >>> from nni.contrib.compression.utils import TorchEvaluator
        >>> model = ...
        >>> optimizer = ...
        >>> max_steps, max_epochs = ..., ...
        >>> evaluator = TorchEvaluator(train, optimizer, training_step)
        >>> quantizer = LsqQuantizer(model, configure_list, evaluator)
        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)
    '''.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, \
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.is_init = False

        self.register_scale()
        self.register_lsq_apply_method()
        self.register_track_func()

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.init_scale)

    def init_scale(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if self.is_init or target_name not in wrapper.quantization_target_spaces:
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        init_target = target.data.detach().abs().mean() * 2 / (target_space.qmax ** 0.5)
        target_space.scale.data = init_target # type: ignore
        target_space.zero_point = torch.tensor(0.0).to(target.device)

    def register_lsq_apply_method(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                target_space.apply_method = "lsq_clamp_round"

    def register_scale(self):
        for module_name, ts in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            for target_name, _ in ts.items():
                if hasattr(wrapper, f"{target_name}_scale"):
                    delattr(wrapper, f"{target_name}_scale")
                try:
                    device = next(wrapper.parameters()).device
                except StopIteration:
                    try:
                        device = next(wrapper.buffers()).device
                    except StopIteration:
                        # NOTE: this will have risk in model parallel
                        device = next(self.bound_model.parameters()).device
                param = torch.nn.Parameter(torch.Tensor([1.0]).to(device))
                wrapper.register_parameter(f"{target_name}_scale", param)

    def patch_optimizer_param_group(self):
        module_name_param_dict = super().patch_optimizer_param_group()
        for module_name, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if module_name not in module_name_param_dict:
                    module_name_param_dict[module_name] = []
                module_name_param_dict[module_name].append(target_space.scale)

        return module_name_param_dict

    def register_trigger(self, evaluator: Evaluator):
        def optimizer_task():
            self.is_init = True

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        module_name_param_dict = self.patch_optimizer_param_group()
        if len(module_name_param_dict) > 0:
            evaluator.patch_optim_param_group(module_name_param_dict)
        self.register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        pass


