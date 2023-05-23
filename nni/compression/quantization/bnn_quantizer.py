# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
from typing import List, Dict, overload

import torch
from torch import Tensor

from ..base.compressor import Compressor, Quantizer
from ..base.wrapper import ModuleWrapper
from ..base.target_space import TargetType
from ..utils import Evaluator, _EVALUATOR_DOCSTRING

_logger = logging.getLogger(__name__)


class BNNQuantizer(Quantizer):
    __doc__ = r"""
    BinaryNet Quantization, as defined in:
    `Binarized Neural Networks: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1 <https://arxiv.org/abs/1602.02830>`__,

    ..

        We introduce a method to train Binarized Neural Networks (BNNs) - neural networks with binary weights and activations at run-time.
        At training-time the binary weights and activations are used for computing the parameters gradients. During the forward pass,
        BNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations,
        which is expected to substantially improve power-efficiency.

    Parameters
    ----------
    model
        Model to be quantized.
    config_list
        A list of dict, each dict configure which module need to be quantized, and how to quantize.
        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.
    evaluator
        {evaluator_docstring}

    Examples
    --------
        >>> from nni.compression.quantization import BNNQuantizer
        >>> from nni.compression.utils import TorchEvaluator
        >>> model = ...
        >>> optimizer = ...
        >>> max_steps, max_epochs = ..., ...
        >>> evaluator = TorchEvaluator(train, optimizer, training_step)
        >>> quantizer = BNNQuantizer(model, configure_list, evaluator)
        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, \
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers)

        self.check_validation()
        self.evaluator: Evaluator
        self.is_init = False
        self.register_bnn_apply_method()
        self.register_track_func()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], evaluator: Evaluator | None = None):
        return super().from_compressor(compressor, new_config_list, evaluator=evaluator)

    def check_validation(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.quant_dtype is not None:
                    warn_msg = "BNNQuantizer will only quantize the value to 1 or -1; the quant_dtype value will not work"
                    _logger.warning(warn_msg)
                if target_space._scaler is not None:
                    raise ValueError("BNNQauntizer doesn't support for granularity, please set it to False")

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.init_scale_zp)

    def register_bnn_apply_method(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                target_space.apply_method = 'bnn_clamp_round'

    def init_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if self.is_init or not self.check_target(wrapper, target_name):
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        target_space.zero_point = torch.tensor(0.0).to(target.device)
        target_space.scale = torch.tensor(1.0).to(target.device)

    def register_trigger(self, evaluator: Evaluator):
        def optimizer_task():
            self.is_init = True
            # clip params to (-1,1)
            for _, ts in self._target_spaces.items():
                for _, target_space in ts.items():
                    if target_space.type is TargetType.PARAMETER and \
                        isinstance(target_space.target, torch.nn.parameter.Parameter):
                        target_space.target.data = target_space.target.data.clamp(-1,1)

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
