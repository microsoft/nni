# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Tuple, List, Optional, Union
import torch

from nni.compression.pytorch.compressor import Compressor
from nni.algorithms.compression.pytorch.pruning.one_shot import LevelPruner, L1FilterPruner, L2FilterPruner
from nni.algorithms.compression.pytorch.quantization.quantizers import NaiveQuantizer, QAT_Quantizer

_logger = logging.getLogger(__name__)


PRUNER_DICT = {
    'level': LevelPruner,
    'l1': L1FilterPruner,
    'l2': L2FilterPruner
}

QUANTIZER_DICT = {
    'naive': NaiveQuantizer,
    'qat': QAT_Quantizer
}

CONCFIG_LIST_TYPE = List[Tuple[str, dict, list]]


class Trainer:
    def __init__(self):
        pass

    def finetune(self):
        pass


class MultiCompressor:
    def __init__(self, model, mixed_config_list: List[dict], optimizer=None, trainer=None):
        self.bound_model = model
        self.pruner_config_list, self.quantizer_config_list = self._convert_config_list(mixed_config_list)
        self.optimizer = optimizer
        self.trainer = trainer

        self.pruners = []
        self.quantizers = []

    def _convert_config_list(self, mixed_config_list: List[dict]) -> Tuple[CONCFIG_LIST_TYPE, CONCFIG_LIST_TYPE]:
        pruner_config_list = []
        quantizer_config_list = []
        for config in mixed_config_list:
            if 'pruner' in config:
                pruner = config.get('pruner')
                pruner_config_list.append((pruner['type'], pruner['args'], config.get('config_list')))
            elif 'quantizer' in config:
                quantizer = config.get('quantizer')
                quantizer_config_list.append((quantizer['type'], quantizer['args'], config.get('config_list')))
        return pruner_config_list, quantizer_config_list

    def compress(self):
        for pruner_name, pruner_args, config_list in self.pruner_config_list:
            pruner = PRUNER_DICT[pruner_name](self.bound_model, config_list, self.optimizer, **pruner_args)
            self.pruners.append(pruner)
            self.bound_model = pruner.compress()
        if self.trainer and len(self.pruner_config_list) > 0:
            self.trainer.finetune(self.bound_model, self.optimizer, self)
        for quantizer_name, quantizer_args, config_list in self.quantizer_config_list:
            quantizer = QUANTIZER_DICT[quantizer_name](self.bound_model, config_list, self.optimizer, **quantizer_args)
            self.quantizers.append(quantizer)
            self.bound_model = quantizer.compress()
        if self.trainer and len(self.quantizer_config_list) > 0:
            self.trainer.finetune(self.bound_model, self.optimizer, self)
        return self.bound_model

    def export_model(self, model_path: str, mask_path: str = None, calibration_path: str = None, onnx_path: str = None,
                     input_shape: Optional[Union[List, Tuple]] = None, device: torch.device = None):
        assert model_path is not None, 'model_path must be specified'
        mask_dict = {}
        calibration_config = {}

        for pruner in self.pruners:
            pruner._unwrap_model()
            for wrapper in pruner.get_modules_wrapper():
                weight_mask = wrapper.weight_mask
                bias_mask = wrapper.bias_mask
                if weight_mask is not None:
                    mask_sum = weight_mask.sum().item()
                    mask_num = weight_mask.numel()
                    _logger.info('Layer: %s  Sparsity: %.4f', wrapper.name, 1 - mask_sum / mask_num)
                    wrapper.module.weight.data = wrapper.module.weight.data.mul(weight_mask)
                if bias_mask is not None:
                    wrapper.module.bias.data = wrapper.module.bias.data.mul(bias_mask)
                # save mask to dict
                mask_dict[wrapper.name] = {"weight": weight_mask, "bias": bias_mask}

        for quantizer in self.quantizers:
            quantizer._unwrap_model()
            for name, module in quantizer.bound_model.named_modules():
                if hasattr(module, 'weight_bit') or hasattr(module, 'activation_bit'):
                    calibration_config[name] = {}
                if hasattr(module, 'weight_bit'):
                    calibration_config[name]['weight_bit'] = int(module.weight_bit)
                    calibration_config[name]['tracked_min_input'] = float(module.tracked_min_input)
                    calibration_config[name]['tracked_max_input'] = float(module.tracked_max_input)
                if hasattr(module, 'activation_bit'):
                    calibration_config[name]['activation_bit'] = int(module.activation_bit)
                    calibration_config[name]['tracked_min_activation'] = float(module.tracked_min_activation)
                    calibration_config[name]['tracked_max_activation'] = float(module.tracked_max_activation)
                quantizer._del_simulated_attr(module)

        torch.save(self.bound_model.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)
        if mask_path is not None:
            torch.save(mask_dict, mask_path)
            _logger.info('Mask dict saved to %s', mask_path)
        if calibration_path is not None:
            torch.save(calibration_config, calibration_path)
            _logger.info('Calibration config saved to %s', calibration_path)
        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            if device is None:
                device = torch.device('cpu')
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self.bound_model, input_data.to(device), onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)

        for pruner in self.pruners:
            pruner._wrap_model()

        for quantizer in self.quantizers:
            quantizer._wrap_model()


if __name__ == '__main__':
    pass
