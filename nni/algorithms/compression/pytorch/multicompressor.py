# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
import logging
from typing import Tuple, List, Optional, Union
import torch

from nni.compression.pytorch.compressor import Compressor
from nni.compression.pytorch.speedup import ModelSpeedup
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

        self.model_path = None
        self.mask_path = None
        self.calibration_path = None
        self.onnx_path = None
        self.input_shape = None
        self.device = None

    def set_config(self, model_path: str, mask_path: str = None, calibration_path: str = None, onnx_path: str = None,
                   input_shape: Optional[Union[List, Tuple]] = None, device: torch.device = None):
        self.model_path = model_path
        self.mask_path = mask_path
        self.calibration_path = calibration_path
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.device = device

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
            print('Use {} pruner, start pruning...'.format(pruner_name))
            self.bound_model = pruner.compress()

        if len(self.pruner_config_list) > 0:
            self._export_pruned_model()
            dummy_input = torch.randn(self.input_shape).to(self.device)
            model_sp = ModelSpeedup(self.bound_model, dummy_input, self.mask_path)
            model_sp.speedup_model()

            for pruner in self.pruners:
                pruner._unwrap_model()

            if self.trainer:
                saved_path = self.trainer.finetune(self.bound_model, self.optimizer, self)
                if saved_path is not None:
                    self.bound_model = torch.load(saved_path)

        for quantizer_name, quantizer_args, config_list in self.quantizer_config_list:
            quantizer = QUANTIZER_DICT[quantizer_name](self.bound_model, config_list, self.optimizer, **quantizer_args)
            self.quantizers.append(quantizer)
            self.bound_model = quantizer.compress()

        self.bound_model.to(self.device)

        if self.trainer and len(self.quantizer_config_list) > 0:
            saved_path = self.trainer.finetune(self.bound_model, self.optimizer, self)
            if saved_path is not None:
                self.bound_model = torch.load(saved_path)
            calibration_config = self._export_quantized_model()

        return self.bound_model

    def save_bound_model(self, path: str):
        torch.save(self.bound_model, path)

    def load_bound_model(self, path: str):
        self.bound_model = torch.load(path)

    def _export_pruned_model(self):
        assert self.model_path is not None, 'model_path must be specified'
        mask_dict = {}

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

        torch.save(self.bound_model.state_dict(), self.model_path)
        _logger.info('Model state_dict saved to %s', self.model_path)
        if self.mask_path is not None:
            torch.save(mask_dict, self.mask_path)
            _logger.info('Mask dict saved to %s', self.mask_path)
        if self.onnx_path is not None:
            assert self.input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            if self.device is None:
                self.device = torch.device('cpu')
            input_data = torch.Tensor(*self.input_shape)
            torch.onnx.export(self.bound_model, input_data.to(self.device), self.onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, self.onnx_path)

        for pruner in self.pruners:
            pruner._wrap_model()

        return mask_dict

    def _export_quantized_model(self):
        assert self.model_path is not None, 'model_path must be specified'
        calibration_config = {}

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

        torch.save(self.bound_model.state_dict(), self.model_path)
        _logger.info('Model state_dict saved to %s', self.model_path)
        if self.calibration_path is not None:
            torch.save(calibration_config, self.calibration_path)
            _logger.info('Calibration config saved to %s', self.calibration_path)
        if self.onnx_path is not None:
            assert self.input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            if self.device is None:
                self.device = torch.device('cpu')
            input_data = torch.Tensor(*self.input_shape)
            torch.onnx.export(self.bound_model, input_data.to(self.device), self.onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, self.onnx_path)

        for quantizer in self.quantizers:
            quantizer._wrap_model()

        return calibration_config

    # def export_model(self, model_path: str, mask_path: str = None, calibration_path: str = None, onnx_path: str = None,
    #                  input_shape: Optional[Union[List, Tuple]] = None, device: torch.device = None):
    #     assert model_path is not None, 'model_path must be specified'
    #     mask_dict = {}
    #     calibration_config = {}

    #     for pruner in self.pruners:
    #         pruner._unwrap_model()
    #         for wrapper in pruner.get_modules_wrapper():
    #             weight_mask = wrapper.weight_mask
    #             bias_mask = wrapper.bias_mask
    #             if weight_mask is not None:
    #                 mask_sum = weight_mask.sum().item()
    #                 mask_num = weight_mask.numel()
    #                 _logger.info('Layer: %s  Sparsity: %.4f', wrapper.name, 1 - mask_sum / mask_num)
    #                 wrapper.module.weight.data = wrapper.module.weight.data.mul(weight_mask)
    #             if bias_mask is not None:
    #                 wrapper.module.bias.data = wrapper.module.bias.data.mul(bias_mask)
    #             # save mask to dict
    #             mask_dict[wrapper.name] = {"weight": weight_mask, "bias": bias_mask}

    #     for quantizer in self.quantizers:
    #         quantizer._unwrap_model()
    #         for name, module in quantizer.bound_model.named_modules():
    #             if hasattr(module, 'weight_bit') or hasattr(module, 'activation_bit'):
    #                 calibration_config[name] = {}
    #             if hasattr(module, 'weight_bit'):
    #                 calibration_config[name]['weight_bit'] = int(module.weight_bit)
    #                 calibration_config[name]['tracked_min_input'] = float(module.tracked_min_input)
    #                 calibration_config[name]['tracked_max_input'] = float(module.tracked_max_input)
    #             if hasattr(module, 'activation_bit'):
    #                 calibration_config[name]['activation_bit'] = int(module.activation_bit)
    #                 calibration_config[name]['tracked_min_activation'] = float(module.tracked_min_activation)
    #                 calibration_config[name]['tracked_max_activation'] = float(module.tracked_max_activation)
    #             quantizer._del_simulated_attr(module)

    #     torch.save(self.bound_model.state_dict(), model_path)
    #     _logger.info('Model state_dict saved to %s', model_path)
    #     if mask_path is not None:
    #         torch.save(mask_dict, mask_path)
    #         _logger.info('Mask dict saved to %s', mask_path)
    #     if calibration_path is not None:
    #         torch.save(calibration_config, calibration_path)
    #         _logger.info('Calibration config saved to %s', calibration_path)
    #     if onnx_path is not None:
    #         assert input_shape is not None, 'input_shape must be specified to export onnx model'
    #         # input info needed
    #         if device is None:
    #             device = torch.device('cpu')
    #         input_data = torch.Tensor(*input_shape)
    #         torch.onnx.export(self.bound_model, input_data.to(device), onnx_path)
    #         _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)

    #     for pruner in self.pruners:
    #         pruner._wrap_model()

    #     for quantizer in self.quantizers:
    #         quantizer._wrap_model()


if __name__ == '__main__':
    pass
