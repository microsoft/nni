# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from schema import Schema, And, Or, Optional
from nni.compression.pytorch.utils.config_validation import QuantizerSchema
from nni.compression.pytorch.compressor import Quantizer
from nni.compression.pytorch.quantization.utils import get_bits_length


logger = logging.getLogger(__name__)


class DoReFaQuantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """

    def __init__(self, model, config_list, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer), "unrecognized optimizer type"
        super().__init__(model, config_list, optimizer)
        device = next(model.parameters()).device
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            if "weight" in config.get("quant_types", []):
                weight_bits = get_bits_length(config, 'weight')
                layer.module.register_buffer('weight_bits', torch.Tensor([int(weight_bits)]))
        self.bound_model.to(device)

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'weight_bits']
        for attr in del_attr_list:
            if hasattr(module, attr):
                delattr(module, attr)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list of dict
            List of configurations
        """
        schema = QuantizerSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('weight'): And(int, lambda n: 0 < n < 32)
            })),
            Optional('op_types'): [str],
            Optional('op_names'): [str],
            Optional('exclude'): bool
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = wrapper.module.weight
        weight_bits = int(wrapper.module.weight_bits)
        weight = weight.tanh()
        weight = weight / (2 * weight.abs().max()) + 0.5
        weight = self.quantize(weight, weight_bits)
        weight = 2 * weight - 1
        wrapper.module.weight = weight
        # wrapper.module.weight.data = weight
        return weight

    def quantize(self, input_ri, q_bits):
        scale = pow(2, q_bits) - 1
        output = torch.round(input_ri * scale) / scale
        return output

    def export_model(self, model_path, calibration_path=None, onnx_path=None, input_shape=None, device=None):
        """
        Export quantized model weights and calibration parameters(optional)

        Parameters
        ----------
        model_path : str
            path to save quantized model weight
        calibration_path : str
            (optional) path to save quantize parameters after calibration
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        device : torch.device
            device of the model, used to place the dummy input tensor for exporting onnx file.
            the tensor is placed on cpu if ```device``` is None

        Returns
        -------
        Dict
        """
        assert model_path is not None, 'model_path must be specified'
        self._unwrap_model()
        calibration_config = {}

        for name, module in self.bound_model.named_modules():
            if hasattr(module, 'weight_bits'):
                calibration_config[name] = {}
                calibration_config[name]['weight_bits'] = int(module.weight_bits)
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path, input_shape, device)

        return calibration_config
