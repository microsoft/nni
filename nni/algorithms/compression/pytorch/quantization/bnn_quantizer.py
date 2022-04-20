# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from schema import Schema, And, Or, Optional
from nni.compression.pytorch.utils.config_validation import QuantizerSchema
from nni.compression.pytorch.compressor import Quantizer, QuantGrad
from nni.compression.pytorch.quantization.literal import QuantType
from nni.compression.pytorch.quantization.utils import get_bits_length


logger = logging.getLogger(__name__)


class ClipGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax):
        if quant_type == QuantType.OUTPUT:
            grad_output[torch.abs(tensor) > 1] = 0
        return grad_output


class BNNQuantizer(Quantizer):
    """Binarized Neural Networks, as defined in:
    Binarized Neural Networks: Training Deep Neural Networks with Weights and Outputs Constrained to +1 or -1
    (https://arxiv.org/abs/1602.02830)
    """

    def __init__(self, model, config_list, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer), "unrecognized optimizer type"
        super().__init__(model, config_list, optimizer)
        device = next(model.parameters()).device
        self.quant_grad = ClipGrad.apply
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
            Optional('quant_types'): Schema([lambda x: x in ['weight', 'output']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('weight'): And(int, lambda n: 0 < n < 32),
                Optional('output'): And(int, lambda n: 0 < n < 32),
            })),
            Optional('op_types'): [str],
            Optional('op_names'): [str],
            Optional('exclude'): bool
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = wrapper.module.weight
        weight = torch.sign(weight)
        # remove zeros
        weight[weight == 0] = 1
        wrapper.module.weight = weight
        wrapper.module.weight_bits = torch.Tensor([1.0])
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        out = torch.sign(output)
        # remove zeros
        out[out == 0] = 1
        return out

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
