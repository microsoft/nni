# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
from curses import wrapper
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Tuple
import torch
import torch.nn as nn
from schema import Schema, And, Or, Optional
from nni.compression.pytorch.utils.config_validation import QuantizerSchema
from nni.compression.pytorch.compressor import Quantizer, QuantForward

from ...v2.pytorch.utils import Evaluator, ForwardHook
from ...v2.pytorch.pruning.tools import EvaluatorPredictingDataCollector

logger = logging.getLogger(__name__)

__all__ = ['PtqQuantizer']

def max_min_collector(buffer: list,
                      collect_input: bool,
                      collect_output: bool,
                      mode: str = None) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
    assert len(buffer) == 0, 'Buffer pass to activation pruner collector is not empty.'
    # the length of the buffer here is always 4.
    # buffer[0] buffer[2] records the min value of input and output
    # buffer[1] buffer[3] records the max value of input and output
    buffer.extend([torch.tensor(float('inf')), torch.tensor(float('-inf')),
                   torch.tensor(float('inf')), torch.tensor(float('-inf'))])

    def collect_maxmin(_module: nn.Module,
                       _input: Tuple[Any],
                       output: Any):
        # TODO: support multiple inputs and outputs
        assert len(_input) == 1 and isinstance(output, torch.Tensor)
        if collect_input and _input[0].numel() != 0:
            min_val, max_val = torch.aminmax(_input[0])
            buffer[0] = torch.min(min_val, buffer[0])
            buffer[1] = torch.max(max_val, buffer[1])
        if collect_output and output.numel() != 0:
            min_val, max_val = torch.aminmax(output)
            buffer[2] = torch.min(min_val, buffer[2])
            buffer[3] = torch.max(max_val, buffer[3])
    return collect_maxmin

def histogram_collector(buffer: list,
                      collect_input: bool,
                      collect_output: bool,
                      mode: str) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
    ...

class PtqQuantizer(Quantizer):
    """
    PTQ (Post Training Quantization)
    Users should implement predicting_func in TorchEvaluator
    """

    def __init__(self, model: nn.Module, config_list: list, evaluator: Evaluator):
        # TODO: canonicalize config list
        super().__init__(model, config_list, None)
        assert not model.training, "Currently the observer quantizer only works in evaluation mode."
        self.quant_grad = QuantForward()
        self.device = next(model.parameters()).device
        self.evaluator = evaluator
        self.collector_hooks = {}
        #weight_qmin, weight_qmax = -127, 127
        #output_qmin, output_qmax = 0, 127  # reduce_range is set to True
        self.compressed = False
        self.bound_model.to(self.device)

    def validate_config(self, model, config_list):
        schema = QuantizerSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight', 'output', 'input']]),
            Optional('quant_bits'): Or(And(int, lambda n: n == 8), Schema({
                Optional('weight'): And(int, lambda n: n == 8),
                Optional('output'): And(int, lambda n: n == 8),
                Optional('input'): And(int, lambda n: n == 8),
            })),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def _prepare_data_collectors(self):
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            layer_name = layer.name
            module = layer.module
            collect_input = collect_output = False
            if "weight" in config.get("quant_types", []):
                # FIXME: handle bits==1, and unsigned int
                bits = config.get('quant_bits')['weight']
                setattr(module, 'weight_qmax', 2**(bits-1)-1)
                setattr(module, 'weight_qmin', -2**(bits-1)+1)
            if "input" in config.get("quant_types", []):
                bits = config.get('quant_bits')['input']
                setattr(module, 'input_qmax', 2**(bits-1)-1)
                setattr(module, 'input_qmin', -2**(bits-1)+1)
                collect_input = True
            if "output" in config.get("quant_types", []):
                bits = config.get('quant_bits')['output']
                setattr(module, "output_qmax", 2**(bits-1)-1)
                setattr(module, "output_qmin", -2**(bits-1)+1)
                collect_output = True
            if collect_input or collect_output:
                self.collector_hooks[layer_name] = {
                    'input_output': ForwardHook(module, layer_name,
                        functools.partial(max_min_collector, collect_input=collect_input, collect_output=collect_output))
                    }
        data_collector = EvaluatorPredictingDataCollector(self, self.evaluator, hooks=self.collector_hooks)
        return data_collector

    def compress(self) -> Tuple[nn.Module, Dict]:
        """
        Calculate quantization information of each tensor. The quantization
        process is simulated.
        """
        assert self.bound_model is not None and self.config_list is not None
        assert self.evaluator is not None
        self.evaluator.bind_only_model(self.bound_model)
        data_collector = self._prepare_data_collectors()
        data = data_collector.collect()
        print('collected data: ', data)

        quant_result_conf = {}
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            module = layer.module
            quant_result_conf[layer.name] = {}
            if "weight" in config.get("quant_types", []):
                # quantize weight directly with weight_qmin and weight_qmax
                vmin, vmax = torch.aminmax(module.weight)
                scale, zero_point = self._calculate_qparams(vmin, vmax, module.weight_qmin, module.weight_qmax)
                module.register_buffer('weight_scale', scale.to(self.device))
                module.register_buffer('weight_zero_point', zero_point.to(self.device))
                quantized_weight = self._quantize(module.weight,
                                                  module.weight_scale,
                                                  module.weight_zero_point,
                                                  module.weight_qmin,
                                                  module.weight_qmax)
                # TODO: do we need to keep the old weight???
                delattr(module, 'weight')
                module.register_buffer('weight', quantized_weight)
                quant_result_conf[layer.name]['weight'] = {'qmin': module.weight_qmin, 'qmax': module.weight_qmax,
                                                           'scale': scale, 'zero_point': zero_point}
            if "input" in config.get("quant_types", []):
                vmin, vmax = data[layer.name]['input_output'][0], data[layer.name]['input_output'][1]
                scale, zero_point = self._calculate_qparams(vmin, vmax, module.input_qmin, module.input_qmax)
                module.register_buffer('input_scale', scale.to(self.device))
                module.register_buffer('input_zero_point', zero_point.to(self.device))
                quant_result_conf[layer.name]['input'] = {'qmin': module.input_qmin, 'qmax': module.input_qmax,
                                                          'scale': scale, 'zero_point': zero_point}
            if "output" in config.get("quant_types", []):
                vmin, vmax = data[layer.name]['input_output'][2], data[layer.name]['input_output'][3]
                scale, zero_point = self._calculate_qparams(vmin, vmax, module.output_qmin, module.output_qmax)
                module.register_buffer('output_scale', scale.to(self.device))
                module.register_buffer('output_zero_point', zero_point.to(self.device))
                quant_result_conf[layer.name]['output'] = {'qmin': module.output_qmin, 'qmax': module.output_qmax,
                                                           'scale': scale, 'zero_point': zero_point}
        self.compressed = True
        # for removing hooks
        self.evaluator.unbind_only_model()
        print('quant resulting config: ', quant_result_conf)
        return self.bound_model, quant_result_conf

    def _calculate_qparams(self, vmin, vmax, qmin, qmax):
        # FIXME: check min max is valid
        vmin_neg = torch.min(vmin, torch.zeros_like(vmin))
        vmax_pos = torch.max(vmax, torch.zeros_like(vmax))
        device = vmin_neg.device
        scale = torch.ones(vmin_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(vmin_neg.size(), dtype=torch.int64, device=device)
        vmax_pos = torch.max(-vmin_neg, vmax_pos)
        scale = vmax_pos / (float(qmax - qmin) / 2)
        scale = torch.max(scale, torch.tensor([torch.finfo(torch.float32).eps]))
        return scale, zero_point

    def _quantize(self, x, scale, zero_point, qmin, qmax):
        x = x / scale + zero_point
        x = torch.clamp(x, qmin, qmax)
        x = torch.round(x)
        x = (x - zero_point) * scale
        return x

    def quantize_input(self, inputs, wrapper, **kwargs):
        if self.compressed:
            module = wrapper.module
            inputs = self._quantize(inputs,
                                    module.input_scale,
                                    module.input_zero_point,
                                    module.input_qmin,
                                    module.input_qmax)
        return inputs

    def quantize_weight(self, wrapper, **kwargs):
        # If ObserverQuantizer.compress is executed, the weight will be set to
        # the Pseudo-quantized one. So there is no need to quantize it
        return

    def quantize_output(self, output, wrapper, **kwargs):
        if self.compressed:
            module = wrapper.module
            output = self._quantize(output,
                                    module.output_scale,
                                    module.output_zero_point,
                                    module.output_qmin,
                                    module.output_qmax)
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
            if hasattr(module, 'weight_scale') or hasattr(module, 'input_scale') or hasattr(module, 'output_scale'):
                calibration_config[name] = {}
            if hasattr(module, 'weight_scale'):
                calibration_config[name]['weight_bits'] = 8
                val = float(module.weight_scale * module.weight_qmax)
                calibration_config[name]['tracked_max_weight'] = val
                calibration_config[name]['tracked_min_weight'] = -val
                calibration_config[name]['tracked_qmin_weight'] = -127
                calibration_config[name]['tracked_qmax_weight'] = 127
                weight = module.weight
                quantized_weight = self._quantize(weight,
                                            module.weight_scale,
                                            module.weight_zero_point,
                                            module.weight_qmin,
                                            module.weight_qmax)
                delattr(module, 'weight')
                module.register_parameter('weight', torch.nn.Parameter(quantized_weight))
            # refactor these magic numbers when customizations of dtype and qscheme are ready.
            if hasattr(module, 'input_scale'):
                calibration_config[name]['input_bits'] = 8
                max_input = float(module.input_scale * (module.input_qmax - module.input_zero_point))
                min_input = float(module.input_scale * (module.input_qmin - module.input_zero_point))
                calibration_config[name]['tracked_min_input'] = min_input
                calibration_config[name]['tracked_max_input'] = max_input
                calibration_config[name]['tracked_qmin_input'] = 0
                calibration_config[name]['tracked_qmax_input'] = 127
            if hasattr(module, 'output_scale'):
                calibration_config[name]['output_bits'] = 8
                max_input = float(module.output_scale * (module.output_qmax - module.output_zero_point))
                min_input = float(module.output_scale * (module.output_qmin - module.output_zero_point))
                calibration_config[name]['tracked_min_output'] = min_input
                calibration_config[name]['tracked_max_output'] = max_input
                calibration_config[name]['tracked_qmin_output'] = 0
                calibration_config[name]['tracked_qmax_output'] = 127
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path,
                               input_shape, device)

        return calibration_config

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'steps', 'weight_qmax', 'weight_qmin', 'input_qmax', 'input_qmin',
                         'output_qmax', 'output_qmin', 'weight_scale', 'weight_zero_point', 'input_scale',
                         'input_zero_point', 'output_scale', 'output_zero_point']
        for attr in del_attr_list:
            if hasattr(module, attr):
                delattr(module, attr)
