# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import defaultdict
import torch
from schema import Schema, And, Or, Optional
from nni.compression.pytorch.utils.config_validation import QuantizerSchema
from nni.compression.pytorch.compressor import Quantizer, QuantForward
from nni.compression.pytorch.quantization.observers import default_weight_observer, default_histogram_observer


logger = logging.getLogger(__name__)


class ObserverQuantizer(Quantizer):
    """This quantizer uses observers to record weight/output statistics to get quantization information.
    The whole process can be divided into three steps:

    1. It will register observers to the place where quantization would happen (just like registering hooks).
    2. The observers would record tensors' statistics during calibration.
    3. Scale & zero point would be obtained after calibration.

    Note that the observer type, tensor dtype and quantization qscheme are hard coded for now. Their customization
    are under development and will be ready soon.
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        # NOTE: this quantizer is experimental for now. The dtype and qscheme of quantization
        # is hard-coded.
        # TODO:
        # 1. support dtype and qscheme customization through config_list. Current settings:
        #  weight observer     : per_tensor_symmetric, qint8
        #  output observer : per_tensor_affine, quint8, reduce_range=True
        # 2. add more kinds of observers, such as Kullback-Leibler divergence.
        # 3. add batch normalization folding
        assert not model.training, "Currently the observer quantizer only works in evaluation mode."
        self.quant_grad = QuantForward()
        self.device = next(model.parameters()).device
        modules_to_compress = self.get_modules_to_compress()
        all_observers = defaultdict(dict)
        weight_qmin, weight_qmax = -127, 127
        output_qmin, output_qmax = 0, 127  # reduce_range is set to True
        self.compressed = False

        for layer, config in modules_to_compress:
            layer_name = layer.name
            module = layer.module
            if "weight" in config.get("quant_types", []):
                all_observers[layer_name]["weight"] = default_weight_observer()
                setattr(module, "weight_qmax", weight_qmax)
                setattr(module, "weight_qmin", weight_qmin)
            if "input" in config.get("quant_types", []):
                all_observers[layer_name]["input"] = default_histogram_observer()
                setattr(module, "input_qmax", output_qmax)
                setattr(module, "input_qmin", output_qmin)
            if "output" in config.get("quant_types", []):
                all_observers[layer_name]["output"] = default_histogram_observer()
                setattr(module, "output_qmax", output_qmax)
                setattr(module, "output_qmin", output_qmin)
        self.all_observers = all_observers
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

    def record(self, wrapper, quant_type, tensor):
        name = wrapper.name
        observer = self.all_observers[name][quant_type]
        observer(tensor.cpu())

    def calculate_qparams(self, name, quant_type):
        observer = self.all_observers[name][quant_type]
        scale, zero_point = observer.calculate_qparams()
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
        else:
            self.record(wrapper, 'input', inputs)
        return inputs

    def quantize_weight(self, wrapper, **kwargs):
        # If ObserverQuantizer.compress is executed, the weight will be set to
        # the Pseudo-quantized one. So there is no need to quantize it
        if self.compressed:
            return
        weight = wrapper.module.weight
        self.record(wrapper, 'weight', weight)

    def quantize_output(self, output, wrapper, **kwargs):
        if self.compressed:
            module = wrapper.module
            new_output = self._quantize(output,
                                       module.output_scale,
                                       module.output_zero_point,
                                       module.output_qmin,
                                       module.output_qmax)
        else:
            self.record(wrapper, 'output', output)
            new_output = output
        return new_output

    def compress(self):
        """
        Calculate quantization information of each tensor. Note that the inference of
        the compressed model will no longer update the corresponding. Instead, the quantization
        process will be simulated, which is used to test the accuracy of the quantization.
        """
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            module = layer.module
            if "weight" in config.get("quant_types", []):
                scale, zero_point = self.calculate_qparams(layer.name, 'weight')
                module.register_buffer('weight_scale', scale.to(self.device))
                module.register_buffer('weight_zero_point', zero_point.to(self.device))
                weight = module.weight
                quantized_weight = self._quantize(weight,
                                            module.weight_scale,
                                            module.weight_zero_point,
                                            module.weight_qmin,
                                            module.weight_qmax)
                delattr(module, 'weight')
                module.register_buffer('weight', quantized_weight)
            if "input" in config.get("quant_types", []):
                scale, zero_point = self.calculate_qparams(layer.name, 'input')
                module.register_buffer('input_scale', scale.to(self.device))
                module.register_buffer('input_zero_point', zero_point.to(self.device))
            if "output" in config.get("quant_types", []):
                scale, zero_point = self.calculate_qparams(layer.name, 'output')
                module.register_buffer('output_scale', scale.to(self.device))
                module.register_buffer('output_zero_point', zero_point.to(self.device))
        self.compressed = True
        super().compress()

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
