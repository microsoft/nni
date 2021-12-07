# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from nni.compression.pytorch.compressor import BN_FOLD_TAG, Quantizer, QuantForward
from nni.compression.pytorch.quantization.utils import get_bits_length


logger = logging.getLogger(__name__)


class LsqQuantizer(Quantizer):
    """Quantizer defined in:
       Learned Step Size Quantization (ICLR 2020)
       https://arxiv.org/pdf/1902.08153.pdf
    """

    def __init__(self, model, config_list, optimizer, dummy_input=None):
        """
        Parameters
        ----------
        model : torch.nn.Module
            the model to be quantized
        config_list : list of dict
            list of configurations for quantization
            supported keys for dict:
                - quant_types : list of string
                    type of quantization you want to apply, currently support 'weight', 'input', 'output'
                - quant_bits : int or dict of {str : int}
                    bits length of quantization, key is the quantization type, value is the length, eg. {'weight': 8},
                    when the type is int, all quantization types share same bits length
                - quant_start_step : int
                    disable quantization until model are run by certain number of steps, this allows the network to enter a more stable
                    state where output quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0
                - op_types : list of string
                    types of nn.module you want to apply quantization, eg. 'Conv2d'
                - dummy_input : tuple of tensor
                    inputs to the model, which are used to get the graph of the module. The graph is used to find
                    Conv-Bn patterns. And then the batch normalization folding would be enabled. If dummy_input is not
                    given, the batch normalization folding would be disabled.
        """
        assert isinstance(optimizer, torch.optim.Optimizer), "unrecognized optimizer type"
        super().__init__(model, config_list, optimizer, dummy_input)
        device = next(model.parameters()).device
        self.quant_grad = QuantForward()
        modules_to_compress = self.get_modules_to_compress()
        self.bound_model.register_buffer("steps", torch.Tensor([1]))
        for layer, config in modules_to_compress:
            if "weight" in config.get("quant_types", []):
                layer.module.register_parameter("weight_scale", torch.nn.Parameter(torch.Tensor([1.0])))
                # todo: support per-channel quantization for weight since TensorRT use it for conv weight
                weight_bits = get_bits_length(config, "weight")
                layer.module.register_buffer('weight_bits', torch.Tensor([weight_bits]))
                qmax = 2 ** (weight_bits - 1) - 1
                qmin = -2 ** (weight_bits - 1)
                init_weight_scale = layer.module.weight.data.detach().abs().mean() * 2 / (qmax ** 0.5)
                layer.module.weight_scale = torch.nn.Parameter(init_weight_scale)
                layer.module.weight_qmax = qmax
                layer.module.weight_qmin = qmin

                self.optimizer.add_param_group({"params": layer.module.weight_scale})

            if "output" in config.get("quant_types", []):
                # scale of output will be initialized using the first batch data
                layer.module.register_parameter("output_scale", torch.nn.Parameter(torch.Tensor([1.0])))
                output_bits = get_bits_length(config, "output")
                layer.module.register_buffer('output_bits', torch.Tensor([output_bits]))
                qmax = 2 ** (output_bits - 1) - 1
                qmin = -2 ** (output_bits - 1)
                layer.module.output_qmax = qmax
                layer.module.output_qmin = qmin

                self.optimizer.add_param_group({"params": layer.module.output_scale})

            if "input" in config.get("quant_types", []):
                # scale of input will be initialized using the first batch data
                layer.module.register_parameter("input_scale", torch.nn.Parameter(torch.Tensor([1.0])))
                input_bits = get_bits_length(config, "input")
                layer.module.register_buffer('input_bits', torch.Tensor([input_bits]))
                qmax = 2 ** (input_bits - 1) - 1
                qmin = -2 ** (input_bits - 1)
                layer.module.input_qmax = qmax
                layer.module.input_qmin = qmin

                self.optimizer.add_param_group({"params": layer.module.input_scale})

        self.bound_model.to(device)

    @staticmethod
    def grad_scale(x, scale):
        """
            Used to scale the gradient. Give tensor `x`, we have `y=grad_scale(x, scale)=x` in the forward pass,
            which means that this function will not change the value of `x`. In the backward pass, we have:

            .. math:

                \frac{\alpha_L}{\alpha_x}=\frac{\alpha_L}{\alpha_y}*\frac{\alpha_y}{\alpha_x}=sclae*\frac{\alpha_L}{\alpha_x}

            This means that the origin gradient of x is scaled by a factor of `scale`. Applying this function
            to a nn.Parameter will scale the gradient of it without changing its value.
        """
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    @staticmethod
    def round_pass(x):
        """
            A simple way to achieve STE operation.
        """
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad

    def quantize(self, x, scale, qmin, qmax):
        grad_scale_factor = 1.0 / ((qmax * x.numel()) ** 0.5)
        scale = self.grad_scale(scale, grad_scale_factor)
        x = x / scale
        x = torch.clamp(x, qmin, qmax)
        x = self.round_pass(x)
        x = x * scale
        return x

    def quantize_weight(self, wrapper, **kwargs):
        module = wrapper.module
        weight = wrapper.module.weight

        # todo: add support for quantize bias. If we use TensorRT as backend, there is no need to quantize
        # bias
        weight = self.quantize(weight, module.weight_scale, module.weight_qmin, module.weight_qmax)
        module.weight = weight
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        module = wrapper.module

        # initialize the scale
        if self.bound_model.steps == 1:
            qmax = module.output_qmax
            init_oup_scale = output.data.detach().abs().mean() * 2 / (qmax ** 0.5)
            module.output_scale.data = init_oup_scale

        output = self.quantize(output, module.output_scale, module.output_qmin, module.output_qmax)
        return output

    def quantize_input(self, inputs, wrapper, **kwargs):
        module = wrapper.module
        # initialize the scale
        if self.bound_model.steps == 1:
            qmax = module.input_qmax
            init_oup_scale = inputs.data.detach().abs().mean() * 2 / (qmax ** 0.5)
            module.input_scale.data = init_oup_scale

        inputs = self.quantize(inputs, module.input_scale, module.input_qmin, module.input_qmax)
        return inputs

    def load_calibration_config(self, calibration_config):
        modules_to_compress = self.get_modules_to_compress()
        for layer, _ in modules_to_compress:
            name, module = layer.name, layer.module
            if name not in calibration_config:
                if hasattr(module, 'weight_bits') or hasattr(module, 'output_bits') or hasattr(module, 'input_bits'):
                    logger.warning(f"Can not find module {name}'s parameter in input config.")
                continue
            if hasattr(module, 'weight_bits'):
                assert calibration_config[name]['weight_bits'] == int(module.weight_bits), f"weight bits of module {name} fail to match"
            if hasattr(module, 'input_bits'):
                assert calibration_config[name]['input_bits'] == int(module.input_bits), f"input bits of module {name} fail to match"
                module.input_scale.data = torch.Tensor([float(calibration_config[name]['tracked_max_input'] / module.input_qmax)])

            if hasattr(module, 'output_bits'):
                assert calibration_config[name]['output_bits'] == int(module.output_bits), f"output bits of module {name} fail to match"
                module.output_scale.data = torch.Tensor([float(calibration_config[name]['tracked_max_output'] / module.output_qmax)])

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
            if hasattr(module, 'input_bits') or hasattr(module, 'weight_bits') or hasattr(module, 'output_bits'):
                calibration_config[name] = {}
            if hasattr(module, 'weight_bits'):
                calibration_config[name]['weight_bits'] = int(module.weight_bits)
                abs_max_input = float(module.input_scale * module.input_qmax)
                calibration_config[name]['tracked_min_input'] = -abs_max_input
                calibration_config[name]['tracked_max_input'] = abs_max_input
                actual_weight = getattr(module, 'old_weight', None)
                if actual_weight is None:
                    logger.warning("Can not recover weight for layer %s. "
                                   "This may lead to a wrong accuracy performance on the backend.", name)
                delattr(module, 'weight')
                module.register_parameter('weight', actual_weight)
                if hasattr(module, BN_FOLD_TAG):
                    actual_bias = getattr(module, 'old_bias', None)
                    delattr(module, 'bias')
                    if actual_bias is not None:
                        module.register_parameter('bias', actual_bias)
                    else:
                        setattr(module, 'bias', None)
            if hasattr(module, 'input_bits'):
                calibration_config[name]['input_bits'] = int(module.input_bits)
                abs_max_input = float(module.input_scale * module.input_qmax)
                calibration_config[name]['tracked_min_input'] = -abs_max_input
                calibration_config[name]['tracked_max_input'] = abs_max_input
            if hasattr(module, 'output_bits'):
                calibration_config[name]['output_bits'] = int(module.output_bits)
                abs_max_output = float(module.output_scale * module.output_qmax)
                calibration_config[name]['tracked_min_output'] = -abs_max_output
                calibration_config[name]['tracked_max_output'] = abs_max_output
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path,
                               input_shape, device)

        return calibration_config

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'tracked_min_input', 'tracked_max_input', 'tracked_min_output', \
        'tracked_max_output', 'output_scale', 'input_scale', 'weight_scale','weight_bits', 'output_bits', 'input_bits', 'BN_FOLD_TAG']
        for attr in del_attr_list:
            if hasattr(module, attr):
                delattr(module, attr)

    def step_with_optimizer(self):
        """
        override `compressor` `step` method, quantization only happens after certain number of steps
        """
        self.bound_model.steps += 1
