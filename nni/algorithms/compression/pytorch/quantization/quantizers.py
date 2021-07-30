# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import copy
from collections import defaultdict
import torch
from schema import Schema, And, Or, Optional
from nni.compression.pytorch.utils.config_validation import QuantizerSchema
from nni.compression.pytorch.compressor import BN_FOLD_TAG, Quantizer, QuantForward, QuantGrad, QuantType

from .observers import default_weight_observer, default_histogram_observer

__all__ = ['NaiveQuantizer', 'QAT_Quantizer', 'DoReFaQuantizer', 'BNNQuantizer', 'LsqQuantizer', 'ObserverQuantizer']

logger = logging.getLogger(__name__)


class NaiveQuantizer(Quantizer):
    """quantize weight to 8 bits
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        self.layer_scale = {}

    def validate_config(self, model, config_list):
        schema = QuantizerSchema([{
            Optional('quant_types'): ['weight'],
            Optional('quant_bits'): Or(8, {'weight': 8}),
            Optional('op_types'): [str],
            Optional('op_names'): [str],
            Optional('exclude'): bool
        }], model, logger)

        schema.validate(config_list)

    def quantize_weight(self, wrapper, **kwargs):
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        new_scale = weight.abs().max() / 127
        scale = max(self.layer_scale.get(wrapper.name, 0), new_scale)
        self.layer_scale[wrapper.name] = scale
        orig_type = weight.type()  # TODO: user layer
        weight = weight.div(scale).type(torch.int8).type(orig_type).mul(scale)
        wrapper.module.weight = weight
        return weight

def update_ema(biased_ema, value, decay):
    """
    calculate biased stat and unbiased stat in each step using exponential moving average method

    Parameters
    ----------
    biased_ema : float
        previous stat value
    value : float
        current stat value
    decay : float
        the weight of previous stat value, larger means smoother curve

    Returns
    -------
    float, float
    """
    biased_ema = biased_ema * decay + (1 - decay) * value
    return biased_ema


def update_quantization_param(bits, rmin, rmax):
    """
    calculate the `zero_point` and `scale`.

    Parameters
    ----------
    bits : int
        quantization bits length
    rmin : Tensor
        min value of real value
    rmax : Tensor
        max value of real value

    Returns
    -------
    float, float
    """
    # extend the [min, max] interval to ensure that it contains 0.
    # Otherwise, we would not meet the requirement that 0 be an exactly
    # representable value.
    rmin = torch.min(rmin, torch.Tensor([0]).to(rmin.device))
    rmax = torch.max(rmax, torch.Tensor([0]).to(rmin.device))
    qmin = torch.Tensor([0]).to(rmin.device)
    qmax = torch.Tensor([(1 << bits) - 1]).to(rmin.device)

    # First determine the scale.
    scale = (rmax - rmin) / (qmax - qmin)

    # Zero-point computation.
    initial_zero_point = qmin - rmin / scale

    # Now we need to nudge the zero point to be an integer
    if initial_zero_point < qmin:
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = torch.round(initial_zero_point)

    return scale, nudged_zero_point


def get_bits_length(config, quant_type):
    if isinstance(config["quant_bits"], int):
        return config["quant_bits"]
    else:
        return config["quant_bits"].get(quant_type)

class QATGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax):
        tensor_q = QuantGrad._quantize(tensor, scale, zero_point)
        mask = (tensor_q < qmin) | (tensor_q > qmax)
        grad_output[mask] = 0
        return grad_output


class ObserverQuantizer(Quantizer):
    """This quantizer uses observers to record weight/activation statistics to get quantization information.
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
        #  activation observer : per_tensor_affine, quint8, reduce_range=True
        # 2. add more kinds of observers, such as Kullback-Leibler divergence.
        # 3. add batch normalization folding
        assert not model.training, "Currently the observer quantizer only works in evaluation mode."
        self.quant_grad = QuantForward()
        self.device = next(model.parameters()).device
        modules_to_compress = self.get_modules_to_compress()
        all_observers = defaultdict(dict)
        weight_q_min, weight_q_max = -127, 127
        activation_q_min, activation_q_max = 0, 127  # reduce_range is set to True
        self.compressed = False

        for layer, config in modules_to_compress:
            layer_name = layer.name
            module = layer.module
            if "weight" in config.get("quant_types", []):
                all_observers[layer_name]["weight"] = default_weight_observer()
                setattr(module, "weight_qmax", weight_q_max)
                setattr(module, "weight_qmin", weight_q_min)
            if "input" in config.get("quant_types", []):
                all_observers[layer_name]["input"] = default_histogram_observer()
                setattr(module, "input_qmax", activation_q_max)
                setattr(module, "input_qmin", activation_q_min)
            if "output" in config.get("quant_types", []):
                all_observers[layer_name]["output"] = default_histogram_observer()
                setattr(module, "output_qmax", activation_q_max)
                setattr(module, "output_qmin", activation_q_min)
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
        if isinstance(tensor, tuple):
            # NB: This only works for single tensor
            tensor = (t.cpu() for t in tensor)
            observer(*tensor)
        else:
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

    def quantize_input(self, *inputs, wrapper, **kwargs):
        if self.compressed:
            module = wrapper.module
            new_input = self._quantize(inputs[0],
                                      module.input_scale,
                                      module.input_zero_point,
                                      module.input_qmin,
                                      module.input_qmax)
            list_inp = list(inputs)
            list_inp[0] = new_input
            inputs = tuple(list_inp)
        else:
            self.record(wrapper, 'input', inputs)
        return inputs

    def quantize_weight(self, wrapper, **kwargs):
        # If ObserverQuantizer.compress is executed, the weight will be set to
        # the Pseudo-quantized one. So there is no need to quantize it
        if self.compressed:
            return

        module = wrapper.module
        old_weight = module.weight
        self.record(wrapper, 'weight', old_weight)

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
                module.register_parameter('weight', torch.nn.Parameter(quantized_weight))
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
                calibration_config[name]['weight_bit'] = 8
                val = float(module.weight_scale * module.weight_qmax)
                calibration_config[name]['tracked_max_weight'] = val
                calibration_config[name]['tracked_min_weight'] = -val
                calibration_config[name]['tracked_weight_qmin'] = -127
                calibration_config[name]['tracked_weight_qmax'] = 127
            # refactor these magic numbers when customizations of dtype and qscheme are ready.
            if hasattr(module, 'input_scale'):
                calibration_config[name]['input_bit'] = 8
                max_input = float(module.input_scale * (module.input_qmax - module.input_zero_point))
                min_input = float(module.input_scale * (module.input_qmin - module.input_zero_point))
                calibration_config[name]['tracked_min_input'] = min_input
                calibration_config[name]['tracked_max_input'] = max_input
                calibration_config[name]['tracked_input_qmin'] = 0
                calibration_config[name]['tracked_input_qmax'] = 127
            if hasattr(module, 'output_scale'):
                calibration_config[name]['activation_bit'] = 8
                max_input = float(module.output_scale * (module.output_qmax - module.output_zero_point))
                min_input = float(module.output_scale * (module.output_qmin - module.output_zero_point))
                calibration_config[name]['tracked_min_activation'] = min_input
                calibration_config[name]['tracked_max_activation'] = max_input
                calibration_config[name]['tracked_activation_qmin'] = 0
                calibration_config[name]['tracked_activation_qmax'] = 127
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


class QAT_Quantizer(Quantizer):
    """Quantizer defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """

    def __init__(self, model, config_list, optimizer=None, dummy_input=None):
        """
        Parameters
        ----------
        layer : LayerInfo
            the layer to quantize
        config_list : list of dict
            list of configurations for quantization
            supported keys for dict:
                - quant_types : list of string
                    type of quantization you want to apply, currently support 'weight', 'input', 'output'
                - quant_bits : int or dict of {str : int}
                    bits length of quantization, key is the quantization type, value is the length, eg. {'weight', 8},
                    when the type is int, all quantization types share same bits length
                - quant_start_step : int
                    disable quantization until model are run by certain number of steps, this allows the network to enter a more stable
                    state where activation quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0
                - op_types : list of string
                    types of nn.module you want to apply quantization, eg. 'Conv2d'
                - dummy_input : tuple of tensor
                    inputs to the model, which are used to get the graph of the module. The graph is used to find
                    Conv-Bn patterns. And then the batch normalization folding would be enabled. If dummy_input is not
                    given, the batch normalization folding would be disabled.
        """

        super().__init__(model, config_list, optimizer, dummy_input)
        self.quant_grad = QATGrad.apply
        modules_to_compress = self.get_modules_to_compress()
        device = next(model.parameters()).device
        self.bound_model.register_buffer("steps", torch.Tensor([1]))
        for layer, config in modules_to_compress:
            layer.module.register_buffer("zero_point", torch.Tensor([0.0]))
            layer.module.register_buffer("scale", torch.Tensor([1.0]))
            layer.module.register_buffer('ema_decay', torch.Tensor([0.99]))
            if "weight" in config.get("quant_types", []):
                layer.module.register_buffer('weight_bit', torch.zeros(1))
                layer.module.register_buffer('tracked_min_input', torch.zeros(1))
                layer.module.register_buffer('tracked_max_input', torch.zeros(1))
            if "output" in config.get("quant_types", []):
                layer.module.register_buffer('activation_bit', torch.zeros(1))
                layer.module.register_buffer('tracked_min_activation', torch.zeros(1))
                layer.module.register_buffer('tracked_max_activation', torch.zeros(1))
        self.bound_model.to(device)

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'old_bias', 'ema_decay', 'tracked_min_activation', 'tracked_max_activation',
                         'tracked_min_input', 'tracked_max_input', 'scale', 'zero_point', 'weight_bit',
                         'activation_bit', 'BN_FOLD_TAG']
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
            Optional('quant_start_step'): And(int, lambda n: n >= 0),
            Optional('op_types'): [str],
            Optional('op_names'): [str],
            Optional('exclude'): bool
        }], model, logger)

        schema.validate(config_list)

    def _quantize(self, bits, op, real_val):
        """
        quantize real value.

        Parameters
        ----------
        bits : int
            quantization bits length
        op : torch.nn.Module
            target module
        real_val : Tensor
            real value to be quantized

        Returns
        -------
        Tensor
        """
        op.zero_point = op.zero_point.to(real_val.device)
        op.scale = op.scale.to(real_val.device)
        transformed_val = op.zero_point + real_val / op.scale
        qmin = 0
        qmax = (1 << bits) - 1
        clamped_val = torch.clamp(transformed_val, qmin, qmax)
        quantized_val = torch.round(clamped_val)
        return quantized_val

    def _dequantize(self, op, quantized_val):
        """
        dequantize quantized value.
        Because we simulate quantization in training process, all the computations still happen as float point computations, which means we
        first quantize tensors then dequantize them. For more details, please refer to the paper.

        Parameters
        ----------
        op : torch.nn.Module
            target module
        quantized_val : float
            quantized_val value to be dequantized

        Returns
        -------
        float
        """
        real_val = op.scale * (quantized_val - op.zero_point)
        return real_val

    def quantize_weight(self, wrapper, **kwargs):
        config = wrapper.config
        module = wrapper.module
        input = kwargs['input_tensor']  # pylint: disable=redefined-builtin
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight_bits = get_bits_length(config, 'weight')
        quant_start_step = config.get('quant_start_step', 0)
        assert weight_bits >= 1, "quant bits length should be at least 1"

        # we dont update weight in evaluation stage
        if quant_start_step > self.bound_model.steps:
            module.tracked_min_input, module.tracked_max_input = torch.min(input), torch.max(input)
            return weight

        if not wrapper.training:
            return weight

        current_min, current_max = torch.min(input), torch.max(input)
        module.tracked_min_input = update_ema(module.tracked_min_input, current_min,
                                                                    module.ema_decay)
        module.tracked_max_input = update_ema(module.tracked_max_input, current_max,
                                                                    module.ema_decay)

        # quantize weight
        rmin, rmax = torch.min(weight), torch.max(weight)
        module.scale, module.zero_point = update_quantization_param(weight_bits, rmin, rmax)
        weight = self._quantize(weight_bits, module, weight)
        weight = self._dequantize(module, weight)
        module.weight_bit = torch.Tensor([weight_bits])
        wrapper.module.weight = weight
        return weight

    def quantize_output(self, output, wrapper, **kwargs):
        config = wrapper.config
        module = wrapper.module
        output_bits = get_bits_length(config, 'output')
        module.activation_bit = torch.Tensor([output_bits])
        quant_start_step = config.get('quant_start_step', 0)
        assert output_bits >= 1, "quant bits length should be at least 1"

        if quant_start_step > self.bound_model.steps:
            module.tracked_min_activation, module.tracked_max_activation = torch.min(output), torch.max(output)
            return output

        # we dont update output quantization parameters in evaluation stage
        if wrapper.training:
            current_min, current_max = torch.min(output), torch.max(output)
            module.tracked_min_activation = update_ema(module.tracked_min_activation, current_min,
                                                                       module.ema_decay)
            module.tracked_max_activation = update_ema(module.tracked_max_activation, current_max,
                                                                       module.ema_decay)
            module.scale, module.zero_point = update_quantization_param(
                output_bits, module.tracked_min_activation, module.tracked_max_activation)
        out = self._quantize(output_bits, module, output)
        out = self._dequantize(module, out)
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
            if hasattr(module, 'weight_bit') or hasattr(module, 'activation_bit'):
                calibration_config[name] = {}
            if hasattr(module, 'weight_bit'):
                calibration_config[name]['weight_bit'] = int(module.weight_bit)
                calibration_config[name]['tracked_min_input'] = float(module.tracked_min_input)
                calibration_config[name]['tracked_max_input'] = float(module.tracked_max_input)

                # Recover weight/bias for batch normalization folding
                if hasattr(module, BN_FOLD_TAG):
                    actual_weight = getattr(module, 'old_weight', None)
                    if actual_weight is None:
                        logger.warning("Can not recover weight for layer %s. "
                                       "This may lead to a wrong accuracy performance on the backend.", name)
                    delattr(module, 'weight')
                    module.register_parameter('weight', actual_weight)

                    actual_bias = getattr(module, 'old_bias', None)
                    delattr(module, 'bias')
                    if actual_bias is not None:
                        module.register_parameter('bias', actual_bias)
                    else:
                        setattr(module, 'bias', None)

            if hasattr(module, 'activation_bit'):
                calibration_config[name]['activation_bit'] = int(module.activation_bit)
                calibration_config[name]['tracked_min_activation'] = float(module.tracked_min_activation)
                calibration_config[name]['tracked_max_activation'] = float(module.tracked_max_activation)
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path, input_shape, device)

        return calibration_config

    def fold_bn(self, *inputs, wrapper):
        """
        Simulate batch normalization folding in the training graph. Folded weight and bias are
        returned for the following operations.

        Parameters
        ----------
        inputs : tuple of torch.Tensor
            inputs for the module
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module

        Returns
        -------
        Tuple of torch.Tensor
        """
        module = wrapper.module
        bn_module = wrapper.bn_module
        with torch.no_grad():
            output = module(*inputs)
            _ = bn_module(output)
        running_mean = bn_module.running_mean
        running_var = torch.sqrt(bn_module.running_var + bn_module.eps)
        bn_weight = bn_module.weight
        bn_bias = bn_module.bias
        dimensions = len(module.weight.shape)
        shape = [-1] + [1] * (dimensions - 1)
        new_weight = module.old_weight * bn_weight.reshape(shape) / running_var.reshape(shape)
        if hasattr(module, 'old_bias'):
            new_bias = bn_bias + (module.old_bias - running_mean) / running_var * bn_weight
        else:
            new_bias = bn_bias - running_mean / running_var * bn_weight
        return new_weight, new_bias

    def step_with_optimizer(self):
        """
        override `compressor` `step` method, quantization only happens after certain number of steps
        """
        self.bound_model.steps += 1


class DoReFaQuantizer(Quantizer):
    """Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        device = next(model.parameters()).device
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            if "weight" in config.get("quant_types", []):
                layer.module.register_buffer('weight_bit', torch.zeros(1))
        self.bound_model.to(device)

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'weight_bit']
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
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight_bits = get_bits_length(wrapper.config, 'weight')
        weight = weight.tanh()
        weight = weight / (2 * weight.abs().max()) + 0.5
        weight = self.quantize(weight, weight_bits)
        weight = 2 * weight - 1
        wrapper.module.weight = weight
        wrapper.module.weight_bit = torch.Tensor([weight_bits])
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
            if hasattr(module, 'weight_bit'):
                calibration_config[name] = {}
                calibration_config[name]['weight_bit'] = int(module.weight_bit)
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path, input_shape, device)

        return calibration_config


class ClipGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax):
        if quant_type == QuantType.QUANT_OUTPUT:
            grad_output[torch.abs(tensor) > 1] = 0
        return grad_output


class BNNQuantizer(Quantizer):
    """Binarized Neural Networks, as defined in:
    Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1
    (https://arxiv.org/abs/1602.02830)
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, optimizer)
        device = next(model.parameters()).device
        self.quant_grad = ClipGrad.apply
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            if "weight" in config.get("quant_types", []):
                layer.module.register_buffer('weight_bit', torch.zeros(1))
        self.bound_model.to(device)

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'weight_bit']
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
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        weight = torch.sign(weight)
        # remove zeros
        weight[weight == 0] = 1
        wrapper.module.weight = weight
        wrapper.module.weight_bit = torch.Tensor([1.0])
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
            if hasattr(module, 'weight_bit'):
                calibration_config[name] = {}
                calibration_config[name]['weight_bit'] = int(module.weight_bit)
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path, input_shape, device)

        return calibration_config


class LsqQuantizer(Quantizer):
    """Quantizer defined in:
       Learned Step Size Quantization (ICLR 2020)
       https://arxiv.org/pdf/1902.08153.pdf
    """

    def __init__(self, model, config_list, optimizer=None):
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
                    state where activation quantization ranges do not exclude a signiﬁcant fraction of values, default value is 0
                - op_types : list of string
                    types of nn.module you want to apply quantization, eg. 'Conv2d'
        """
        super().__init__(model, config_list, optimizer)
        device = next(model.parameters()).device
        self.quant_grad = QuantForward()
        modules_to_compress = self.get_modules_to_compress()
        self.bound_model.register_buffer("steps", torch.Tensor([1]))
        for layer, config in modules_to_compress:
            if "weight" in config.get("quant_types", []):
                layer.module.register_parameter("weight_scale", torch.nn.Parameter(torch.Tensor([1.0])))
                # todo: support per-channel quantization for weight since TensorRT use it for conv weight
                q_bit = get_bits_length(config, "weight")
                layer.module.register_buffer('weight_bit', torch.Tensor([q_bit]))
                qmax = 2 ** (q_bit - 1) - 1
                qmin = -2 ** (q_bit - 1)
                init_weight_scale = layer.module.weight.data.detach().abs().mean() * 2 / (qmax ** 0.5)
                layer.module.weight_scale = torch.nn.Parameter(init_weight_scale)
                layer.module.weight_qmax = qmax
                layer.module.weight_qmin = qmin

                self.optimizer.add_param_group({"params": layer.module.weight_scale})

            if "output" in config.get("quant_types", []):
                # scale of activation will be initialized using the first batch data
                layer.module.register_parameter("output_scale", torch.nn.Parameter(torch.Tensor([1.0])))
                q_bit = get_bits_length(config, "output")
                layer.module.register_buffer('output_bit', torch.Tensor([q_bit]))
                qmax = 2 ** (q_bit - 1) - 1
                qmin = -2 ** (q_bit - 1)
                layer.module.output_qmax = qmax
                layer.module.output_qmin = qmin

                self.optimizer.add_param_group({"params": layer.module.output_scale})

            if "input" in config.get("quant_types", []):
                # scale of input will be initialized using the first batch data
                layer.module.register_parameter("input_scale", torch.nn.Parameter(torch.Tensor([1.0])))
                q_bit = get_bits_length(config, "input")
                layer.module.register_buffer('input_bit', torch.Tensor([q_bit]))
                qmax = 2 ** (q_bit - 1) - 1
                qmin = -2 ** (q_bit - 1)
                layer.module.input_qmax = qmax
                layer.module.input_qmin = qmin

                self.optimizer.add_param_group({"params": layer.module.input_scale})

        self.bound_model.to(device)

    @staticmethod
    def grad_scale(x, scale):
        """
            Used to scale the gradient. Give tensor `x`, we have `y=grad_scale(x, scale)=x` in the forward pass,
            which means that this function will not change the value of `x`. In the backward pass, we have:

            :math:`\frac{\alpha_L}{\alpha_x}=\frac{\alpha_L}{\alpha_y}*\frac{\alpha_y}{\alpha_x}=sclae*\frac{\alpha_L}{\alpha_x}`

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

        # todo: add support for quantize bias. If we use TensorRT as backend, there is no need to quantize
        # bias
        old_weight = module.old_weight
        weight = self.quantize(old_weight, module.weight_scale, module.weight_qmin, module.weight_qmax)
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

    def quantize_input(self, *inputs, wrapper, **kwargs):
        # This is hacky since it is not recommended to modify a tuple
        # NB: support layers with multi inputs
        module = wrapper.module
        # initialize the scale
        if self.bound_model.steps == 1:
            qmax = module.input_qmax
            init_oup_scale = inputs[0].data.detach().abs().mean() * 2 / (qmax ** 0.5)
            module.input_scale.data = init_oup_scale

        new_input = self.quantize(inputs[0], module.input_scale, module.input_qmin, module.input_qmax)
        list_inp = list(inputs)
        list_inp[0] = new_input
        return tuple(list_inp)

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
            if hasattr(module, 'input_bit') or hasattr(module, 'output_bit'):
                calibration_config[name] = {}
            if hasattr(module, 'weight_bit'):
                calibration_config[name]['weight_bit'] = int(module.weight_bit)
                abs_max_input = float(module.input_scale * module.input_qmax)
                calibration_config[name]['tracked_min_input'] = -abs_max_input
                calibration_config[name]['tracked_max_input'] = abs_max_input
            if hasattr(module, 'output_bit'):
                calibration_config[name]['activation_bit'] = int(module.output_bit)
                abs_max_output = float(module.output_scale * module.output_qmax)
                calibration_config[name]['tracked_min_activation'] = -abs_max_output
                calibration_config[name]['tracked_max_activation'] = abs_max_output
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path,
                               input_shape, device)

        return calibration_config

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'tracked_min_input', 'tracked_max_input', 'tracked_min_activation', \
        'tracked_max_activation', 'output_scale', 'input_scale', 'weight_scale','weight_bit', 'output_bit', 'input_bit']
        for attr in del_attr_list:
            if hasattr(module, attr):
                delattr(module, attr)

    def step_with_optimizer(self):
        """
        override `compressor` `step` method, quantization only happens after certain number of steps
        """
        self.bound_model.steps += 1
