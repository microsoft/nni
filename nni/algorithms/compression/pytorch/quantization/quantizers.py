# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import defaultdict
import torch
from schema import Schema, And, Or, Optional
from nni.compression.pytorch.utils.config_validation import QuantizerSchema
from nni.compression.pytorch.compressor import BN_FOLD_TAG, Quantizer, QuantForward, QuantGrad
from nni.compression.pytorch.quantization.literal import (
    PER_CHANNEL_QUANT_SCHEME,
    QuantScheme,
    QuantDtype,
    QuantType
)
from nni.compression.pytorch.quantization.observers import default_weight_observer, default_histogram_observer
from nni.compression.pytorch.quantization.settings import LayerQuantSetting
from nni.compression.pytorch.quantization.utils import (
    calculate_qmin_qmax,
    get_bits_length,
    get_min_max_value,
    get_quant_shape
)

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
        weight = wrapper.module.weight
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


def update_quantization_param(bits, rmin, rmax, dtype, scheme):
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
    dtype : QuantDtype
        quantized data type
    scheme : QuantScheme
        quantization scheme to be used
    Returns
    -------
    float, float
    """

    # extend the [min, max] interval to ensure that it contains 0.
    # Otherwise, we would not meet the requirement that 0 be an exactly
    # representable value.
    # I think this is for activations that need to be pad in the training.
    # However this is a default behavior in PyTorch quantization observer.
    # So we also make it a default behavior
    rmin = torch.min(rmin, torch.zeros_like(rmin))
    rmax = torch.max(rmax, torch.zeros_like(rmax))
    zero_point = torch.zeros_like(rmin)

    # todo: there is no need to calculate qmin and qmax again
    qmin, qmax = calculate_qmin_qmax(bits, dtype)

    if scheme in [QuantScheme.PER_TENSOR_SYMMETRIC, QuantScheme.PER_CHANNEL_SYMMETRIC]:
        abs_max = torch.max(torch.abs(rmin), torch.abs(rmax))
        scale = abs_max / (float(qmax - qmin) / 2)
        if dtype == QuantDtype.UINT:
            zero_point_val = (qmin + qmax) // 2
            zero_point = zero_point.new_full(zero_point.size(), zero_point_val)
    else:
        scale = (rmax - rmin) / float(qmax - qmin)
        zero_point = qmin - torch.round(rmin / scale)

    zero_point = torch.clamp(zero_point, qmin, qmax)

    # todo: add these lines
    # eps = torch.finfo(torch.float32).eps
    # scale = torch.max(scale, eps)

    return scale, zero_point


class QATGrad(QuantGrad):
    @staticmethod
    def quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax):
        tensor_q = QuantGrad._quantize(tensor, scale, zero_point)
        mask = (tensor_q < qmin) | (tensor_q > qmax)
        grad_output[mask] = 0
        return grad_output


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


class QAT_Quantizer(Quantizer):
    """Quantizer defined in:
    Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    """

    def __init__(self, model, config_list, optimizer, dummy_input=None):
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
        self.quant_grad = QATGrad.apply
        modules_to_compress = self.get_modules_to_compress()
        device = next(model.parameters()).device
        self.bound_model.register_buffer("steps", torch.tensor(1))
        for layer, config in modules_to_compress:
            module = layer.module
            name = layer.name
            # TODO: may relax this limitation?
            assert name in self.all_shapes, "Could not found shapes for layer {}".format(name)
            input_shape, output_shape = self.all_shapes[name]
            layer_quant_setting = LayerQuantSetting(config)
            layer_quant_setting.ema_decay = 0.99
            quant_start_step = config.get('quant_start_step', 0)
            layer_quant_setting.quant_start_step = quant_start_step
            # todo: support other ranks and remove this check
            if isinstance(module, torch.nn.Linear):
                if "input" in config.get("quant_types", []) and \
                        layer_quant_setting.input.quant_scheme in PER_CHANNEL_QUANT_SCHEME:
                    if len(input_shape) != 2:
                        logger.warning("When quantize torch.nn.Linear, make sure that the rank of the inputs "
                                       "of the layer is 2. Skip quantization of layer %s.", name)
                        continue
                if "output" in config.get("quant_types", []) and \
                        layer_quant_setting.output.quant_scheme in PER_CHANNEL_QUANT_SCHEME:
                    if len(output_shape) != 2:
                        logger.warning("When quantize torch.nn.Linear, make sure that the rank of the outputs "
                                       "of the layer is 2. Skip quantization of layer %s.", name)
                        continue

            if "weight" in config.get("quant_types", []):
                quant_shape = get_quant_shape(module.weight.shape, QuantType.WEIGHT, layer_quant_setting.weight.quant_scheme)
                module.register_buffer('weight_scale', torch.zeros(quant_shape))
                module.register_buffer('weight_zero_point', torch.zeros(quant_shape))

            if "input" in config.get("quant_types", []):
                quant_shape = get_quant_shape(input_shape, QuantType.INPUT, layer_quant_setting.input.quant_scheme)
                module.register_buffer('tracked_min_input', torch.zeros(quant_shape))
                module.register_buffer('tracked_max_input', torch.zeros(quant_shape))
                module.register_buffer('input_scale', torch.zeros(quant_shape))
                module.register_buffer('input_zero_point', torch.zeros(quant_shape))

            if "output" in config.get("quant_types", []):
                quant_shape = get_quant_shape(output_shape, QuantType.OUTPUT, layer_quant_setting.output.quant_scheme)
                module.register_buffer('tracked_min_output', torch.zeros(quant_shape))
                module.register_buffer('tracked_max_output', torch.zeros(quant_shape))
                module.register_buffer('output_scale', torch.zeros(quant_shape))
                module.register_buffer('output_zero_point', torch.zeros(quant_shape))

            setattr(module, "layer_quant_setting", layer_quant_setting)
        self.bound_model.to(device)

    def _del_simulated_attr(self, module):
        """
        delete redundant parameters in quantize module
        """
        del_attr_list = ['old_weight', 'old_bias', 'ema_decay', 'tracked_min_output', 'tracked_max_output',
                         'tracked_min_input', 'tracked_max_input', 'BN_FOLD_TAG',
                         'weight_scale', 'weight_zero_point', 'input_scale', 'input_zero_point',
                         'output_scale', 'output_zero_point', 'layer_quant_setting']
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
        SUPPORTED_OPS = ['Conv2d', 'Linear', 'ReLU', 'ReLU6']
        schema = QuantizerSchema([{
            Optional('quant_types'): Schema([lambda x: x in ['weight', 'output', 'input']]),
            Optional('quant_bits'): Or(And(int, lambda n: 0 < n < 32), Schema({
                Optional('input'): And(int, lambda n: 0 < n < 32),
                Optional('weight'): And(int, lambda n: 0 < n < 32),
                Optional('output'): And(int, lambda n: 0 < n < 32),
            })),
            Optional('quant_scheme'): Or(lambda x: x in QuantScheme, Schema({
                Optional('input'): lambda x: x in QuantScheme,
                Optional('weight'): lambda x: x in QuantScheme,
                Optional('output'): lambda x: x in QuantScheme
            })),
            Optional('quant_dtype'): Or(lambda x: x in QuantDtype, Schema({
                Optional('input'): lambda x: x in QuantDtype,
                Optional('weight'): lambda x: x in QuantDtype,
                Optional('output'): lambda x: x in QuantDtype
            })),
            Optional('quant_start_step'): And(int, lambda n: n >= 0),
            Optional('op_types'): [And(str, lambda n: n in SUPPORTED_OPS)],
            Optional('op_names'): [str],
            Optional('exclude'): bool
        }], model, logger)

        schema.validate(config_list)

    def _quantize(self, real_value, scale, zero_point, qmin, qmax):
        """
        quantize real value.

        Parameters
        ----------
        real_value : torch.Tensor
            the real value to be quantized
        scale : torch.Tensor
            quantization scale
        zero_point : torch.Tensor
            quantization zero point
        qmin : int
            lower bound of the int range
        qmax : int
            upper bound of the int range

        Returns
        -------
        Tensor
        """
        transformed_val = zero_point + real_value / scale
        clamped_val = torch.clamp(transformed_val, qmin, qmax)
        quantized_val = torch.round(clamped_val)
        return quantized_val

    def _dequantize(self, quantized_val, scale, zero_point):
        """
        dequantize quantized value.
        Because we simulate quantization in training process, all the computations still happen as float point computations, which means we
        first quantize tensors then dequantize them. For more details, please refer to the paper.

        Parameters
        ----------
        quantized_val : torch.Tensor
            the quantized value to be de-quantized
        scale : torch.Tensor
            quantization scale
        zero_point : torch.Tensor
            quantization zero point

        Returns
        -------
        Tensor
        """
        real_val = scale * (quantized_val - zero_point)
        return real_val

    def quantize_weight(self, wrapper, **kwargs):
        module = wrapper.module
        weight = module.weight
        layer_quant_setting = module.layer_quant_setting
        tensor_quant_setting = layer_quant_setting.weight

        # layer-wise settings
        quant_start_step = layer_quant_setting.quant_start_step

        # tensor-wise settings
        dtype = tensor_quant_setting.quant_dtype
        scheme = tensor_quant_setting.quant_scheme
        qmin, qmax = tensor_quant_setting.get_qmin_qmax()
        bits = tensor_quant_setting.bits

        # In evaluation mode, we only quantize weight without updating statistics
        if not wrapper.training:
            scale, zero_point = module.weight_scale, module.weight_zero_point
            weight = self._quantize(weight, scale, zero_point, qmin, qmax)
            weight = self._dequantize(weight, scale, zero_point)
            module.weight = weight
            return weight

        if quant_start_step > int(self.bound_model.steps):
            return weight

        current_min, current_max = get_min_max_value(weight, QuantType.WEIGHT, scheme)
        scale, zero_point = update_quantization_param(bits, current_min, current_max, dtype, scheme)
        module.weight_scale.copy_(scale)
        module.weight_zero_point.copy_(zero_point)
        weight = self._quantize(weight, scale, zero_point, qmin, qmax)
        weight = self._dequantize(weight, scale, zero_point)
        # Weight can not be in-place modified, so when use torch.nn.DataParallel, this update
        # will be lost after each forward process. However, this update takes effect on each
        # replicated module during each forward process, which will make the quantized weight
        # be used correctly.
        wrapper.module.weight = weight
        return weight

    def quantize_input(self, inputs, wrapper, **kwargs):
        module = wrapper.module

        layer_quant_setting = module.layer_quant_setting
        tensor_quant_setting = layer_quant_setting.input

        # layer-wise settings
        quant_start_step = layer_quant_setting.quant_start_step
        ema_decay = layer_quant_setting.ema_decay

        # tensor-wise settings
        dtype = tensor_quant_setting.quant_dtype
        scheme = tensor_quant_setting.quant_scheme
        qmin, qmax = tensor_quant_setting.get_qmin_qmax()
        bits = tensor_quant_setting.bits

        if not wrapper.training:
            scale = module.input_scale
            zero_point = module.input_zero_point
            inputs = self._quantize(inputs, scale, zero_point, qmin, qmax)
            inputs = self._dequantize(inputs, scale, zero_point)
            return inputs

        current_min, current_max = get_min_max_value(inputs, QuantType.INPUT, scheme)

        if int(self.bound_model.steps) == 1:
            module.tracked_min_input.copy_(current_min)
            module.tracked_max_input.copy_(current_max)

        tracked_min_input = update_ema(module.tracked_min_input, current_min, ema_decay)
        tracked_max_input = update_ema(module.tracked_max_input, current_max, ema_decay)
        module.tracked_min_input.copy_(tracked_min_input)
        module.tracked_max_input.copy_(tracked_max_input)

        if quant_start_step > int(self.bound_model.steps):
            return inputs

        scale, zero_point = update_quantization_param(
            bits, module.tracked_min_input, module.tracked_max_input, dtype, scheme)
        module.input_scale.copy_(scale)
        module.input_zero_point.copy_(zero_point)

        inputs = self._quantize(inputs, scale, zero_point, qmin, qmax)
        inputs = self._dequantize(inputs, scale, zero_point)
        return inputs

    def quantize_output(self, output, wrapper, **kwargs):
        module = wrapper.module
        layer_quant_setting = module.layer_quant_setting
        tensor_quant_setting = layer_quant_setting.output

        # layer-wise settings
        quant_start_step = layer_quant_setting.quant_start_step
        ema_decay = layer_quant_setting.ema_decay

        # tensor-wise settings
        dtype = tensor_quant_setting.quant_dtype
        scheme = tensor_quant_setting.quant_scheme
        qmin, qmax = tensor_quant_setting.get_qmin_qmax()
        bits = tensor_quant_setting.bits

        if not wrapper.training:
            scale = module.output_scale
            zero_point = module.output_zero_point
            output = self._quantize(output, scale, zero_point, qmin, qmax)
            output = self._dequantize(output, scale, zero_point)
            return output

        current_min, current_max = get_min_max_value(output, QuantType.OUTPUT, scheme)

        if int(self.bound_model.steps) == 1:
            module.tracked_min_output.copy_(current_min)
            module.tracked_max_output.copy_(current_max)

        tracked_min_output = update_ema(module.tracked_min_output, current_min, ema_decay)
        tracked_max_output = update_ema(module.tracked_max_output, current_max, ema_decay)
        module.tracked_min_output.copy_(tracked_min_output)
        module.tracked_max_output.copy_(tracked_max_output)

        if quant_start_step > int(self.bound_model.steps):
            return output

        scale, zero_point = update_quantization_param(
            bits, module.tracked_min_output, module.tracked_max_output, dtype, scheme)
        module.output_scale.copy_(scale)
        module.output_zero_point.copy_(zero_point)

        output = self._quantize(output, scale, zero_point, qmin, qmax)
        output = self._dequantize(output, scale, zero_point)
        return output

    def load_calibration_config(self, calibration_config):
        modules_to_compress = self.get_modules_to_compress()
        for layer, _ in modules_to_compress:
            name, module = layer.name, layer.module
            if name not in calibration_config:
                if hasattr(module, 'weight_bits') or hasattr(module, 'output_bits') or hasattr(module, 'input_bits'):
                    logger.warning(f"Can not find module {name}'s parameter in input config.")
                continue
            if hasattr(module, 'weight_bits'):
                assert calibration_config[name]['weight_bits'] == module.weight_bits, f"weight bits of module {name} fail to match"
            if hasattr(module, 'input_bits'):
                assert calibration_config[name]['input_bits'] == module.input_bits, f"input bits of module {name} fail to match"
                module.tracked_min_input.data = torch.tensor([calibration_config[name]['tracked_min_input']])
                module.tracked_max_input.data = torch.tensor([calibration_config[name]['tracked_max_input']])
            if hasattr(module, 'output_bits'):
                assert calibration_config[name]['output_bits'] == module.output_bits, f"output bits of module {name} fail to match"
                module.tracked_min_output.data = torch.tensor([calibration_config[name]['tracked_min_output']])
                module.tracked_max_output.data = torch.tensor([calibration_config[name]['tracked_max_output']])

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
            if hasattr(module, 'weight_bits') or hasattr(module, 'output_bits'):
                calibration_config[name] = {}
            if hasattr(module, 'weight_bits'):
                calibration_config[name]['weight_bits'] = int(module.weight_bits)
                calibration_config[name]['weight_scale'] = module.weight_scale
                calibration_config[name]['weight_zero_point'] = module.weight_zero_point

                # Recover weight/bias for batch normalization folding
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
                calibration_config[name]['tracked_min_input'] = float(module.tracked_min_input)
                calibration_config[name]['tracked_max_input'] = float(module.tracked_max_input)

            if hasattr(module, 'output_bits'):
                calibration_config[name]['output_bits'] = int(module.output_bits)
                calibration_config[name]['tracked_min_output'] = float(module.tracked_min_output)
                calibration_config[name]['tracked_max_output'] = float(module.tracked_max_output)
            self._del_simulated_attr(module)

        self.export_model_save(self.bound_model, model_path, calibration_config, calibration_path, onnx_path, input_shape, device)

        return calibration_config

    def step_with_optimizer(self):
        """
        override `compressor` `step` method, quantization only happens after certain number of steps
        """
        self.bound_model.steps.add_(1)


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
            if hasattr(module, 'input_bits') or hasattr(module, 'output_bits'):
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
