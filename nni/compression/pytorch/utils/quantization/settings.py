# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Optional

from .literal import QuantDtype, QuantType, QuantScheme
from .utils import calculate_qmin_qmax, get_bits_length


# default settings for quantization module
quant_default_settings = {
    QuantType.WEIGHT: {
        'quant_scheme': QuantScheme.PER_TENSOR_AFFINE,
        'quant_dtype': QuantDtype.UINT,
    },
    QuantType.INPUT: {
        'quant_scheme': QuantScheme.PER_TENSOR_AFFINE,
        'quant_dtype': QuantDtype.UINT
    },
    QuantType.OUTPUT: {
        'quant_scheme': QuantScheme.PER_TENSOR_AFFINE,
        'quant_dtype': QuantDtype.UINT
    }
}


class TensorQuantSetting(object):
    def __init__(self, **kwargs):
        self._fields = {}
        for k, v in kwargs.items():
            self._fields[k] = v

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self._fields[name] = val

    def __getattr__(self, name):
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find {} in TensorQuantSetting!".format(name))
        return self._fields[name]

    def get_qmin_qmax(self):
        assert 'qmin' in self._fields and 'qmax' in self._fields, \
            "Can not found qmin & qmax in TensorQuantSetting"
        return self._fields['qmin'], self._fields['qmax']


class LayerQuantSetting(object):
    def __init__(self, config):
        self.input: Optional[TensorQuantSetting] = None
        self.weight: Optional[TensorQuantSetting] = None
        self.output: Optional[TensorQuantSetting] = None
        self._extra_layer_setting = {}

        for quant_type in QuantType:
            if quant_type in config.get("quant_types", []):
                setting = TensorQuantSetting()

                quant_scheme = self.parse_optional_config(config, quant_type, 'quant_scheme')
                setting.quant_scheme = quant_scheme
                quant_dtype = self.parse_optional_config(config, quant_type, 'quant_dtype')
                setting.quant_dtype = quant_dtype

                bits = get_bits_length(config, quant_type)
                qmin, qmax = calculate_qmin_qmax(bits, quant_dtype)
                setting.bits = bits
                setting.qmin = qmin
                setting.qmax = qmax
                setattr(self, quant_type, setting)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_") or name in QuantType:
            super().__setattr__(name, val)
        else:
            self._extra_layer_setting[name] = val

    def __getattr__(self, name):
        if name == "_extra_layer_setting" or name not in self._extra_layer_setting:
            raise AttributeError("Cannot find {} in LayerQuantSetting!".format(name))
        return self._extra_layer_setting[name]

    @staticmethod
    def parse_optional_config(config, quant_type, target):
        def get_config(config, quant_type, target):
            if not config.get(target):
                return None

            if isinstance(config[target], dict):
                return config[target].get(quant_type)
            else:
                return config[target]

        default_val = quant_default_settings[quant_type].get(target, None)
        config_val = get_config(config, quant_type, target)
        val = config_val if config_val else default_val
        return val


def set_quant_scheme_dtype(quant_type, new_scheme=None, new_dtype=None):
    # todo: remove this if we convert string config to enum type.
    if isinstance(quant_type, str):
        assert quant_type in QuantType, "Wrong quant_type"
    if isinstance(new_scheme, str):
        assert new_scheme in QuantScheme, "Wrong quant_scheme"
    if isinstance(new_dtype, str):
        assert new_dtype in QuantDtype, "Wrong quant_dtype"

    # TODO: It is not a good idea to directly modify global settings. A better choice is
    # making this function an attribute function of Quantizer and call this function after
    # the quantizer is initialized. However, within current framework of quantization, if
    # we want to modify the dtype & scheme when the quantizer is initialized, we must do
    # some other things (like changing the shapes of scales and zero_points and other quantization
    # information in the subclass).
    global quant_default_settings
    if new_scheme is not None:
        quant_default_settings[quant_type]['quant_scheme'] = new_scheme
    if new_dtype is not None:
        quant_default_settings[quant_type]['quant_dtype'] = new_dtype
    return
