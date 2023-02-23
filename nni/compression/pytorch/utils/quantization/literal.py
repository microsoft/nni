# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum, EnumMeta


class _QuantLiteralEnumMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)  # pylint: disable=no-value-for-parameter
        except ValueError:
            return False
        return True


class _QuantLiteralEnum(Enum, metaclass=_QuantLiteralEnumMeta):
    pass


class QuantScheme(str, _QuantLiteralEnum):
    PER_TENSOR_AFFINE = 'per_tensor_affine'
    PER_TENSOR_SYMMETRIC = 'per_tensor_symmetric'
    PER_CHANNEL_AFFINE = 'per_channel_affine'
    PER_CHANNEL_SYMMETRIC = 'per_channel_symmetric'


PER_CHANNEL_QUANT_SCHEME = [QuantScheme.PER_CHANNEL_AFFINE, QuantScheme.PER_CHANNEL_SYMMETRIC]


class QuantDtype(str, _QuantLiteralEnum):
    UINT = 'uint'
    INT = 'int'


class QuantType(str, _QuantLiteralEnum):
    INPUT = 'input'
    WEIGHT = 'weight'
    OUTPUT = 'output'

    def type_to_scale_zero_point_name(self):
        if self == QuantType.INPUT:
            return 'input_scale', 'input_zero_point'
        elif self == QuantType.WEIGHT:
            return 'weight_scale', 'weight_zero_point'
        elif self == QuantType.OUTPUT:
            return 'output_scale', 'output_zero_point'
        else:
            raise TypeError


# Just show each attribute's name, no practical effect
class QuantConfigLiteral(str, _QuantLiteralEnum):
    QUANT_SETTINGS = 'quant_settings'
    QUANT_SCHEME = 'quant_scheme'
    QUANT_DTYPE = 'quant_dtype'
    BITS = 'bits'
    QMIN = 'qmin'
    QMAX = 'qmax'
    INPUT_SCALE = 'input_scale'
    INPUT_ZERO_POINT = 'input_zero_point'
    OUTPUT_SCALE = 'output_scale'
    OUTPUT_ZERO_POINT = 'output_zero_point'
    WEIGHT_SCALE = 'weight_scale'
    WEIGHT_ZERO_POINT = 'weight_zero_point'


BN_FOLD_OP = ["Conv2d"]
BN_FOLD_TAG = 'BN_FOLD_TAG'
