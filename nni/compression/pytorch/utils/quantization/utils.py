# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from nni.common.version import TORCH_VERSION

from .literal import QuantDtype, QuantScheme, QuantType


def calculate_qmin_qmax(bits, dtype):
    if dtype == QuantDtype.INT:
        qmin, qmax = -2 ** (bits - 1) + 1, 2 ** (bits - 1) - 1
    elif dtype == QuantDtype.UINT:
        qmin, qmax = 0, 2 ** bits - 1
    else:
        raise TypeError("Wrong quantization dtype, please make sure it is one of 'int' and 'uint'.")
    return qmin, qmax


def get_bits_length(config, quant_type):
    if isinstance(config["quant_bits"], int):
        return config["quant_bits"]
    else:
        return config["quant_bits"].get(quant_type)


def get_target_dim(quant_type, quant_scheme):
    # for weight: c_out x c_in x (h) * (w)
    # for feature maps: batch * channel * (t) * h * w
    # other type is not supported for now
    default_idx = 0 if quant_type == QuantType.WEIGHT else 1
    if is_per_channel(quant_scheme):
        target_dim = default_idx
    else:
        target_dim = None
    return target_dim


def get_min_max_value(x, quant_type, quant_scheme):

    target_dim = get_target_dim(quant_type, quant_scheme)
    if target_dim is None:
        return torch.min(x), torch.max(x)

    indices = list(range(len(x.shape)))
    assert target_dim < len(indices), "target_dim needs to be less than the number of dim of the tensor"
    del indices[target_dim]

    if TORCH_VERSION > (1, 6):
        min_val = torch.amin(x, indices, keepdims=True)
        max_val = torch.amax(x, indices, keepdims=True)
    else:
        min_val = max_val = x
        for ind in indices:
            min_val = torch.min(min_val, dim=ind, keepdim=True)[0]
            max_val = torch.max(max_val, dim=ind, keepdim=True)[0]
    return min_val, max_val


def get_mean_value(x, target_dim=None):
    if target_dim is None:
        return torch.mean(x)

    indices = list(range(len(x.shape)))
    assert target_dim < len(indices), "target_dim needs to be less than the number of dim of the tensor"
    del indices[target_dim]

    mean_val = torch.mean(x, dim=indices, keepdim=True)
    return mean_val


def is_per_channel(quant_scheme):
    if quant_scheme in [QuantScheme.PER_CHANNEL_AFFINE, QuantScheme.PER_CHANNEL_SYMMETRIC]:
        return True
    else:
        return False


def get_quant_shape(shape, quant_type, quant_scheme):
    default_idx = 0 if quant_type == QuantType.WEIGHT else 1
    if is_per_channel(quant_scheme):
        quant_shape = [1 if idx != default_idx else s for idx, s in enumerate(shape)]
    else:
        quant_shape = [1]
    return quant_shape
