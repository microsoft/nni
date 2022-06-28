# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from nni.algorithms.compression.v2.pytorch.utils.scaling import Scaling


def test_scaling():
    data = torch.tensor([_ for _ in range(100)]).reshape(10, 10)

    scaler = Scaling([5], kernel_padding_mode='front')
    shrinked_data = scaler.shrink(data)
    assert list(shrinked_data.shape) == [10, 2]
    expanded_data = scaler.expand(data, [10, 50])
    assert list(expanded_data.shape) == [10, 50]

    scaler = Scaling([5, 5], kernel_padding_mode='back')
    shrinked_data = scaler.shrink(data)
    assert list(shrinked_data.shape) == [2, 2]
    expanded_data = scaler.expand(data, [50, 50, 10])
    assert list(expanded_data.shape) == [50, 50, 10]

    scaler.validate([10, 10, 10])


if __name__ == '__main__':
    test_scaling()
