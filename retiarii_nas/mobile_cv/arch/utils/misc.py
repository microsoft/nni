#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


def drop_connect_batch(inputs, drop_prob, training):
    """ Randomly drop batch during training """
    assert drop_prob < 1.0, f"Invalid drop_prob {drop_prob}"
    if not training or drop_prob == 0.0:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - drop_prob
    random_tensor = (
        torch.rand(
            [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
        )
        + keep_prob
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    # output = inputs * binary_tensor
    return output


def add_dropout(dropout_ratio):
    if dropout_ratio > 0:
        return torch.nn.Dropout(dropout_ratio)
    return None
