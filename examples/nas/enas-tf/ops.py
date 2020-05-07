# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, MaxPool2D, ReLU


class StdConv(Model):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv = Sequential([
            Conv2D(C_out, kernel_size=1, strides=1, use_bias=False),
            BatchNormalization(trainable=False),
            ReLU()
        ])

    def call(self, x):
        return self.conv(x)


class PoolBranch(Model):
    def __init__(self, pool_type, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.preproc = StdConv(C_in, C_out)
        self.pool = Pool(pool_type, kernel_size, stride, padding)
        self.bn = BatchNormalization(trainable=affine)

    def call(self, x):
        out = self.preproc(x)
        out = self.pool(out)
        out = self.bn(out)
        return out


class SeparableConv(Model):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        # FIXME: padding, groups
        self.depthwise = Conv2D(C_in, kernel_size=kernel_size, strides=stride, use_bias=False)
        self.pointwise = Conv2D(C_out, kernel_size=1, use_bias=False)

    def call(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBranch(Model):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, separable):
        super().__init__()
        self.preproc = StdConv(C_in, C_out)
        if separable:
            self.conv = SeparableConv(C_out, C_out, kernel_size, stride, padding)
        else:
            self.conv = Conv2D(C_out, kernel_size, strides=stride)
        self.postproc = Sequential([
            BatchNormalization(trainable=False),
            ReLU()
        ])

    def call(self, x):
        out = self.preproc(x)
        out = self.conv(out)
        out = self.postproc(out)
        return out


class FactorizedReduce(Model):
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = Conv2D(C_out // 2, kernel_size=1, strides=2, use_bias=False)
        self.conv2 = Conv2D(C_out // 2, kernel_size=1, strides=2, use_bias=False)
        self.bn = BatchNormalization(trainable=affine)

    def call(self, x):
        out = tf.concat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], axis=1)
        out = self.bn(out)
        return out


class Pool(Model):
    def __init__(self, pool_type, kernel_size, stride, padding):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = MaxPool2D(pool_size=kernel_size, strides=stride)
        elif pool_type.lower() == 'avg':
            self.pool = AveragePooling2D(pool_size=kernel_size, strides=stride)
        else:
            raise ValueError()

    def call(self, x):
        return self.pool(x)


class SepConvBN(Model):
    def __init__(self, C_in, C_out, kernel_size, padding):
        super().__init__()
        self.relu = ReLU()
        self.conv = SeparableConv(C_in, C_out, kernel_size, 1, padding)
        self.bn = BatchNormalization(C_out, affine=True)

    def call(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class Identity(Model):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return x
