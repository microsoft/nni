# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

from abc import abstractmethod
from collections import Iterable

import torch
from torch import nn
from torch.nn import functional
from nni.networkmorphism_tuner.utils import Constant


class AvgPool(nn.Module):
    '''AvgPool Module.
    '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input_tensor):
        pass


class GlobalAvgPool1d(AvgPool):
    '''GlobalAvgPool1d Module.
    '''
    def forward(self, input_tensor):
        return functional.avg_pool1d(input_tensor, input_tensor.size()[2:]).view(
            input_tensor.size()[:2]
        )


class GlobalAvgPool2d(AvgPool):
    '''GlobalAvgPool2d Module.
    '''
    def forward(self, input_tensor):
        return functional.avg_pool2d(input_tensor, input_tensor.size()[2:]).view(
            input_tensor.size()[:2]
        )


class GlobalAvgPool3d(AvgPool):
    '''GlobalAvgPool3d Module.
    '''
    def forward(self, input_tensor):
        return functional.avg_pool3d(input_tensor, input_tensor.size()[2:]).view(
            input_tensor.size()[:2]
        )


class StubLayer:
    '''StubLayer Module. Base Module.
    '''
    def __init__(self, input_node=None, output_node=None):
        self.input = input_node
        self.output = output_node
        self.weights = None

    def build(self, shape):
        '''build shape.
        '''
        pass

    def set_weights(self, weights):
        '''set weights.
        '''
        self.weights = weights

    def import_weights(self, torch_layer):
        '''import weights.
        '''
        pass

    def import_weights_keras(self, keras_layer):
        '''import weights from keras layer.
        '''
        pass

    def export_weights(self, torch_layer):
        '''export weights.
        '''
        pass

    def export_weights_keras(self, keras_layer):
        '''export weights to keras layer.
        '''
        pass

    def get_weights(self):
        '''get weights.
        '''
        return self.weights

    def size(self):
        '''size().
        '''
        return 0

    @property
    def output_shape(self):
        '''output shape.
        '''
        return self.input.shape

    def to_real_layer(self):
        '''to real layer.
        '''
        pass

    def __str__(self):
        '''str() function to print.
        '''
        return type(self).__name__[4:]


class StubWeightBiasLayer(StubLayer):
    '''StubWeightBiasLayer Module to set the bias.
    '''
    def import_weights(self, torch_layer):
        self.set_weights(
            (torch_layer.weight.data.cpu().numpy(), torch_layer.bias.data.cpu().numpy())
        )

    def import_weights_keras(self, keras_layer):
        self.set_weights(keras_layer.get_weights())

    def export_weights(self, torch_layer):
        torch_layer.weight.data = torch.Tensor(self.weights[0])
        torch_layer.bias.data = torch.Tensor(self.weights[1])

    def export_weights_keras(self, keras_layer):
        keras_layer.set_weights(self.weights)


class StubBatchNormalization(StubWeightBiasLayer):
    '''StubBatchNormalization Module. Batch Norm.
    '''
    def __init__(self, num_features, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.num_features = num_features

    def import_weights(self, torch_layer):
        self.set_weights(
            (
                torch_layer.weight.data.cpu().numpy(),
                torch_layer.bias.data.cpu().numpy(),
                torch_layer.running_mean.cpu().numpy(),
                torch_layer.running_var.cpu().numpy(),
            )
        )

    def export_weights(self, torch_layer):
        torch_layer.weight.data = torch.Tensor(self.weights[0])
        torch_layer.bias.data = torch.Tensor(self.weights[1])
        torch_layer.running_mean = torch.Tensor(self.weights[2])
        torch_layer.running_var = torch.Tensor(self.weights[3])

    def size(self):
        return self.num_features * 4

    @abstractmethod
    def to_real_layer(self):
        pass


class StubBatchNormalization1d(StubBatchNormalization):
    '''StubBatchNormalization1d Module.
    '''
    def to_real_layer(self):
        return torch.nn.BatchNorm1d(self.num_features)


class StubBatchNormalization2d(StubBatchNormalization):
    '''StubBatchNormalization2d Module.
    '''
    def to_real_layer(self):
        return torch.nn.BatchNorm2d(self.num_features)


class StubBatchNormalization3d(StubBatchNormalization):
    '''StubBatchNormalization3d Module.
    '''
    def to_real_layer(self):
        return torch.nn.BatchNorm3d(self.num_features)


class StubDense(StubWeightBiasLayer):
    '''StubDense Module. Linear.
    '''
    def __init__(self, input_units, units, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.input_units = input_units
        self.units = units

    @property
    def output_shape(self):
        return (self.units,)

    def import_weights_keras(self, keras_layer):
        self.set_weights((keras_layer.get_weights()[0].T, keras_layer.get_weights()[1]))

    def export_weights_keras(self, keras_layer):
        keras_layer.set_weights((self.weights[0].T, self.weights[1]))

    def size(self):
        return self.input_units * self.units + self.units

    def to_real_layer(self):
        return torch.nn.Linear(self.input_units, self.units)


class StubConv(StubWeightBiasLayer):
    '''StubConv Module. Conv.
    '''
    def __init__(self, input_channel, filters, kernel_size, stride=1, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.input_channel = input_channel
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = int(self.kernel_size / 2)

    @property
    def output_shape(self):
        ret = list(self.input.shape[:-1])
        for index, dim in enumerate(ret):
            ret[index] = (
                int((dim + 2 * self.padding - self.kernel_size) / self.stride) + 1
            )
        ret = ret + [self.filters]
        return tuple(ret)

    def import_weights_keras(self, keras_layer):
        self.set_weights((keras_layer.get_weights()[0].T, keras_layer.get_weights()[1]))

    def export_weights_keras(self, keras_layer):
        keras_layer.set_weights((self.weights[0].T, self.weights[1]))

    def size(self):
        return (self.input_channel * self.kernel_size * self.kernel_size + 1) * self.filters

    @abstractmethod
    def to_real_layer(self):
        pass

    def __str__(self):
        return (
            super().__str__()
            + "("
            + ", ".join(
                str(item)
                for item in [
                    self.input_channel,
                    self.filters,
                    self.kernel_size,
                    self.stride,
                ]
            )
            + ")"
        )


class StubConv1d(StubConv):
    '''StubConv1d Module.
    '''
    def to_real_layer(self):
        return torch.nn.Conv1d(
            self.input_channel,
            self.filters,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


class StubConv2d(StubConv):
    '''StubConv2d Module.
    '''
    def to_real_layer(self):
        return torch.nn.Conv2d(
            self.input_channel,
            self.filters,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


class StubConv3d(StubConv):
    '''StubConv3d Module.
    '''
    def to_real_layer(self):
        return torch.nn.Conv3d(
            self.input_channel,
            self.filters,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


class StubAggregateLayer(StubLayer):
    '''StubAggregateLayer Module.
    '''
    def __init__(self, input_nodes=None, output_node=None):
        if input_nodes is None:
            input_nodes = []
        super().__init__(input_nodes, output_node)


class StubConcatenate(StubAggregateLayer):
    '''StubConcatenate Module.
    '''
    @property
    def output_shape(self):
        ret = 0
        for current_input in self.input:
            ret += current_input.shape[-1]
        ret = self.input[0].shape[:-1] + (ret,)
        return ret

    def to_real_layer(self):
        return TorchConcatenate()


class StubAdd(StubAggregateLayer):
    '''StubAdd Module.
    '''
    @property
    def output_shape(self):
        return self.input[0].shape

    def to_real_layer(self):
        return TorchAdd()


class StubFlatten(StubLayer):
    '''StubFlatten Module.
    '''
    @property
    def output_shape(self):
        ret = 1
        for dim in self.input.shape:
            ret *= dim
        return (ret,)

    def to_real_layer(self):
        return TorchFlatten()


class StubReLU(StubLayer):
    '''StubReLU Module.
    '''
    def to_real_layer(self):
        return torch.nn.ReLU()


class StubSoftmax(StubLayer):
    '''StubSoftmax Module.
    '''
    def to_real_layer(self):
        return torch.nn.LogSoftmax(dim=1)


class StubDropout(StubLayer):
    '''StubDropout Module.
    '''
    def __init__(self, rate, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.rate = rate

    @abstractmethod
    def to_real_layer(self):
        pass


class StubDropout1d(StubDropout):
    '''StubDropout1d Module.
    '''
    def to_real_layer(self):
        return torch.nn.Dropout(self.rate)


class StubDropout2d(StubDropout):
    '''StubDropout2d Module.
    '''
    def to_real_layer(self):
        return torch.nn.Dropout2d(self.rate)


class StubDropout3d(StubDropout):
    '''StubDropout3d Module.
    '''
    def to_real_layer(self):
        return torch.nn.Dropout3d(self.rate)


class StubInput(StubLayer):
    '''StubInput Module.
    '''
    def __init__(self, input_node=None, output_node=None):
        super().__init__(input_node, output_node)


class StubPooling(StubLayer):
    '''StubPooling Module.
    '''

    def __init__(self,
                 kernel_size=None,
                 stride=None,
                 padding=0,
                 input_node=None,
                 output_node=None):
        super().__init__(input_node, output_node)
        self.kernel_size = (
            kernel_size if kernel_size is not None else Constant.POOLING_KERNEL_SIZE
        )
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding

    @property
    def output_shape(self):
        ret = tuple()
        for dim in self.input.shape[:-1]:
            ret = ret + (max(int(dim / self.kernel_size), 1),)
        ret = ret + (self.input.shape[-1],)
        return ret

    @abstractmethod
    def to_real_layer(self):
        pass


class StubPooling1d(StubPooling):
    '''StubPooling1d Module.
    '''

    def to_real_layer(self):
        return torch.nn.MaxPool1d(self.kernel_size, stride=self.stride)


class StubPooling2d(StubPooling):
    '''StubPooling2d Module.
    '''
    def to_real_layer(self):
        return torch.nn.MaxPool2d(self.kernel_size, stride=self.stride)


class StubPooling3d(StubPooling):
    '''StubPooling3d Module.
    '''
    def to_real_layer(self):
        return torch.nn.MaxPool3d(self.kernel_size, stride=self.stride)


class StubGlobalPooling(StubLayer):
    '''StubGlobalPooling Module.
    '''
    def __init__(self, input_node=None, output_node=None):
        super().__init__(input_node, output_node)

    @property
    def output_shape(self):
        return (self.input.shape[-1],)

    @abstractmethod
    def to_real_layer(self):
        pass


class StubGlobalPooling1d(StubGlobalPooling):
    '''StubGlobalPooling1d Module.
    '''
    def to_real_layer(self):
        return GlobalAvgPool1d()


class StubGlobalPooling2d(StubGlobalPooling):
    '''StubGlobalPooling2d Module.
    '''
    def to_real_layer(self):
        return GlobalAvgPool2d()


class StubGlobalPooling3d(StubGlobalPooling):
    '''StubGlobalPooling3d Module.
    '''
    def to_real_layer(self):
        return GlobalAvgPool3d()


class TorchConcatenate(nn.Module):
    '''TorchConcatenate Module.
    '''
    def forward(self, input_list):
        return torch.cat(input_list, dim=1)


class TorchAdd(nn.Module):
    '''TorchAdd Module.
    '''
    def forward(self, input_list):
        return input_list[0] + input_list[1]


class TorchFlatten(nn.Module):
    '''TorchFlatten Module.
    '''
    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

def keras_dropout(layer, rate):
    '''keras dropout layer.
    '''

    from keras import layers

    input_dim = len(layer.input.shape)
    if input_dim == 2:
        return layers.SpatialDropout1D(rate)
    elif input_dim == 3:
        return layers.SpatialDropout2D(rate)
    elif input_dim == 4:
        return layers.SpatialDropout3D(rate)
    else:
        return layers.Dropout(rate)


def to_real_keras_layer(layer):
    ''' real keras layer.
    '''
    from keras import layers

    if is_layer(layer, "Dense"):
        return layers.Dense(layer.units, input_shape=(layer.input_units,))
    if is_layer(layer, "Conv"):
        return layers.Conv2D(
            layer.filters,
            layer.kernel_size,
            input_shape=layer.input.shape,
            padding="same",
        )  # padding
    if is_layer(layer, "Pooling"):
        return layers.MaxPool2D(2)
    if is_layer(layer, "BatchNormalization"):
        return layers.BatchNormalization(input_shape=layer.input.shape)
    if is_layer(layer, "Concatenate"):
        return layers.Concatenate()
    if is_layer(layer, "Add"):
        return layers.Add()
    if is_layer(layer, "Dropout"):
        return keras_dropout(layer, layer.rate)
    if is_layer(layer, "ReLU"):
        return layers.Activation("relu")
    if is_layer(layer, "Softmax"):
        return layers.Activation("softmax")
    if is_layer(layer, "Flatten"):
        return layers.Flatten()
    if is_layer(layer, "GlobalAveragePooling"):
        return layers.GlobalAveragePooling2D()


def is_layer(layer, layer_type):
    '''judge the layer type.
    Returns:
        boolean -- True or False
    '''

    if layer_type == "Input":
        return isinstance(layer, StubInput)
    elif layer_type == "Conv":
        return isinstance(layer, StubConv)
    elif layer_type == "Dense":
        return isinstance(layer, (StubDense,))
    elif layer_type == "BatchNormalization":
        return isinstance(layer, (StubBatchNormalization,))
    elif layer_type == "Concatenate":
        return isinstance(layer, (StubConcatenate,))
    elif layer_type == "Add":
        return isinstance(layer, (StubAdd,))
    elif layer_type == "Pooling":
        return isinstance(layer, StubPooling)
    elif layer_type == "Dropout":
        return isinstance(layer, (StubDropout,))
    elif layer_type == "Softmax":
        return isinstance(layer, (StubSoftmax,))
    elif layer_type == "ReLU":
        return isinstance(layer, (StubReLU,))
    elif layer_type == "Flatten":
        return isinstance(layer, (StubFlatten,))
    elif layer_type == "GlobalAveragePooling":
        return isinstance(layer, StubGlobalPooling)


def layer_description_extractor(layer, node_to_id):
    '''get layer description.
    '''

    layer_input = layer.input
    layer_output = layer.output
    if layer_input is not None:
        if isinstance(layer_input, Iterable):
            layer_input = list(map(lambda x: node_to_id[x], layer_input))
        else:
            layer_input = node_to_id[layer_input]

    if layer_output is not None:
        layer_output = node_to_id[layer_output]

    if isinstance(layer, StubConv):
        return (
            type(layer).__name__,
            layer_input,
            layer_output,
            layer.input_channel,
            layer.filters,
            layer.kernel_size,
            layer.stride,
            layer.padding,
        )
    elif isinstance(layer, (StubDense,)):
        return [
            type(layer).__name__,
            layer_input,
            layer_output,
            layer.input_units,
            layer.units,
        ]
    elif isinstance(layer, (StubBatchNormalization,)):
        return (type(layer).__name__, layer_input, layer_output, layer.num_features)
    elif isinstance(layer, (StubDropout,)):
        return (type(layer).__name__, layer_input, layer_output, layer.rate)
    elif isinstance(layer, StubPooling):
        return (
            type(layer).__name__,
            layer_input,
            layer_output,
            layer.kernel_size,
            layer.stride,
            layer.padding,
        )
    else:
        return (type(layer).__name__, layer_input, layer_output)


def layer_description_builder(layer_information, id_to_node):
    '''build layer from description.
    '''
    # pylint: disable=W0123
    layer_type = layer_information[0]

    layer_input_ids = layer_information[1]
    if isinstance(layer_input_ids, Iterable):
        layer_input = list(map(lambda x: id_to_node[x], layer_input_ids))
    else:
        layer_input = id_to_node[layer_input_ids]
    layer_output = id_to_node[layer_information[2]]
    if layer_type.startswith("StubConv"):
        input_channel = layer_information[3]
        filters = layer_information[4]
        kernel_size = layer_information[5]
        stride = layer_information[6]
        return eval(layer_type)(
            input_channel, filters, kernel_size, stride, layer_input, layer_output
        )
    elif layer_type.startswith("StubDense"):
        input_units = layer_information[3]
        units = layer_information[4]
        return eval(layer_type)(input_units, units, layer_input, layer_output)
    elif layer_type.startswith("StubBatchNormalization"):
        num_features = layer_information[3]
        return eval(layer_type)(num_features, layer_input, layer_output)
    elif layer_type.startswith("StubDropout"):
        rate = layer_information[3]
        return eval(layer_type)(rate, layer_input, layer_output)
    elif layer_type.startswith("StubPooling"):
        kernel_size = layer_information[3]
        stride = layer_information[4]
        padding = layer_information[5]
        return eval(layer_type)(kernel_size, stride, padding, layer_input, layer_output)
    else:
        return eval(layer_type)(layer_input, layer_output)


def layer_width(layer):
    '''get layer width.
    '''

    if is_layer(layer, "Dense"):
        return layer.units
    if is_layer(layer, "Conv"):
        return layer.filters
    raise TypeError("The layer should be either Dense or Conv layer.")


def set_torch_weight_to_stub(torch_layer, stub_layer):
    stub_layer.import_weights(torch_layer)


def set_keras_weight_to_stub(keras_layer, stub_layer):
    stub_layer.import_weights_keras(keras_layer)


def set_stub_weight_to_torch(stub_layer, torch_layer):
    stub_layer.export_weights(torch_layer)


def set_stub_weight_to_keras(stub_layer, keras_layer):
    stub_layer.export_weights_keras(keras_layer)


def get_conv_class(n_dim):
    conv_class_list = [StubConv1d, StubConv2d, StubConv3d]
    return conv_class_list[n_dim - 1]


def get_dropout_class(n_dim):
    dropout_class_list = [StubDropout1d, StubDropout2d, StubDropout3d]
    return dropout_class_list[n_dim - 1]


def get_global_avg_pooling_class(n_dim):
    global_avg_pooling_class_list = [
        StubGlobalPooling1d,
        StubGlobalPooling2d,
        StubGlobalPooling3d,
    ]
    return global_avg_pooling_class_list[n_dim - 1]


def get_pooling_class(n_dim):
    pooling_class_list = [StubPooling1d, StubPooling2d, StubPooling3d]
    return pooling_class_list[n_dim - 1]


def get_batch_norm_class(n_dim):
    batch_norm_class_list = [
        StubBatchNormalization1d,
        StubBatchNormalization2d,
        StubBatchNormalization3d,
    ]
    return batch_norm_class_list[n_dim - 1]


def get_n_dim(layer):
    if isinstance(layer, (
            StubConv1d,
            StubDropout1d,
            StubGlobalPooling1d,
            StubPooling1d,
            StubBatchNormalization1d,
    )):
        return 1
    if isinstance(layer, (
            StubConv2d,
            StubDropout2d,
            StubGlobalPooling2d,
            StubPooling2d,
            StubBatchNormalization2d,
    )):
        return 2
    if isinstance(layer, (
            StubConv3d,
            StubDropout3d,
            StubGlobalPooling3d,
            StubPooling3d,
            StubBatchNormalization3d,
    )):
        return 3
    return -1
