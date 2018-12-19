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

from nni.networkmorphism_tuner.graph import Graph
from nni.networkmorphism_tuner.layers import (StubDense, StubDropout1d,
                                              StubReLU, get_batch_norm_class,
                                              get_conv_class,
                                              get_dropout_class,
                                              get_global_avg_pooling_class,
                                              get_pooling_class)
from nni.networkmorphism_tuner.utils import Constant


class NetworkGenerator:
    """The base class for generating a network.
    It can be used to generate a CNN or Multi-Layer Perceptron.
    Attributes:
        n_output_node: Number of output nodes in the network.
        input_shape: A tuple to represent the input shape.
    """

    def __init__(self, n_output_node, input_shape):
        self.n_output_node = n_output_node
        self.input_shape = input_shape

    @abstractmethod
    def generate(self, model_len, model_width):
        pass


class CnnGenerator(NetworkGenerator):
    """A class to generate CNN.
    Attributes:
          n_dim: `len(self.input_shape) - 1`
          conv: A class that represents `(n_dim-1)` dimensional convolution.
          dropout: A class that represents `(n_dim-1)` dimensional dropout.
          global_avg_pooling: A class that represents `(n_dim-1)` dimensional Global Average Pooling.
          pooling: A class that represents `(n_dim-1)` dimensional pooling.
          batch_norm: A class that represents `(n_dim-1)` dimensional batch normalization.
    """

    def __init__(self, n_output_node, input_shape):
        super(CnnGenerator, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError("The input dimension is too high.")
        if len(self.input_shape) < 2:
            raise ValueError("The input dimension is too low.")
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):
        """Generates a CNN.
        Args:
            model_len: An integer. Number of convolutional layers.
            model_width: An integer. Number of filters for the convolutional layers.
        Returns:
            An instance of the class Graph. Represents the neural architecture graph of the generated model.
        """

        if model_len is None:
            model_len = Constant.MODEL_LEN
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        pooling_len = int(model_len / 4)
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        stride = 1
        for i in range(model_len):
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            output_node_id = graph.add_layer(
                self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id
            )
            output_node_id = graph.add_layer(
                self.conv(temp_input_channel, model_width, kernel_size=3, stride=stride),
                output_node_id,
            )
            temp_input_channel = model_width
            if pooling_len == 0 or ((i + 1) % pooling_len == 0 and i != model_len - 1):
                output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        output_node_id = graph.add_layer(
            self.dropout(Constant.CONV_DROPOUT_RATE), output_node_id
        )
        output_node_id = graph.add_layer(
            StubDense(graph.node_list[output_node_id].shape[0], model_width),
            output_node_id,
        )
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        graph.add_layer(StubDense(model_width, self.n_output_node), output_node_id)
        return graph


class MlpGenerator(NetworkGenerator):
    """A class to generate Multi-Layer Perceptron.
    """

    def __init__(self, n_output_node, input_shape):
        """Initialize the instance.
        Args:
            n_output_node: An integer. Number of output nodes in the network.
            input_shape: A tuple. Input shape of the network. If it is 1D, ensure the value is appended by a comma
                in the tuple.
        """
        super(MlpGenerator, self).__init__(n_output_node, input_shape)
        if len(self.input_shape) > 1:
            raise ValueError("The input dimension is too high.")

    def generate(self, model_len=None, model_width=None):
        """Generates a Multi-Layer Perceptron.
        Args:
            model_len: An integer. Number of hidden layers.
            model_width: An integer or a list of integers of length `model_len`. If it is a list, it represents the
                number of nodes in each hidden layer. If it is an integer, all hidden layers have nodes equal to this
                value.
        Returns:
            An instance of the class Graph. Represents the neural architecture graph of the generated model.
        """
        if model_len is None:
            model_len = Constant.MODEL_LEN
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        if isinstance(model_width, list) and not len(model_width) == model_len:
            raise ValueError("The length of 'model_width' does not match 'model_len'")
        elif isinstance(model_width, int):
            model_width = [model_width] * model_len

        graph = Graph(self.input_shape, False)
        output_node_id = 0
        n_nodes_prev_layer = self.input_shape[0]
        for width in model_width:
            output_node_id = graph.add_layer(
                StubDense(n_nodes_prev_layer, width), output_node_id
            )
            output_node_id = graph.add_layer(
                StubDropout1d(Constant.MLP_DROPOUT_RATE), output_node_id
            )
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            n_nodes_prev_layer = width

        graph.add_layer(StubDense(n_nodes_prev_layer, self.n_output_node), output_node_id)
        return graph
