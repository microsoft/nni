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
from copy import deepcopy
from queue import Queue

import numpy as np
import onnx
from onnx import (IR_VERSION, AttributeProto, GraphProto, ModelProto,
                  NodeProto, TensorProto)

from nni.networkmorphism_tuner.graph import Graph
from nni.networkmorphism_tuner.layers import (StubDense, StubDropout1d,
                                              StubReLU, get_batch_norm_class,
                                              get_conv_class,
                                              get_dropout_class,
                                              get_global_avg_pooling_class,
                                              get_pooling_class)
from nni.networkmorphism_tuner.utils import Constant


class NetworkGenerator:
    def __init__(self, n_output_node, input_shape):
        self.n_output_node = n_output_node
        self.input_shape = input_shape

    @abstractmethod
    def generate(self, model_len, model_width):
        pass


class CnnGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super(CnnGenerator, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=Constant.MODEL_LEN, model_width=Constant.MODEL_WIDTH):
        # model = ModelProto()
        # model.ir_version = IR_VERSION

        pooling_len = int(model_len / 4)
        graph = Graph(self.input_shape, True)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        for i in range(model_len):
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            output_node_id = graph.add_layer(self.conv(temp_input_channel, model_width, kernel_size=3), output_node_id)
            output_node_id = graph.add_layer(self.batch_norm(model_width), output_node_id)
            temp_input_channel = model_width
            if pooling_len == 0 or ((i + 1) % pooling_len == 0 and i != model_len - 1):
                output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        output_node_id = graph.add_layer(self.dropout(Constant.CONV_DROPOUT_RATE), output_node_id)
        output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], model_width),
                                         output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        graph.add_layer(StubDense(model_width, self.n_output_node), output_node_id)
        return graph

class RnnGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super(RnnGenerator, self).__init__(n_output_node, input_shape)
        pass

    def generate(self, model_len, model_width):
        model = ModelProto()
        model.ir_version = IR_VERSION

        return model

class MlpGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super(MlpGenerator, self).__init__(n_output_node, input_shape)
        if len(self.input_shape) > 1:
            raise ValueError('The input dimension is too high.')

    def generate(self, model_len, model_width):
        model = ModelProto()
        model.ir_version = IR_VERSION
        return model
