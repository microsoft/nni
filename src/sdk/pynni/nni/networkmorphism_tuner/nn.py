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
import numpy as np
from queue import Queue
from copy import deepcopy

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
        pass

    def generate(self, model_len, model_width):
        graph =[]

        return graph

class RnnGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super(RnnGenerator, self).__init__(n_output_node, input_shape)
        pass

    def generate(self, model_len, model_width):
        graph =[]

        return graph

class MlpGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super(MlpGenerator, self).__init__(n_output_node, input_shape)
        pass

    def generate(self, model_len, model_width):
        graph =[]
        return graph



class NetworkDescriptor:
    CONCAT_CONNECT = 'concat'
    ADD_CONNECT = 'add'

    def __init__(self):
        self.skip_connections = []
        self.conv_widths = []
        self.dense_widths = []

    @property
    def n_dense(self):
        return len(self.dense_widths)

    @property
    def n_conv(self):
        return len(self.conv_widths)

    def add_conv_width(self, width):
        self.conv_widths.append(width)

    def add_dense_width(self, width):
        self.dense_widths.append(width)

    def add_skip_connection(self, u, v, connection_type):
        if connection_type not in [self.CONCAT_CONNECT, self.ADD_CONNECT]:
            raise ValueError('connection_type should be NetworkDescriptor.CONCAT_CONNECT '
                             'or NetworkDescriptor.ADD_CONNECT.')
        self.skip_connections.append((u, v, connection_type))

    def to_json(self):
        skip_list = []
        for u, v, connection_type in self.skip_connections:
            skip_list.append({'from': u, 'to': v, 'type': connection_type})
        return {'node_list': self.conv_widths, 'skip_list': skip_list}

def is_layer(layer,layer_type):
    return 

class Node:
    def __init__(self, shape):
        self.shape = shape
