# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
Graph is customed-define class, this module contains related class and function about graph.
'''


import copy
import json
import random
from enum import Enum, unique

@unique
class LayerType(Enum):
    '''
    Layer type
    '''
    attention = 0
    self_attention = 1
    rnn = 2
    input = 3
    output = 4

class Layer(object):
    '''
    Layer class, which contains the information of graph.
    '''
    def __init__(self, graph_type, inputs=None, output=None, size=None):
        self.input = inputs if inputs is not None else []
        self.output = output if output is not None else []
        self.graph_type = graph_type
        self.is_delete = False
        self.size = size
        if graph_type == LayerType.attention.value:
            self.input_size = 2
            self.output_size = 1
        elif graph_type == LayerType.rnn.value:
            self.input_size = 1
            self.output_size = 1
        elif graph_type == LayerType.self_attention.value:
            self.input_size = 1
            self.output_size = 1
        elif graph_type == LayerType.input.value:
            self.input_size = 0
            self.output_size = 1
        elif graph_type == LayerType.output.value:
            self.input_size = 1
            self.output_size = 0
        else:
            print(graph_type)
    def set_size(self, graph_id, size):
        '''
        Set size.
        '''
        if self.graph_type == LayerType.attention.value:
            if self.input[0] == graph_id:
                self.size = size
        if self.graph_type == LayerType.rnn.value:
            self.size = size
        if self.graph_type == LayerType.self_attention.value:
            self.size = size
        if self.graph_type == LayerType.output.value:
            if self.size != size:
                return False
        return True

    def clear_size(self):
        '''
        Clear size
        '''
        if self.graph_type == LayerType.attention.value or \
            LayerType.rnn.value or LayerType.self_attention.value:
            self.size = None

    def __str__(self):
        return 'input:' + str(self.input) + ' output:' + str(self.output) + ' type:' + str(
            self.graph_type) + ' is_delete:' + str(self.is_delete) + ' size:' + str(self.size)

def graph_dumps(graph):
    '''
    Dump the graph.
    '''
    return json.dumps(graph, default=lambda obj: obj.__dict__)

def graph_loads(graph_json):
    '''
    Load graph
    '''
    layers = []
    for layer in graph_json['layers']:
        layer_info = Layer(layer['type'], layer['input'], layer['output'], layer['size'])
        layer_info.is_delete = layer['is_delete']
        layers.append(layer_info)
    graph = Graph(graph_json['max_layer_num'], [], [], [])
    graph.layers = layers
    return graph

class Graph(object):
    '''
    Customed Graph class.
    '''
    def __init__(self, max_layer_num, inputs, output, hide):
        self.layers = []
        self.max_layer_num = max_layer_num

        for layer in inputs:
            self.layers.append(layer)
        for layer in output:
            self.layers.append(layer)
        if hide is not None:
            for layer in hide:
                self.layers.append(layer)
        assert self.is_legal()

    def is_topology(self, layers=None):
        '''
        valid the topology
        '''
        if layers is None:
            layers = self.layers
        layers_nodle = []
        result = []
        for i, layer in enumerate(layers):
            if layer.is_delete is False:
                layers_nodle.append(i)
        while True:
            flag_break = True
            layers_toremove = []
            for layer1 in layers_nodle:
                flag_arrive = True
                for layer2 in layers[layer1].input:
                    if layer2 in layers_nodle:
                        flag_arrive = False
                if flag_arrive is True:
                    for layer2 in layers[layer1].output:
                        # Size is error
                        if layers[layer2].set_size(layer1, layers[layer1].size) is False:
                            return False
                    layers_toremove.append(layer1)
                    result.append(layer1)
                    flag_break = False
            for layer in layers_toremove:
                layers_nodle.remove(layer)
            result.append('|')
            if flag_break:
                break
        # There is loop in graph || some layers can't to arrive
        if layers_nodle:
            return False
        return result

    def layer_num(self, layers=None):
        '''
        Reutn number of layer.
        '''
        if layers is None:
            layers = self.layers
        layer_num = 0
        for layer in layers:
            if layer.is_delete is False and layer.graph_type != LayerType.input.value\
                and layer.graph_type != LayerType.output.value:
                layer_num += 1
        return layer_num

    def is_legal(self, layers=None):
        '''
        Judge whether is legal for layers
        '''
        if layers is None:
            layers = self.layers

        for layer in layers:
            if layer.is_delete is False:
                if len(layer.input) != layer.input_size:
                    return False
                if len(layer.output) < layer.output_size:
                    return False

        # layer_num <= max_layer_num
        if self.layer_num(layers) > self.max_layer_num:
            return False

        # There is loop in graph || some layers can't to arrive
        if self.is_topology(layers) is False:
            return False

        return True

    def mutation(self, only_add=False):
        '''
        Mutation for a graph
        '''
        types = []
        if self.layer_num() < self.max_layer_num:
            types.append(0)
            types.append(1)
        if self.layer_num() > 5 and only_add is False:
            types.append(2)
            types.append(3)
        # 0 : add a layer , delete a edge
        # 1 : add a layer , change a edge
        # 2 : delete a layer, delete a edge
        # 3 : delete a layer, change a edge
        graph_type = random.choice(types)
        layer_type = random.choice([LayerType.attention.value,\
            LayerType.self_attention.value, LayerType.rnn.value])
        layers = copy.deepcopy(self.layers)
        cnt_try = 0
        while True:
            layers_in = []
            layers_out = []
            layers_del = []
            for i, layer in enumerate(layers):
                if layer.is_delete is False:
                    if layer.graph_type != LayerType.output.value:
                        layers_in.append(i)
                    if layer.graph_type != LayerType.input.value:
                        layers_out.append(i)
                    if layer.graph_type != LayerType.output.value\
                            and layer.graph_type != LayerType.input.value:
                        layers_del.append(i)
            if graph_type <= 1:
                new_id = len(layers)
                out = random.choice(layers_out)
                inputs = []
                output = [out]
                pos = random.randint(0, len(layers[out].input) - 1)
                last_in = layers[out].input[pos]
                layers[out].input[pos] = new_id
                if graph_type == 0:
                    layers[last_in].output.remove(out)
                if graph_type == 1:
                    layers[last_in].output.remove(out)
                    layers[last_in].output.append(new_id)
                    inputs = [last_in]
                lay = Layer(graph_type=layer_type, inputs=inputs, output=output)
                while len(inputs) < lay.input_size:
                    layer1 = random.choice(layers_in)
                    inputs.append(layer1)
                    layers[layer1].output.append(new_id)
                lay.input = inputs
                layers.append(lay)
            else:
                layer1 = random.choice(layers_del)
                for layer2 in layers[layer1].output:
                    layers[layer2].input.remove(layer1)
                    if graph_type == 2:
                        random_in = random.choice(layers_in)
                    else:
                        random_in = random.choice(layers[layer1].input)
                    layers[layer2].input.append(random_in)
                    layers[random_in].output.append(layer2)
                for layer2 in layers[layer1].input:
                    layers[layer2].output.remove(layer1)
                layers[layer1].is_delete = True

            if self.is_legal(layers):
                self.layers = layers
                break
            else:
                layers = copy.deepcopy(self.layers)
                cnt_try += 1

    def __str__(self):
        info = ""
        for l_id, layer in enumerate(self.layers):
            if layer.is_delete is False:
                info += 'id:%d ' % l_id + str(layer) + '\n'
        return info
