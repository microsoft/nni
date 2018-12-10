# -*- coding: utf-8 -*-

import copy
import json
import random
from enum import Enum, unique

@unique
class LayerType(Enum):
    attention = 0
    self_attention = 1
    rnn = 2
    input = 3
    output = 4

class Layer(object):
    def __init__(self, type, input=None, output=None, size=None):
        self.input = input if input is not None else []
        self.output = output if output is not None else []
        self.type = type
        self.is_delete = False
        self.size = size
        if type == LayerType.attention.value:
            self.input_size = 2
            self.output_size = 1
        elif type == LayerType.rnn.value:
            self.input_size = 1
            self.output_size = 1
        elif type == LayerType.self_attention.value:
            self.input_size = 1
            self.output_size = 1
        elif type == LayerType.input.value:
            self.input_size = 0
            self.output_size = 1
        elif type == LayerType.output.value:
            self.input_size = 1
            self.output_size = 0
        else:
            print(type)
    def set_size(self, id, size):
        if self.type == LayerType.attention.value:
            if self.input[0] == id:
                self.size = size
        if self.type == LayerType.rnn.value:
            self.size = size
        if self.type == LayerType.self_attention.value:
            self.size = size
        if self.type == LayerType.output.value:
            if self.size != size:
                return False
        return True

    def clear_size(self):
        if self.type == LayerType.attention.value or LayerType.rnn.value or LayerType.self_attention.value:
            self.size = None

    def __str__(self):
        return 'input:' + str(self.input) + ' output:' + str(self.output) + ' type:' + str(
            self.type) + ' is_delete:' + str(self.is_delete) + ' size:' + str(self.size)

def graph_dumps(graph):
    return json.dumps(graph, default=lambda obj: obj.__dict__)

def graph_loads(js):
    layers = []
    for layer in js['layers']:
        p = Layer(layer['type'],layer['input'],layer['output'],layer['size'])
        p.is_delete = layer['is_delete']
        layers.append(p)
    graph = Graph(js['max_layer_num'],[], [], [])
    graph.layers = layers
    return graph

class Graph(object):
    def __init__(self, max_layer_num, input, output, hide):
        self.layers = []
        self.max_layer_num = max_layer_num

        for layer in input:
            self.layers.append(layer)
        for layer in output:
            self.layers.append(layer)
        if hide is not None:
            for layer in hide:
                self.layers.append(layer)
        assert self.is_legal()

    def is_topology(self, layers=None):
        if layers == None:
            layers = self.layers
        layers_nodle = []
        xx = []
        for i in range(len(layers)):
            if layers[i].is_delete == False:
                layers_nodle.append(i)
        while True:
            flag_break = True
            layers_toremove = []
            for layer1 in layers_nodle:
                flag_arrive = True
                for layer2 in layers[layer1].input:
                    if layer2 in layers_nodle:
                        flag_arrive = False
                if flag_arrive == True:
                    for layer2 in layers[layer1].output:
                        if layers[layer2].set_size(layer1, layers[layer1].size) == False:  # Size is error
                            return False
                    layers_toremove.append(layer1)
                    xx.append(layer1)
                    flag_break = False
            for layer in layers_toremove:
                layers_nodle.remove(layer)
            xx.append('|')
            if flag_break == True:
                break
        if len(layers_nodle) > 0:  # There is loop in graph || some layers can't to arrive
            return False
        return xx

    def layer_num(self, layers=None):
        if layers == None:
            layers = self.layers
        layer_num = 0
        for layer in layers:
            if layer.is_delete == False and layer.type != LayerType.input.value and layer.type != LayerType.output.value:
                layer_num += 1
        return layer_num

    def is_legal(self, layers=None):
        if layers == None:
            layers = self.layers

        for layer in layers:
            if layer.is_delete == False:
                if len(layer.input) != layer.input_size:
                    return False
                if len(layer.output) < layer.output_size:
                    return False

        # layer_num <= max_layer_num
        if self.layer_num(layers) > self.max_layer_num:
            return False

        if self.is_topology(layers) == False:  # There is loop in graph || some layers can't to arrive
            return False

        return True

    def mutation(self, only_add=False):
        types = []
        if self.layer_num() < self.max_layer_num:
            types.append(0)
            types.append(1)
        if self.layer_num() > 0:
            types.append(2)
            types.append(3)
        # 0 : add a layer , delete a edge
        # 1 : add a layer , change a edge
        # 2 : delete a layer, delete a edge
        # 3 : delete a layer, change a edge
        type = random.choice(types)
        layer_type = random.choice([LayerType.attention.value, LayerType.self_attention.value, LayerType.rnn.value])
        layers = copy.deepcopy(self.layers)
        cnt_try = 0
        while True:
            layers_in = []
            layers_out = []
            layers_del = []
            for layer1 in range(len(layers)):
                layer = layers[layer1]
                if layer.is_delete == False:
                    if layer.type != LayerType.output.value:
                        layers_in.append(layer1)
                    if layer.type != LayerType.input.value:
                        layers_out.append(layer1)
                    if layer.type != LayerType.output.value and layer.type != LayerType.input.value:
                        layers_del.append(layer1)
            if type <= 1:
                new_id = len(layers)
                out = random.choice(layers_out)
                input = []
                output = [out]
                pos = random.randint(0, len(layers[out].input) - 1)
                last_in = layers[out].input[pos]
                layers[out].input[pos] = new_id
                if type == 0:
                    layers[last_in].output.remove(out)
                if type == 1:
                    layers[last_in].output.remove(out)
                    layers[last_in].output.append(new_id)
                    input = [last_in]
                lay = Layer(type=layer_type, input=input, output=output)
                while len(input) < lay.input_size:
                    layer1 = random.choice(layers_in)
                    input.append(layer1)
                    layers[layer1].output.append(new_id)
                lay.input = input
                layers.append(lay)
            else:
                layer1 = random.choice(layers_del)
                for layer2 in layers[layer1].output:
                    layers[layer2].input.remove(layer1)
                    if type == 2:
                        v2 = random.choice(layers_in)
                    else:
                        v2 = random.choice(layers[layer1].input)
                    layers[layer2].input.append(v2)
                    layers[v2].output.append(layer2)
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
        for id, layer in enumerate(self.layers):
            if layer.is_delete == False:
                info += 'id:%d ' % id + str(layer) + '\n'
        return info

if __name__ == '__main__':
    graph = Graph(10,
                  input=[Layer(LayerType.input.value, output=[4, 5], size='x'), Layer(LayerType.input.value, output=[4, 5], size='y')],
                  output=[Layer(LayerType.output.value, input=[4], size='x'), Layer(LayerType.output.value, input=[5], size='y')],
                  hide=[Layer(LayerType.attention.value, input=[0, 1], output=[2]), Layer(LayerType.attention.value, input=[1, 0], output=[3])])

    s = graph_dumps(graph)
    g = graph_loads(json.loads(s))
    print(g)
    print(s)

    s = '''{"count":2,"array":[{"input":%s,"output":{"output":0.7}}]}'''%s
    print(len(s))
    print(s)