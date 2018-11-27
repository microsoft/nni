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

from copy import deepcopy
from operator import itemgetter
from random import randrange, sample

from nni.networkmorphism_tuner.utils import Constant
from nni.networkmorphism_tuner.layers import is_layer
from nni.networkmorphism_tuner.graph import NetworkDescriptor


def to_wider_graph(graph):
    weighted_layer_ids = graph.wide_layer_ids()
    weighted_layer_ids = list(
        filter(lambda x: graph.layer_list[x].output.shape[-1], weighted_layer_ids)
    )
    wider_layers = sample(weighted_layer_ids, 1)

    for layer_id in wider_layers:
        layer = graph.layer_list[layer_id]
        if is_layer(layer, "Conv"):
            n_add = layer.filters
        else:
            n_add = layer.units

        graph.to_wider_model(layer_id, n_add)
    return graph


def to_skip_connection_graph(graph):
    # The last conv layer cannot be widen since wider operator cannot be done over the two sides of flatten.
    weighted_layer_ids = graph.skip_connection_layer_ids()
    descriptor = graph.extract_descriptor()
    sorted_skips = sorted(descriptor.skip_connections, key=itemgetter(2, 0, 1))
    p = 0
    valid_connection = []
    for skip_type in sorted(
        [NetworkDescriptor.ADD_CONNECT, NetworkDescriptor.CONCAT_CONNECT]
    ):
        for index_a in range(len(weighted_layer_ids)):
            for index_b in range(len(weighted_layer_ids))[index_a + 1 :]:
                if p < len(sorted_skips) and sorted_skips[p] == (
                    index_a + 1,
                    index_b + 1,
                    skip_type,
                ):
                    p += 1
                else:
                    valid_connection.append((index_a, index_b, skip_type))

    if len(valid_connection) < 1:
        return graph
    # n_skip_connection = randint(1, len(valid_connection))
    # for index_a, index_b, skip_type in sample(valid_connection, n_skip_connection):
    for index_a, index_b, skip_type in sample(valid_connection, 1):
        a_id = weighted_layer_ids[index_a]
        b_id = weighted_layer_ids[index_b]
        if skip_type == NetworkDescriptor.ADD_CONNECT:
            graph.to_add_skip_model(a_id, b_id)
        else:
            graph.to_concat_skip_model(a_id, b_id)
    return graph


def to_deeper_graph(graph):
    weighted_layer_ids = graph.deep_layer_ids()
    if len(weighted_layer_ids) >= Constant.MAX_LAYERS:
        return None

    deeper_layer_ids = sample(weighted_layer_ids, 1)

    for layer_id in deeper_layer_ids:
        layer = graph.layer_list[layer_id]
        if is_layer(layer, "Conv"):
            graph.to_conv_deeper_model(layer_id, 3)
        else:
            graph.to_dense_deeper_model(layer_id)
    return graph


def legal_graph(graph):
    descriptor = graph.extract_descriptor()
    skips = descriptor.skip_connections
    if len(skips) != len(set(skips)):
        return False
    return True


def transform(graph):
    graphs = []
    for _ in range(Constant.N_NEIGHBOURS * 2):
        a = randrange(3)
        temp_graph = None
        if a == 0:
            temp_graph = to_deeper_graph(deepcopy(graph))
        elif a == 1:
            temp_graph = to_wider_graph(deepcopy(graph))
        elif a == 2:
            temp_graph = to_skip_connection_graph(deepcopy(graph))

        if temp_graph is not None and temp_graph.size() <= Constant.MAX_MODEL_SIZE:
            graphs.append(temp_graph)

        if len(graphs) >= Constant.N_NEIGHBOURS:
            break

    return list(filter(lambda x: legal_graph(x), graphs))
