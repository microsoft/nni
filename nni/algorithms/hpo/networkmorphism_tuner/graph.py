# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from collections.abc import Iterable
from copy import deepcopy, copy
from queue import Queue

import numpy as np
import torch

from .layer_transformer import (
    add_noise,
    wider_bn,
    wider_next_conv,
    wider_next_dense,
    wider_pre_conv,
    wider_pre_dense,
    init_dense_weight,
    init_conv_weight,
    init_bn_weight,
)
from .layers import (
    StubAdd,
    StubConcatenate,
    StubReLU,
    get_batch_norm_class,
    get_conv_class,
    is_layer,
    layer_width,
    set_keras_weight_to_stub,
    set_stub_weight_to_keras,
    set_stub_weight_to_torch,
    set_torch_weight_to_stub,
    to_real_keras_layer,
    layer_description_extractor,
    layer_description_builder,
)
from .utils import Constant


class NetworkDescriptor:
    """A class describing the neural architecture for neural network kernel.
    It only record the width of convolutional and dense layers, and the skip-connection types and positions.
    """

    CONCAT_CONNECT = "concat"
    ADD_CONNECT = "add"

    def __init__(self):
        self.skip_connections = []
        self.layers = []

    @property
    def n_layers(self):
        return len(self.layers)

    def add_skip_connection(self, u, v, connection_type):
        """ Add a skip-connection to the descriptor.
        Args:
            u: Number of convolutional layers before the starting point.
            v: Number of convolutional layers before the ending point.
            connection_type: Must be either CONCAT_CONNECT or ADD_CONNECT.
        """
        if connection_type not in [self.CONCAT_CONNECT, self.ADD_CONNECT]:
            raise ValueError(
                "connection_type should be NetworkDescriptor.CONCAT_CONNECT "
                "or NetworkDescriptor.ADD_CONNECT."
            )
        self.skip_connections.append((u, v, connection_type))

    def to_json(self):
        ''' NetworkDescriptor to json representation
        '''

        skip_list = []
        for u, v, connection_type in self.skip_connections:
            skip_list.append({"from": u, "to": v, "type": connection_type})
        return {"node_list": self.layers, "skip_list": skip_list}

    def add_layer(self, layer):
        ''' add one layer
        '''

        self.layers.append(layer)


class Node:
    """A class for intermediate output tensor (node) in the Graph.
    Attributes:
        shape: A tuple describing the shape of the tensor.
    """

    def __init__(self, shape):
        self.shape = shape


class Graph:
    """A class representing the neural architecture graph of a model.
    Graph extracts the neural architecture graph from a model.
    Each node in the graph is a intermediate tensor between layers.
    Each layer is an edge in the graph.
    Notably, multiple edges may refer to the same layer.
    (e.g. Add layer is adding two tensor into one tensor. So it is related to two edges.)
    Attributes:
        weighted: A boolean of whether the weights and biases in the neural network
            should be included in the graph.
        input_shape: A tuple of integers, which does not include the batch axis.
        node_list: A list of integers. The indices of the list are the identifiers.
        layer_list: A list of stub layers. The indices of the list are the identifiers.
        node_to_id: A dict instance mapping from node integers to their identifiers.
        layer_to_id: A dict instance mapping from stub layers to their identifiers.
        layer_id_to_input_node_ids: A dict instance mapping from layer identifiers
            to their input nodes identifiers.
        layer_id_to_output_node_ids: A dict instance mapping from layer identifiers
            to their output nodes identifiers.
        adj_list: A two dimensional list. The adjacency list of the graph. The first dimension is
            identified by tensor identifiers. In each edge list, the elements are two-element tuples
            of (tensor identifier, layer identifier).
        reverse_adj_list: A reverse adjacent list in the same format as adj_list.
        operation_history: A list saving all the network morphism operations.
        vis: A dictionary of temporary storage for whether an local operation has been done
            during the network morphism.
    """

    def __init__(self, input_shape, weighted=True):
        """Initializer for Graph.
        """
        self.input_shape = input_shape
        self.weighted = weighted
        self.node_list = []
        self.layer_list = []
        # node id start with 0
        self.node_to_id = {}
        self.layer_to_id = {}
        self.layer_id_to_input_node_ids = {}
        self.layer_id_to_output_node_ids = {}
        self.adj_list = {}
        self.reverse_adj_list = {}
        self.operation_history = []
        self.n_dim = len(input_shape) - 1
        self.conv = get_conv_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

        self.vis = None
        self._add_node(Node(input_shape))

    def add_layer(self, layer, input_node_id):
        """Add a layer to the Graph.
        Args:
            layer: An instance of the subclasses of StubLayer in layers.py.
            input_node_id: An integer. The ID of the input node of the layer.
        Returns:
            output_node_id: An integer. The ID of the output node of the layer.
        """
        if isinstance(input_node_id, Iterable):
            layer.input = list(map(lambda x: self.node_list[x], input_node_id))
            output_node_id = self._add_node(Node(layer.output_shape))
            for node_id in input_node_id:
                self._add_edge(layer, node_id, output_node_id)

        else:
            layer.input = self.node_list[input_node_id]
            output_node_id = self._add_node(Node(layer.output_shape))
            self._add_edge(layer, input_node_id, output_node_id)

        layer.output = self.node_list[output_node_id]
        return output_node_id

    def clear_operation_history(self):
        self.operation_history = []

    @property
    def n_nodes(self):
        """Return the number of nodes in the model."""
        return len(self.node_list)

    @property
    def n_layers(self):
        """Return the number of layers in the model."""
        return len(self.layer_list)

    def _add_node(self, node):
        """Add a new node to node_list and give the node an ID.
        Args:
            node: An instance of Node.
        Returns:
            node_id: An integer.
        """
        node_id = len(self.node_list)
        self.node_to_id[node] = node_id
        self.node_list.append(node)
        self.adj_list[node_id] = []
        self.reverse_adj_list[node_id] = []
        return node_id

    def _add_edge(self, layer, input_id, output_id):
        """Add a new layer to the graph. The nodes should be created in advance."""

        if layer in self.layer_to_id:
            layer_id = self.layer_to_id[layer]
            if input_id not in self.layer_id_to_input_node_ids[layer_id]:
                self.layer_id_to_input_node_ids[layer_id].append(input_id)
            if output_id not in self.layer_id_to_output_node_ids[layer_id]:
                self.layer_id_to_output_node_ids[layer_id].append(output_id)
        else:
            layer_id = len(self.layer_list)
            self.layer_list.append(layer)
            self.layer_to_id[layer] = layer_id
            self.layer_id_to_input_node_ids[layer_id] = [input_id]
            self.layer_id_to_output_node_ids[layer_id] = [output_id]

        self.adj_list[input_id].append((output_id, layer_id))
        self.reverse_adj_list[output_id].append((input_id, layer_id))

    def _redirect_edge(self, u_id, v_id, new_v_id):
        """Redirect the layer to a new node.
        Change the edge originally from `u_id` to `v_id` into an edge from `u_id` to `new_v_id`
        while keeping all other property of the edge the same.
        """
        layer_id = None
        for index, edge_tuple in enumerate(self.adj_list[u_id]):
            if edge_tuple[0] == v_id:
                layer_id = edge_tuple[1]
                self.adj_list[u_id][index] = (new_v_id, layer_id)
                self.layer_list[layer_id].output = self.node_list[new_v_id]
                break

        for index, edge_tuple in enumerate(self.reverse_adj_list[v_id]):
            if edge_tuple[0] == u_id:
                layer_id = edge_tuple[1]
                self.reverse_adj_list[v_id].remove(edge_tuple)
                break
        self.reverse_adj_list[new_v_id].append((u_id, layer_id))
        for index, value in enumerate(
                self.layer_id_to_output_node_ids[layer_id]):
            if value == v_id:
                self.layer_id_to_output_node_ids[layer_id][index] = new_v_id
                break

    def _replace_layer(self, layer_id, new_layer):
        """Replace the layer with a new layer."""
        old_layer = self.layer_list[layer_id]
        new_layer.input = old_layer.input
        new_layer.output = old_layer.output
        new_layer.output.shape = new_layer.output_shape
        self.layer_list[layer_id] = new_layer
        self.layer_to_id[new_layer] = layer_id
        self.layer_to_id.pop(old_layer)

    @property
    def topological_order(self):
        """Return the topological order of the node IDs from the input node to the output node."""
        q = Queue()
        in_degree = {}
        for i in range(self.n_nodes):
            in_degree[i] = 0
        for u in range(self.n_nodes):
            for v, _ in self.adj_list[u]:
                in_degree[v] += 1
        for i in range(self.n_nodes):
            if in_degree[i] == 0:
                q.put(i)

        order_list = []
        while not q.empty():
            u = q.get()
            order_list.append(u)
            for v, _ in self.adj_list[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.put(v)
        return order_list

    def _get_pooling_layers(self, start_node_id, end_node_id):
        """Given two node IDs, return all the pooling layers between them."""
        layer_list = []
        node_list = [start_node_id]
        assert self._depth_first_search(end_node_id, layer_list, node_list)
        ret = []
        for layer_id in layer_list:
            layer = self.layer_list[layer_id]
            if is_layer(layer, "Pooling"):
                ret.append(layer)
            elif is_layer(layer, "Conv") and layer.stride != 1:
                ret.append(layer)
        return ret

    def _depth_first_search(self, target_id, layer_id_list, node_list):
        """Search for all the layers and nodes down the path.
        A recursive function to search all the layers and nodes between the node in the node_list
            and the node with target_id."""
        assert len(node_list) <= self.n_nodes
        u = node_list[-1]
        if u == target_id:
            return True

        for v, layer_id in self.adj_list[u]:
            layer_id_list.append(layer_id)
            node_list.append(v)
            if self._depth_first_search(target_id, layer_id_list, node_list):
                return True
            layer_id_list.pop()
            node_list.pop()

        return False

    def _search(self, u, start_dim, total_dim, n_add):
        """Search the graph for all the layers to be widened caused by an operation.
        It is an recursive function with duplication check to avoid deadlock.
        It searches from a starting node u until the corresponding layers has been widened.
        Args:
            u: The starting node ID.
            start_dim: The position to insert the additional dimensions.
            total_dim: The total number of dimensions the layer has before widening.
            n_add: The number of dimensions to add.
        """
        if (u, start_dim, total_dim, n_add) in self.vis:
            return
        self.vis[(u, start_dim, total_dim, n_add)] = True
        for v, layer_id in self.adj_list[u]:
            layer = self.layer_list[layer_id]

            if is_layer(layer, "Conv"):
                new_layer = wider_next_conv(
                    layer, start_dim, total_dim, n_add, self.weighted
                )
                self._replace_layer(layer_id, new_layer)

            elif is_layer(layer, "Dense"):
                new_layer = wider_next_dense(
                    layer, start_dim, total_dim, n_add, self.weighted
                )
                self._replace_layer(layer_id, new_layer)

            elif is_layer(layer, "BatchNormalization"):
                new_layer = wider_bn(
                    layer, start_dim, total_dim, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)
                self._search(v, start_dim, total_dim, n_add)

            elif is_layer(layer, "Concatenate"):
                if self.layer_id_to_input_node_ids[layer_id][1] == u:
                    # u is on the right of the concat
                    # next_start_dim += next_total_dim - total_dim
                    left_dim = self._upper_layer_width(
                        self.layer_id_to_input_node_ids[layer_id][0]
                    )
                    next_start_dim = start_dim + left_dim
                    next_total_dim = total_dim + left_dim
                else:
                    next_start_dim = start_dim
                    next_total_dim = total_dim + self._upper_layer_width(
                        self.layer_id_to_input_node_ids[layer_id][1]
                    )
                self._search(v, next_start_dim, next_total_dim, n_add)

            else:
                self._search(v, start_dim, total_dim, n_add)

        for v, layer_id in self.reverse_adj_list[u]:
            layer = self.layer_list[layer_id]
            if is_layer(layer, "Conv"):
                new_layer = wider_pre_conv(layer, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)
            elif is_layer(layer, "Dense"):
                new_layer = wider_pre_dense(layer, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)
            elif is_layer(layer, "Concatenate"):
                continue
            else:
                self._search(v, start_dim, total_dim, n_add)

    def _upper_layer_width(self, u):
        for v, layer_id in self.reverse_adj_list[u]:
            layer = self.layer_list[layer_id]
            if is_layer(layer, "Conv") or is_layer(layer, "Dense"):
                return layer_width(layer)
            elif is_layer(layer, "Concatenate"):
                a = self.layer_id_to_input_node_ids[layer_id][0]
                b = self.layer_id_to_input_node_ids[layer_id][1]
                return self._upper_layer_width(a) + self._upper_layer_width(b)
            else:
                return self._upper_layer_width(v)
        return self.node_list[0].shape[-1]

    def to_deeper_model(self, target_id, new_layer):
        """Insert a relu-conv-bn block after the target block.
        Args:
            target_id: A convolutional layer ID. The new block should be inserted after the block.
            new_layer: An instance of StubLayer subclasses.
        """
        self.operation_history.append(
            ("to_deeper_model", target_id, new_layer))
        input_id = self.layer_id_to_input_node_ids[target_id][0]
        output_id = self.layer_id_to_output_node_ids[target_id][0]
        if self.weighted:
            if is_layer(new_layer, "Dense"):
                init_dense_weight(new_layer)
            elif is_layer(new_layer, "Conv"):
                init_conv_weight(new_layer)
            elif is_layer(new_layer, "BatchNormalization"):
                init_bn_weight(new_layer)

        self._insert_new_layers([new_layer], input_id, output_id)

    def to_wider_model(self, pre_layer_id, n_add):
        """Widen the last dimension of the output of the pre_layer.
        Args:
            pre_layer_id: The ID of a convolutional layer or dense layer.
            n_add: The number of dimensions to add.
        """
        self.operation_history.append(("to_wider_model", pre_layer_id, n_add))
        pre_layer = self.layer_list[pre_layer_id]
        output_id = self.layer_id_to_output_node_ids[pre_layer_id][0]
        dim = layer_width(pre_layer)
        self.vis = {}
        self._search(output_id, dim, dim, n_add)
        # Update the tensor shapes.
        for u in self.topological_order:
            for v, layer_id in self.adj_list[u]:
                self.node_list[v].shape = self.layer_list[layer_id].output_shape

    def _insert_new_layers(self, new_layers, start_node_id, end_node_id):
        """Insert the new_layers after the node with start_node_id."""
        new_node_id = self._add_node(deepcopy(self.node_list[end_node_id]))
        temp_output_id = new_node_id
        for layer in new_layers[:-1]:
            temp_output_id = self.add_layer(layer, temp_output_id)

        self._add_edge(new_layers[-1], temp_output_id, end_node_id)
        new_layers[-1].input = self.node_list[temp_output_id]
        new_layers[-1].output = self.node_list[end_node_id]
        self._redirect_edge(start_node_id, end_node_id, new_node_id)

    def _block_end_node(self, layer_id, block_size):
        ret = self.layer_id_to_output_node_ids[layer_id][0]
        for _ in range(block_size - 2):
            ret = self.adj_list[ret][0][0]
        return ret

    def _dense_block_end_node(self, layer_id):
        return self.layer_id_to_input_node_ids[layer_id][0]

    def _conv_block_end_node(self, layer_id):
        """Get the input node ID of the last layer in the block by layer ID.
            Return the input node ID of the last layer in the convolutional block.
        Args:
            layer_id: the convolutional layer ID.
        """
        return self._block_end_node(layer_id, Constant.CONV_BLOCK_DISTANCE)

    def to_add_skip_model(self, start_id, end_id):
        """Add a weighted add skip-connection from after start node to end node.
        Args:
            start_id: The convolutional layer ID, after which to start the skip-connection.
            end_id: The convolutional layer ID, after which to end the skip-connection.
        """
        self.operation_history.append(("to_add_skip_model", start_id, end_id))
        filters_end = self.layer_list[end_id].output.shape[-1]
        filters_start = self.layer_list[start_id].output.shape[-1]
        start_node_id = self.layer_id_to_output_node_ids[start_id][0]

        pre_end_node_id = self.layer_id_to_input_node_ids[end_id][0]
        end_node_id = self.layer_id_to_output_node_ids[end_id][0]

        skip_output_id = self._insert_pooling_layer_chain(
            start_node_id, end_node_id)

        # Add the conv layer
        new_conv_layer = get_conv_class(
            self.n_dim)(
                filters_start,
                filters_end,
                1)
        skip_output_id = self.add_layer(new_conv_layer, skip_output_id)

        # Add the add layer.
        add_input_node_id = self._add_node(
            deepcopy(self.node_list[end_node_id]))
        add_layer = StubAdd()

        self._redirect_edge(pre_end_node_id, end_node_id, add_input_node_id)
        self._add_edge(add_layer, add_input_node_id, end_node_id)
        self._add_edge(add_layer, skip_output_id, end_node_id)
        add_layer.input = [
            self.node_list[add_input_node_id],
            self.node_list[skip_output_id],
        ]
        add_layer.output = self.node_list[end_node_id]
        self.node_list[end_node_id].shape = add_layer.output_shape

        # Set weights to the additional conv layer.
        if self.weighted:
            filter_shape = (1,) * self.n_dim
            weights = np.zeros((filters_end, filters_start) + filter_shape)
            bias = np.zeros(filters_end)
            new_conv_layer.set_weights(
                (add_noise(weights, np.array([0, 1])), add_noise(
                    bias, np.array([0, 1])))
            )

    def to_concat_skip_model(self, start_id, end_id):
        """Add a weighted add concatenate connection from after start node to end node.
        Args:
            start_id: The convolutional layer ID, after which to start the skip-connection.
            end_id: The convolutional layer ID, after which to end the skip-connection.
        """
        self.operation_history.append(
            ("to_concat_skip_model", start_id, end_id))
        filters_end = self.layer_list[end_id].output.shape[-1]
        filters_start = self.layer_list[start_id].output.shape[-1]
        start_node_id = self.layer_id_to_output_node_ids[start_id][0]

        pre_end_node_id = self.layer_id_to_input_node_ids[end_id][0]
        end_node_id = self.layer_id_to_output_node_ids[end_id][0]

        skip_output_id = self._insert_pooling_layer_chain(
            start_node_id, end_node_id)

        concat_input_node_id = self._add_node(
            deepcopy(self.node_list[end_node_id]))
        self._redirect_edge(pre_end_node_id, end_node_id, concat_input_node_id)

        concat_layer = StubConcatenate()
        concat_layer.input = [
            self.node_list[concat_input_node_id],
            self.node_list[skip_output_id],
        ]
        concat_output_node_id = self._add_node(Node(concat_layer.output_shape))
        self._add_edge(
            concat_layer,
            concat_input_node_id,
            concat_output_node_id)
        self._add_edge(concat_layer, skip_output_id, concat_output_node_id)
        concat_layer.output = self.node_list[concat_output_node_id]
        self.node_list[concat_output_node_id].shape = concat_layer.output_shape

        # Add the concatenate layer.
        new_conv_layer = get_conv_class(self.n_dim)(
            filters_start + filters_end, filters_end, 1
        )
        self._add_edge(new_conv_layer, concat_output_node_id, end_node_id)
        new_conv_layer.input = self.node_list[concat_output_node_id]
        new_conv_layer.output = self.node_list[end_node_id]
        self.node_list[end_node_id].shape = new_conv_layer.output_shape

        if self.weighted:
            filter_shape = (1,) * self.n_dim
            weights = np.zeros((filters_end, filters_end) + filter_shape)
            for i in range(filters_end):
                filter_weight = np.zeros((filters_end,) + filter_shape)
                center_index = (i,) + (0,) * self.n_dim
                filter_weight[center_index] = 1
                weights[i, ...] = filter_weight
            weights = np.concatenate(
                (weights, np.zeros((filters_end, filters_start) + filter_shape)), axis=1
            )
            bias = np.zeros(filters_end)
            new_conv_layer.set_weights(
                (add_noise(weights, np.array([0, 1])), add_noise(
                    bias, np.array([0, 1])))
            )

    def _insert_pooling_layer_chain(self, start_node_id, end_node_id):
        skip_output_id = start_node_id
        for layer in self._get_pooling_layers(start_node_id, end_node_id):
            new_layer = deepcopy(layer)
            if is_layer(new_layer, "Conv"):
                filters = self.node_list[start_node_id].shape[-1]
                new_layer = get_conv_class(self.n_dim)(
                    filters, filters, 1, layer.stride)
                if self.weighted:
                    init_conv_weight(new_layer)
            else:
                new_layer = deepcopy(layer)
            skip_output_id = self.add_layer(new_layer, skip_output_id)
        skip_output_id = self.add_layer(StubReLU(), skip_output_id)
        return skip_output_id

    def extract_descriptor(self):
        """Extract the the description of the Graph as an instance of NetworkDescriptor."""
        main_chain = self.get_main_chain()
        index_in_main_chain = {}
        for index, u in enumerate(main_chain):
            index_in_main_chain[u] = index

        ret = NetworkDescriptor()
        for u in main_chain:
            for v, layer_id in self.adj_list[u]:
                if v not in index_in_main_chain:
                    continue
                layer = self.layer_list[layer_id]
                copied_layer = copy(layer)
                copied_layer.weights = None
                ret.add_layer(deepcopy(copied_layer))

        for u in index_in_main_chain:
            for v, layer_id in self.adj_list[u]:
                if v not in index_in_main_chain:
                    temp_u = u
                    temp_v = v
                    temp_layer_id = layer_id
                    skip_type = None
                    while not (
                            temp_v in index_in_main_chain and temp_u in index_in_main_chain):
                        if is_layer(
                                self.layer_list[temp_layer_id], "Concatenate"):
                            skip_type = NetworkDescriptor.CONCAT_CONNECT
                        if is_layer(self.layer_list[temp_layer_id], "Add"):
                            skip_type = NetworkDescriptor.ADD_CONNECT
                        temp_u = temp_v
                        temp_v, temp_layer_id = self.adj_list[temp_v][0]
                    ret.add_skip_connection(
                        index_in_main_chain[u], index_in_main_chain[temp_u], skip_type
                    )

                elif index_in_main_chain[v] - index_in_main_chain[u] != 1:
                    skip_type = None
                    if is_layer(self.layer_list[layer_id], "Concatenate"):
                        skip_type = NetworkDescriptor.CONCAT_CONNECT
                    if is_layer(self.layer_list[layer_id], "Add"):
                        skip_type = NetworkDescriptor.ADD_CONNECT
                    ret.add_skip_connection(
                        index_in_main_chain[u], index_in_main_chain[v], skip_type
                    )

        return ret

    def clear_weights(self):
        ''' clear weights of the graph
        '''
        self.weighted = False
        for layer in self.layer_list:
            layer.weights = None

    def produce_torch_model(self):
        """Build a new Torch model based on the current graph."""
        return TorchModel(self)

    def produce_keras_model(self):
        """Build a new keras model based on the current graph."""
        return KerasModel(self).model

    def produce_onnx_model(self):
        """Build a new ONNX model based on the current graph."""
        return ONNXModel(self)

    def parsing_onnx_model(self, onnx_model):
        '''to do in the future to use the onnx model
        '''
        return ONNXModel(onnx_model)

    def produce_json_model(self):
        """Build a new Json model based on the current graph."""
        return JSONModel(self).data

    @classmethod
    def parsing_json_model(cls, json_model):
        '''build a graph from json
        '''
        return json_to_graph(json_model)

    def _layer_ids_in_order(self, layer_ids):
        node_id_to_order_index = {}
        for index, node_id in enumerate(self.topological_order):
            node_id_to_order_index[node_id] = index
        return sorted(
            layer_ids,
            key=lambda layer_id: node_id_to_order_index[
                self.layer_id_to_output_node_ids[layer_id][0]
            ],
        )

    def _layer_ids_by_type(self, type_str):
        return list(
            filter(
                lambda layer_id: is_layer(self.layer_list[layer_id], type_str),
                range(self.n_layers),
            )
        )

    def get_main_chain_layers(self):
        """Return a list of layer IDs in the main chain."""
        main_chain = self.get_main_chain()
        ret = []
        for u in main_chain:
            for v, layer_id in self.adj_list[u]:
                if v in main_chain and u in main_chain:
                    ret.append(layer_id)
        return ret

    def _conv_layer_ids_in_order(self):
        return list(
            filter(
                lambda layer_id: is_layer(self.layer_list[layer_id], "Conv"),
                self.get_main_chain_layers(),
            )
        )

    def _dense_layer_ids_in_order(self):
        return self._layer_ids_in_order(self._layer_ids_by_type("Dense"))

    def deep_layer_ids(self):
        ret = []
        for layer_id in self.get_main_chain_layers():
            layer = self.layer_list[layer_id]
            if is_layer(layer, "GlobalAveragePooling"):
                break
            if is_layer(layer, "Add") or is_layer(layer, "Concatenate"):
                continue
            ret.append(layer_id)
        return ret

    def wide_layer_ids(self):
        return (
            self._conv_layer_ids_in_order(
            )[:-1] + self._dense_layer_ids_in_order()[:-1]
        )

    def skip_connection_layer_ids(self):
        return self.deep_layer_ids()[:-1]

    def size(self):
        return sum(list(map(lambda x: x.size(), self.layer_list)))

    def get_main_chain(self):
        """Returns the main chain node ID list."""
        pre_node = {}
        distance = {}
        for i in range(self.n_nodes):
            distance[i] = 0
            pre_node[i] = i
        for i in range(self.n_nodes - 1):
            for u in range(self.n_nodes):
                for v, _ in self.adj_list[u]:
                    if distance[u] + 1 > distance[v]:
                        distance[v] = distance[u] + 1
                        pre_node[v] = u
        temp_id = 0
        for i in range(self.n_nodes):
            if distance[i] > distance[temp_id]:
                temp_id = i
        ret = []
        for i in range(self.n_nodes + 5):
            ret.append(temp_id)
            if pre_node[temp_id] == temp_id:
                break
            temp_id = pre_node[temp_id]
        assert temp_id == pre_node[temp_id]
        ret.reverse()
        return ret


class TorchModel(torch.nn.Module):
    """A neural network class using pytorch constructed from an instance of Graph."""

    def __init__(self, graph):
        super(TorchModel, self).__init__()
        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(layer.to_real_layer())
        if graph.weighted:
            for index, layer in enumerate(self.layers):
                set_stub_weight_to_torch(self.graph.layer_list[index], layer)
        for index, layer in enumerate(self.layers):
            self.add_module(str(index), layer)

    def forward(self, input_tensor):
        topo_node_list = self.graph.topological_order
        output_id = topo_node_list[-1]
        input_id = topo_node_list[0]

        node_list = deepcopy(self.graph.node_list)
        node_list[input_id] = input_tensor

        for v in topo_node_list:
            for u, layer_id in self.graph.reverse_adj_list[v]:
                layer = self.graph.layer_list[layer_id]
                torch_layer = self.layers[layer_id]

                if isinstance(layer, (StubAdd, StubConcatenate)):
                    edge_input_tensor = list(
                        map(
                            lambda x: node_list[x],
                            self.graph.layer_id_to_input_node_ids[layer_id],
                        )
                    )
                else:
                    edge_input_tensor = node_list[u]

                temp_tensor = torch_layer(edge_input_tensor)
                node_list[v] = temp_tensor
        return node_list[output_id]

    def set_weight_to_graph(self):
        self.graph.weighted = True
        for index, layer in enumerate(self.layers):
            set_torch_weight_to_stub(layer, self.graph.layer_list[index])


class KerasModel:
    def __init__(self, graph):
        import keras

        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(to_real_keras_layer(layer))

        # Construct the keras graph.
        # Input
        topo_node_list = self.graph.topological_order
        output_id = topo_node_list[-1]
        input_id = topo_node_list[0]
        input_tensor = keras.layers.Input(
            shape=graph.node_list[input_id].shape)

        node_list = deepcopy(self.graph.node_list)
        node_list[input_id] = input_tensor

        # Output
        for v in topo_node_list:
            for u, layer_id in self.graph.reverse_adj_list[v]:
                layer = self.graph.layer_list[layer_id]
                keras_layer = self.layers[layer_id]

                if isinstance(layer, (StubAdd, StubConcatenate)):
                    edge_input_tensor = list(
                        map(
                            lambda x: node_list[x],
                            self.graph.layer_id_to_input_node_ids[layer_id],
                        )
                    )
                else:
                    edge_input_tensor = node_list[u]

                temp_tensor = keras_layer(edge_input_tensor)
                node_list[v] = temp_tensor

        output_tensor = node_list[output_id]
        output_tensor = keras.layers.Activation("softmax", name="activation_add")(
            output_tensor
        )
        self.model = keras.models.Model(
            inputs=input_tensor, outputs=output_tensor)

        if graph.weighted:
            for index, layer in enumerate(self.layers):
                set_stub_weight_to_keras(self.graph.layer_list[index], layer)

    def set_weight_to_graph(self):
        self.graph.weighted = True
        for index, layer in enumerate(self.layers):
            set_keras_weight_to_stub(layer, self.graph.layer_list[index])


class ONNXModel:
    # to do in the future using onnx ir
    def __init__(self, graph):
        pass


class JSONModel:
    def __init__(self, graph):
        data = dict()
        node_list = list()
        layer_list = list()
        operation_history = list()

        data["input_shape"] = graph.input_shape
        vis = graph.vis
        data["vis"] = list(vis.keys()) if vis is not None else None
        data["weighted"] = graph.weighted

        for item in graph.operation_history:
            if item[0] == "to_deeper_model":
                operation_history.append(
                    [
                        item[0],
                        item[1],
                        layer_description_extractor(item[2], graph.node_to_id),
                    ]
                )
            else:
                operation_history.append(item)
        data["operation_history"] = operation_history
        data["layer_id_to_input_node_ids"] = graph.layer_id_to_input_node_ids
        data["layer_id_to_output_node_ids"] = graph.layer_id_to_output_node_ids
        data["adj_list"] = graph.adj_list
        data["reverse_adj_list"] = graph.reverse_adj_list

        for node in graph.node_list:
            node_id = graph.node_to_id[node]
            node_information = node.shape
            node_list.append((node_id, node_information))

        for layer_id, item in enumerate(graph.layer_list):
            layer = graph.layer_list[layer_id]
            layer_information = layer_description_extractor(
                layer, graph.node_to_id)
            layer_list.append((layer_id, layer_information))

        data["node_list"] = node_list
        data["layer_list"] = layer_list

        self.data = data


def graph_to_onnx(graph, onnx_model_path):
    import onnx
    # to do in the future using onnx ir
    onnx_out = graph.produce_onnx_model()
    onnx.save(onnx_out, onnx_model_path)
    return onnx_out


def onnx_to_graph(onnx_model, input_shape):
    # to do in the future using onnx ir
    graph = Graph(input_shape, False)
    graph.parsing_onnx_model(onnx_model)
    return graph


def graph_to_json(graph, json_model_path):
    json_out = graph.produce_json_model()
    with open(json_model_path, "w") as fout:
        json.dump(json_out, fout)
    json_out = json.dumps(json_out)
    return json_out


def json_to_graph(json_model: str):
    json_model = json.loads(json_model)
    # restore graph data from json data
    input_shape = tuple(json_model["input_shape"])
    node_list = list()
    node_to_id = dict()
    id_to_node = dict()
    layer_list = list()
    layer_to_id = dict()
    operation_history = list()
    graph = Graph(input_shape, False)

    graph.input_shape = input_shape
    vis = json_model["vis"]
    graph.vis = {
        tuple(item): True for item in vis} if vis is not None else None
    graph.weighted = json_model["weighted"]
    layer_id_to_input_node_ids = json_model["layer_id_to_input_node_ids"]
    graph.layer_id_to_input_node_ids = {
        int(k): v for k, v in layer_id_to_input_node_ids.items()
    }
    layer_id_to_output_node_ids = json_model["layer_id_to_output_node_ids"]
    graph.layer_id_to_output_node_ids = {
        int(k): v for k, v in layer_id_to_output_node_ids.items()
    }
    adj_list = {}
    for k, v in json_model["adj_list"].items():
        adj_list[int(k)] = [tuple(i) for i in v]
    graph.adj_list = adj_list
    reverse_adj_list = {}
    for k, v in json_model["reverse_adj_list"].items():
        reverse_adj_list[int(k)] = [tuple(i) for i in v]
    graph.reverse_adj_list = reverse_adj_list

    for item in json_model["node_list"]:
        new_node = Node(tuple(item[1]))
        node_id = item[0]
        node_list.append(new_node)
        node_to_id[new_node] = node_id
        id_to_node[node_id] = new_node

    for item in json_model["operation_history"]:
        if item[0] == "to_deeper_model":
            operation_history.append(
                (item[0], item[1], layer_description_builder(item[2], id_to_node))
            )
        else:
            operation_history.append(item)
    graph.operation_history = operation_history

    for item in json_model["layer_list"]:
        new_layer = layer_description_builder(item[1], id_to_node)
        layer_id = int(item[0])
        layer_list.append(new_layer)
        layer_to_id[new_layer] = layer_id

    graph.node_list = node_list
    graph.node_to_id = node_to_id
    graph.layer_list = layer_list
    graph.layer_to_id = layer_to_id

    return graph
