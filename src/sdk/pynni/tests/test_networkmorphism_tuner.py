import logging
import random
from io import BytesIO

import nni
import nni.protocol
from nni.protocol import CommandType, send, receive

from unittest import TestCase, main

from nni.networkmorphism_tuner.networkmorphism_tuner import NetworkMorphismTuner
from nni.networkmorphism_tuner.graph import json_to_graph, graph_to_json
from nni.networkmorphism_tuner.nn import CnnGenerator
from nni.networkmorphism_tuner.layers import layer_description_extractor

import os
import json

_in_buf = BytesIO()
_out_buf = BytesIO()


def _reverse_io():
    _in_buf.seek(0)
    _out_buf.seek(0)
    nni.protocol._out_file = _in_buf
    nni.protocol._in_file = _out_buf


def _restore_io():
    _in_buf.seek(0)
    _out_buf.seek(0)
    nni.protocol._in_file = _in_buf
    nni.protocol._out_file = _out_buf


class NetworkMorphismTestCase(TestCase):
    def test_graph_json_transform(self):
        graph_init = CnnGenerator(10, (32, 32, 3)).generate()
        json_out = graph_to_json(graph_init, "temp.json")

        graph_recover = json_to_graph(json_out)

        # compare all data in graph
        self.assertEqual(graph_init.input_shape, graph_recover.input_shape)
        self.assertEqual(graph_init.weighted, graph_recover.weighted)
        self.assertEqual(
            graph_init.layer_id_to_input_node_ids,
            graph_recover.layer_id_to_input_node_ids,
        )
        self.assertEqual(graph_init.adj_list, graph_recover.adj_list)
        self.assertEqual(graph_init.reverse_adj_list, graph_recover.reverse_adj_list)
        self.assertEqual(graph_init.operation_history, graph_recover.operation_history)
        self.assertEqual(graph_init.n_dim, graph_recover.n_dim)
        self.assertEqual(graph_init.conv, graph_recover.conv)
        self.assertEqual(graph_init.batch_norm, graph_recover.batch_norm)
        self.assertEqual(graph_init.vis, graph_recover.vis)

        node_list_init = [node.shape for node in graph_init.node_list]
        node_list_recover = [node.shape for node in graph_recover.node_list]
        self.assertEqual(node_list_init, node_list_recover)
        self.assertEqual(len(graph_init.node_to_id), len(graph_recover.node_to_id))
        layer_list_init = [
            layer_description_extractor(item, graph_init.node_to_id)
            for item in graph_init.layer_list
        ]
        layer_list_recover = [
            layer_description_extractor(item, graph_recover.node_to_id)
            for item in graph_recover.layer_list
        ]
        self.assertEqual(layer_list_init, layer_list_recover)

        node_to_id_init=[graph_init.node_to_id[node] for node in graph_init.node_list]
        node_to_id_recover=[graph_recover.node_to_id[node] for node in graph_recover.node_list]
        self.assertEqual(node_to_id_init, node_to_id_recover)

        layer_to_id_init = [graph_init.layer_to_id[layer] for layer in graph_init.layer_list]
        layer_to_id_recover = [graph_recover.layer_to_id[layer] for layer in graph_recover.layer_list]
        self.assertEqual(layer_to_id_init, layer_to_id_recover)

    def test_generate_parameters(self):
        tuner = NetworkMorphismTuner()
        model_json = tuner.generate_parameters(0)
        model_json = json.loads(model_json)
        self.assertEqual(model_json["input_shape"], [32, 32, 3])
        self.assertEqual(tuner.total_data[0][1:], (-1, 0))

    def test_receive_trial_result(self):
        tuner = NetworkMorphismTuner()
        model_json = tuner.generate_parameters(0)
        tuner.receive_trial_result(0, {}, 0.7)
        (json_out, father_id, model_id) = tuner.total_data[0]

        self.assertEqual(father_id, -1)
        self.assertEqual(model_json, json_out)

        graph = json_to_graph(model_json)

        ret = {"model_id": 0, "metric_value": 0.7}
        self.assertEqual(tuner.bo.search_tree.adj_list[model_id], [])
        self.assertEqual(tuner.history[-1], ret)
        self.assertEqual(
            tuner.x_queue[-1].to_json(), graph.extract_descriptor().to_json()
        )
        self.assertEqual(tuner.y_queue[-1], 0.7)

    def test_update_search_space(self):
        tuner = NetworkMorphismTuner()
        self.assertEqual(tuner.search_space, dict())
        tuner.update_search_space("Test")
        self.assertEqual(tuner.search_space, "Test")

    def test_init_search(self):
        tuner = NetworkMorphismTuner()
        self.assertEqual(tuner.history, [])
        tuner.init_search()
        self.assertEqual(tuner.model_count, 1)
        self.assertEqual(len(tuner.training_queue), 1)
        self.assertEqual(len(tuner.descriptors), 1)

    def test_add_model(self):
        tuner = NetworkMorphismTuner()
        model_json = tuner.generate_parameters(0)
        graph = json_to_graph(model_json)

        tuner.add_model(0.8, graph, 0)
        ret = {"model_id": 0, "metric_value": 0.8}
        self.assertEqual(tuner.history[-1], ret)
        json_queue = tuner.x_queue[-1].to_json()
        json_graph = graph.extract_descriptor().to_json()
        self.assertEqual(json_queue, json_graph)
        self.assertEqual(tuner.y_queue[-1], 0.8)

    def test_get_best_model_id(self):
        tuner = NetworkMorphismTuner()
        model_json1 = tuner.generate_parameters(0)
        graph1 = json_to_graph(model_json1)
        model_json2 = tuner.generate_parameters(1)
        graph2 = json_to_graph(model_json2)
        tuner.add_model(0.8, graph1, 0)
        tuner.add_model(0.9, graph2, 1)
        self.assertEqual(tuner.get_best_model_id(), 1)


if __name__ == "__main__":
    main()
