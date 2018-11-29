import logging
import random
from io import BytesIO

import nni
import nni.protocol
from nni.protocol import CommandType, send, receive

from unittest import TestCase, main

from nni.networkmorphism_tuner.networkmorphism_tuner import NetworkMorphismTuner
from nni.networkmorphism_tuner.graph import json_to_graph,graph_to_json
from nni.networkmorphism_tuner.nn import CnnGenerator

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
        graph_init = CnnGenerator(10,(32,32,3)).generate()
        json_out = graph_to_json(graph_init,"temp.json")
        print(json_out)
        print(json_out["input_shape"])
        graph_recover = json_to_graph(json_out)

        # compare all data in graph
        # for item in dir(graph_init):
        #     self.assertEqual(getattr(graph_init,item),getattr(graph_recover,item))
        self.assertEqual(graph_init.input_shape, graph_recover.input_shape)
        self.assertEqual(graph_init.weighted, graph_recover.weighted)
        self.assertEqual(graph_init.layer_id_to_input_node_ids, graph_recover.layer_id_to_input_node_ids)
        self.assertEqual(graph_init.adj_list, graph_recover.adj_list)
        self.assertEqual(graph_init.reverse_adj_list, graph_recover.reverse_adj_list)
        self.assertEqual(graph_init.operation_history, graph_recover.operation_history)
        self.assertEqual(graph_init.n_dim, graph_recover.n_dim)
        self.assertEqual(graph_init.conv, graph_recover.conv)
        self.assertEqual(graph_init.batch_norm, graph_recover.batch_norm)
        self.assertEqual(graph_init.vis, graph_recover.vis)

        self.assertEqual(graph_init.node_list, graph_recover.node_list)
        self.assertEqual(graph_init.layer_list, graph_recover.layer_list)
        self.assertEqual(graph_init.node_to_id, graph_recover.node_to_id)
        self.assertEqual(graph_init.layer_to_id, graph_recover.layer_to_id)
        


    def test_generate_parameters(self):
        tuner = NetworkMorphismTuner()
        model_json = tuner.generate_parameters(0)
        expect_json = os.path.join(os.getcwd(), "model_path", "0.graph")
        self.assertEqual(model_json, expect_json)
        self.assertEqual(tuner.total_data[0], (expect_json, -1, 0))

    def test_receive_trial_result(self):
        tuner = NetworkMorphismTuner()
        model_json = tuner.generate_parameters(0)
        tuner.receive_trial_result(0, {}, 0.7)
        (json_out, father_id, model_id) = tuner.total_data[0]

        self.assertEqual(father_id, -1)
        self.assertEqual(model_json, json_out)

        graph = json_to_graph(json_out)

        ret = {'model_id': 0, 'metric_value': 0.7}
        self.assertEqual(tuner.bo.search_tree.adj_list[model_id], [])
        self.assertEqual(tuner.history[-1], ret)
        self.assertEqual(tuner.x_queue[-1].to_json(), graph.extract_descriptor().to_json())
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
        ret = {'model_id': 0, 'metric_value': 0.8}
        self.assertEqual(tuner.history[-1], ret)
        self.assertEqual(tuner.x_queue[-1].to_json(), graph.extract_descriptor().to_json())
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


if __name__ == '__main__':
    main()
