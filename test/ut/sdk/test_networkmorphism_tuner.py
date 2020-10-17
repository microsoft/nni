# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from unittest import TestCase, main
from copy import deepcopy
import torch

from nni.algorithms.hpo.networkmorphism_tuner.graph import graph_to_json, json_to_graph
from nni.algorithms.hpo.networkmorphism_tuner.graph_transformer import (
    to_deeper_graph,
    to_skip_connection_graph,
    to_wider_graph,
)
from nni.algorithms.hpo.networkmorphism_tuner.layers import layer_description_extractor
from nni.algorithms.hpo.networkmorphism_tuner.networkmorphism_tuner import NetworkMorphismTuner
from nni.algorithms.hpo.networkmorphism_tuner.nn import CnnGenerator


class NetworkMorphismTestCase(TestCase):
    """  unittest for NetworkMorphismTuner
    """

    def test_graph_json_transform(self):
        """ unittest for graph_json_transform function
        """

        graph_init = CnnGenerator(10, (32, 32, 3)).generate()
        graph_init = to_wider_graph(deepcopy(graph_init))
        graph_init = to_deeper_graph(deepcopy(graph_init))
        graph_init = to_skip_connection_graph(deepcopy(graph_init))
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
        self.assertEqual(
            graph_init.reverse_adj_list,
            graph_recover.reverse_adj_list)
        self.assertEqual(
            len(graph_init.operation_history), len(
                graph_recover.operation_history)
        )
        self.assertEqual(graph_init.n_dim, graph_recover.n_dim)
        self.assertEqual(graph_init.conv, graph_recover.conv)
        self.assertEqual(graph_init.batch_norm, graph_recover.batch_norm)
        self.assertEqual(graph_init.vis, graph_recover.vis)

        node_list_init = [node.shape for node in graph_init.node_list]
        node_list_recover = [node.shape for node in graph_recover.node_list]
        self.assertEqual(node_list_init, node_list_recover)
        self.assertEqual(len(graph_init.node_to_id),
                         len(graph_recover.node_to_id))
        layer_list_init = [
            layer_description_extractor(item, graph_init.node_to_id)
            for item in graph_init.layer_list
        ]
        layer_list_recover = [
            layer_description_extractor(item, graph_recover.node_to_id)
            for item in graph_recover.layer_list
        ]
        self.assertEqual(layer_list_init, layer_list_recover)

        node_to_id_init = [graph_init.node_to_id[node]
                           for node in graph_init.node_list]
        node_to_id_recover = [
            graph_recover.node_to_id[node] for node in graph_recover.node_list
        ]
        self.assertEqual(node_to_id_init, node_to_id_recover)

        layer_to_id_init = [
            graph_init.layer_to_id[layer] for layer in graph_init.layer_list
        ]
        layer_to_id_recover = [
            graph_recover.layer_to_id[layer] for layer in graph_recover.layer_list
        ]
        self.assertEqual(layer_to_id_init, layer_to_id_recover)

    def test_to_wider_graph(self):
        """ unittest for to_wider_graph function
        """

        graph_init = CnnGenerator(10, (32, 32, 3)).generate()
        json_out = graph_to_json(graph_init, "temp.json")
        graph_recover = json_to_graph(json_out)
        wider_graph = to_wider_graph(deepcopy(graph_recover))
        model = wider_graph.produce_torch_model()
        out = model(torch.ones(1, 3, 32, 32))
        self.assertEqual(out.shape, torch.Size([1, 10]))

    def test_to_deeper_graph(self):
        """ unittest for to_deeper_graph function
        """

        graph_init = CnnGenerator(10, (32, 32, 3)).generate()
        json_out = graph_to_json(graph_init, "temp.json")
        graph_recover = json_to_graph(json_out)
        deeper_graph = to_deeper_graph(deepcopy(graph_recover))
        model = deeper_graph.produce_torch_model()
        out = model(torch.ones(1, 3, 32, 32))
        self.assertEqual(out.shape, torch.Size([1, 10]))

    def test_to_skip_connection_graph(self):
        """ unittest for to_skip_connection_graph function
        """

        graph_init = CnnGenerator(10, (32, 32, 3)).generate()
        json_out = graph_to_json(graph_init, "temp.json")
        graph_recover = json_to_graph(json_out)
        skip_connection_graph = to_skip_connection_graph(deepcopy(graph_recover))
        model = skip_connection_graph.produce_torch_model()
        out = model(torch.ones(1, 3, 32, 32))
        self.assertEqual(out.shape, torch.Size([1, 10]))

    def test_generate_parameters(self):
        """ unittest for generate_parameters function
        """

        tuner = NetworkMorphismTuner()
        model_json = tuner.generate_parameters(0)
        model_json = json.loads(model_json)
        self.assertEqual(model_json["input_shape"], [32, 32, 3])
        self.assertEqual(tuner.total_data[0][1:], (-1, 0))

    def test_receive_trial_result(self):
        """ unittest for receive_trial_result function
        """

        tuner = NetworkMorphismTuner()
        model_json = tuner.generate_parameters(0)
        tuner.receive_trial_result(0, {}, 0.7)
        (json_out, father_id, model_id) = tuner.total_data[0]

        self.assertEqual(father_id, -1)
        self.assertEqual(model_json, json_out)

        ret = {"model_id": 0, "metric_value": 0.7}
        self.assertEqual(tuner.bo.search_tree.adj_list[model_id], [])
        self.assertEqual(tuner.history[-1], ret)

    def test_update_search_space(self):
        """ unittest for update_search_space function
        """

        tuner = NetworkMorphismTuner()
        self.assertEqual(tuner.search_space, dict())
        tuner.update_search_space("Test")
        self.assertEqual(tuner.search_space, "Test")

    def test_init_search(self):
        """ unittest for init_search function
        """

        tuner = NetworkMorphismTuner()
        self.assertEqual(tuner.history, [])
        tuner.init_search()
        self.assertEqual(tuner.model_count, 1)
        self.assertEqual(len(tuner.training_queue), 1)
        self.assertEqual(len(tuner.descriptors), 1)

    def test_add_model(self):
        """ unittest for add_model function
        """

        tuner = NetworkMorphismTuner()
        tuner.add_model(0.8, 0)
        ret = {"model_id": 0, "metric_value": 0.8}
        self.assertEqual(tuner.history[-1], ret)

    def test_get_best_model_id(self):
        """ unittest for get_best_model_id function
        """

        tuner = NetworkMorphismTuner()
        tuner.add_model(0.8, 0)
        tuner.add_model(0.9, 1)
        self.assertEqual(tuner.get_best_model_id(), 1)


if __name__ == "__main__":
    main()
