import logging
import random
from io import BytesIO

import nni
import nni.protocol
from nni.protocol import CommandType, send, receive

from unittest import TestCase, main

from nni.networkmorphism_tuner.networkmorphism_tuner import NetworkMorphismTuner

import os
import onnx
import pickle

_in_buf = BytesIO()
_out_buf = BytesIO()


def pickle_from_file(path):
    with open(path,'rb') as f:
        graph = pickle.load(f)
    return graph


def pickle_to_file(obj, path):
    with open(path,'wb') as f:
        pickle.dump(obj, f)


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


graph1 = pickle_from_file("0.graph")
graph2 = pickle_from_file("1.graph")


class NetworkMorphismTestCase(TestCase):
    def test_generate_parameters(self):
        tuner = NetworkMorphismTuner()
        model_path = tuner.generate_parameters(0)
        expect_model_path = os.path.join(os.getcwd(), "model_path", "0.graph")
        self.assertEqual(model_path, expect_model_path)
        self.assertEqual(tuner.total_data[0], (expect_model_path, -1, 0))

    def test_receive_trial_result(self):
        tuner = NetworkMorphismTuner()
        model_path_save = tuner.generate_parameters(0)
        tuner.receive_trial_result(0, {}, 0.7)
        (model_path_load, father_id, model_id) = tuner.total_data[0]

        self.assertEqual(model_path_save, model_path_load)

        graph = pickle_from_file(model_path_load)

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
        self.assertEqual(tuner.model_count, 2)
        self.assertEqual(len(tuner.training_queue), 2)
        self.assertEqual(len(tuner.descriptors), 2)

    def test_add_model(self):
        tuner = NetworkMorphismTuner()
        tuner.add_model(0.8, graph1, 0)
        ret = {'model_id': 0, 'metric_value': 0.8}
        self.assertEqual(tuner.history[-1], ret)
        self.assertEqual(tuner.x_queue[-1].to_json(), graph1.extract_descriptor().to_json())
        self.assertEqual(tuner.y_queue[-1], 0.8)

    def test_get_best_model_id(self):
        tuner = NetworkMorphismTuner()
        tuner.add_model(0.8, graph1, 0)
        tuner.add_model(0.9, graph2, 1)
        self.assertEqual(tuner.get_best_model_id(), 1)


if __name__ == '__main__':
    main()
