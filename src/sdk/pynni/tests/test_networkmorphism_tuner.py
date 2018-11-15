import logging
import random
from io import BytesIO

import nni
import nni.protocol
from nni.protocol import CommandType, send, receive

from unittest import TestCase, main

from nni.networkmorphism_tuner.networkmorphism_tuner import NetworkMorphismTuner
from nni.networkmorphism_tuner.networkmorphism_dispatcher import NetworkMorphismMsgDispatcher

import onnx

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
    def test_tuner(self):
        # now we are sending to Tuner's incoming stream
        _reverse_io() 
        send(CommandType.UpdateSearchSpace, "{\"learning_rate\": {\"_value\": [0.0001, 0.001, 0.002, 0.005, 0.01], \"_type\": \"choice\"}, \"optimizer\": {\"_value\": [\"Adam\", \"SGD\"], \"_type\": \"choice\"}}")
        send(CommandType.RequestTrialJobs, '2')
        send(CommandType.ReportMetricData, '{"parameter_id":0,"type":"PERIODICAL","value":10,"trial_job_id":"abc"}')
        send(CommandType.ReportMetricData, '{"parameter_id":1,"type":"FINAL","value":11,"trial_job_id":"abc"}')
        send(CommandType.AddCustomizedTrialJob, '{"param":-1}')
        send(CommandType.ReportMetricData, '{"parameter_id":2,"type":"FINAL","value":22,"trial_job_id":"abc"}')
        send(CommandType.RequestTrialJobs, '1')
        send(CommandType.TrialEnd, '{"trial_job_id":"abc"}')
        _restore_io()

        
        tuner = NetworkMorphismTuner()
        dispatcher = NetworkMorphismMsgDispatcher(tuner)
        
        try:
            dispatcher.run()
        except Exception as e:
            self.assertIs(type(e),AssertionError)
            self.assertEqual(e.args[0], 'Unsupported command: CommandType.KillTrialJob')


        # now we are receiving from Tuner's outgoing stream
        _reverse_io()  
        # this one is customized
        command, data = receive()
        print(command, data)

        self.assertIs(command, CommandType.NewTrialJob)
        self.assertEqual(data, "/path/to/model")
        self.assertEqual(len(_out_buf.read()), 0)  # no more commands
        

    def test_generate_parameters(self):
        tuner = NetworkMorphismTuner()
        onnx_model_path = tuner.generate_parameters(0)
        self.assertEqual(onnx_model_path,"/path/to/model")
        self.assertEqual(tuner.total_data[0],("/path/to/model",0,0))


    def test_receive_trial_result(self):
        tuner = NetworkMorphismTuner()
        onnx_model_path_save = tuner.generate_parameters(0)
        tuner.receive_trial_result(0,{},0.7)
        (onnx_model_path_load, father_id, model_id) = tuner.total_data[0]

        self.assertEqual(onnx_model_path_save,onnx_model_path_load)

        graph = onnx.load(onnx_model_path_load)
        ret = {'model_id': 0, 'metric_value': 0.7}
        self.assertEqual(tuner.search_tree.adj_list[father_id][-1],model_id)
        self.assertEqual(tuner.history[-1],ret)
        self.assertEqual(tuner.x_queue[-1],graph.extract_descriptor())
        self.assertEqual(tuner.y_queue[-1],0.8)


    def test_update_search_space(self):
        tuner = NetworkMorphismTuner()
        self.assertEqual(tuner.search_space,[])
        tuner.update_search_space("Test")
        self.assertEqual(tuner.search_space,"Test")

    def test_init_search(self):
        tuner = NetworkMorphismTuner()
        self.assertEqual(tuner.history,[])
        tuner.init_search()
        self.assertEqual(tuner.model_count,9)
        self.assertEqual(len(tuner.training_queue),9)
        self.assertEqual(len(tuner.descriptors),9)

    def test_add_model(self):
        tuner = NetworkMorphismTuner()
        tuner.add_model(0.8,graph1,0)
        ret = {'model_id': 0, 'metric_value': 0.8} 
        self.assertEqual(tuner.history[-1],ret)
        self.assertEqual(tuner.x_queue[-1],graph1.extract_descriptor())
        self.assertEqual(tuner.y_queue[-1],0.8)

    def test_get_best_model_id(self):
        tuner = NetworkMorphismTuner()
        tuner.add_model(0.8,graph1,0)
        tuner.add_model(0.9,graph2,1)
        self.assertEqual(tuner.get_best_model_id(),1) 



if __name__ == '__main__':
    main()
