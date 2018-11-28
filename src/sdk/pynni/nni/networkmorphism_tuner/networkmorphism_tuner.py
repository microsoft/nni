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

import json
import logging
import os
from enum import Enum, unique

import numpy as np

from nni.networkmorphism_tuner.bayesian import BayesianOptimizer
from nni.networkmorphism_tuner.metric import Accuracy
from nni.networkmorphism_tuner.nn import CnnGenerator, MlpGenerator
from nni.networkmorphism_tuner.utils import Constant, pickle_from_file, pickle_to_file
from nni.tuner import Tuner

logger = logging.getLogger("NetworkMorphism_AutoML")


@unique
class OptimizeMode(Enum):
    """
    Oprimize Mode class
    """

    Minimize = "minimize"
    Maximize = "maximize"


class NetworkMorphismTuner(Tuner):
    """ 
    NetworkMorphismTuner is a tuner which using network morphism techniques.
    """

    def __init__(
        self,
        task="cv",
        input_width=32,
        input_channel=3,
        n_output_node=10,
        algorithm_name="Bayesian",
        optimize_mode="minimize",
        path="model_path",
        verbose=True,
        metric=Accuracy,
        beta=Constant.BETA,
        t_min=Constant.T_MIN,
        max_model_size=Constant.MAX_MODEL_SIZE,
        default_model_len=Constant.MODEL_LEN,
        default_model_width=Constant.MODEL_WIDTH,
    ):
        """ initilizer of the NetworkMorphismTuner
        
        Keyword Arguments:
            task {str} -- [task mode, such as "cv","common" etc.] (default: {"cv"})
            input_width {int} -- [input sample shape] (default: {32})
            input_channel {int} -- [input sample shape] (default: {3})
            n_output_node {int} -- [output node number] (default: {10})
            algorithm_name {str} -- [algorithm name used in the network morphism] (default: {"Bayesian"})
            optimize_mode {str} -- [optimize mode "minimize" or "maximize"] (default: {"minimize"})
            path {str} -- [default mode path to save the model file] (default: {"model_path"})
            verbose {bool} -- [verbose to print the log] (default: {True})
            metric {Class} -- [An instance of the Metric subclasses. Accuracy or MSE] (default: {Accuracy})
            beta {float} -- [The beta in acquisition function. (refer to our paper)] (default: {Constant.BETA})
            t_min {float} -- [The minimum temperature for simulated annealing.] (default: {Constant.T_MIN})
            max_model_size {int} -- [max model size to the graph] (default: {Constant.MAX_MODEL_SIZE})
            default_model_len {int} -- [default model length] (default: {Constant.MODEL_LEN})
            default_model_width {int} -- [default model width] (default: {Constant.MODEL_WIDTH})
        """

        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(os.getcwd(), path)
        if task == "cv":
            self.generators = [CnnGenerator]
        elif task == "common":
            self.generators = [MlpGenerator]
        else:
            raise NotImplementedError('{} task not supported in List ["cv","common"]')

        self.n_classes = n_output_node
        self.input_shape = (input_width,input_width,input_channel)

        self.t_min = t_min
        self.metric = metric
        self.beta = beta
        self.algorithm_name = algorithm_name
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.json = None
        self.total_data = {}
        self.verbose = verbose
        self.model_count = 0

        self.bo = BayesianOptimizer(self, self.t_min, self.metric, self.beta)
        self.training_queue = []
        self.x_queue = []
        self.y_queue = []
        self.descriptors = []
        self.history = []

        self.max_model_size = max_model_size
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width

        self.search_space = dict()

    def update_search_space(self, search_space):
        """
        Update search space definition in tuner by search_space in neural architecture.
        """
        self.search_space = search_space

    def generate_parameters(self, parameter_id):
        """
        Returns a set of trial neural architecture, as a serializable object.
        parameter_id : int
        """
        if not self.history:
            self.init_search()

        new_father_id = None
        generated_graph = None
        if not self.training_queue:
            new_father_id, generated_graph = self.generate()
            new_model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((generated_graph, new_father_id, new_model_id))
            self.descriptors.append(generated_graph.extract_descriptor())

        graph, father_id, model_id = self.training_queue.pop(0)

        # from gragh to onnx
        # onnx_model_path = os.path.join(self.path, str(model_id) + '.onnx')
        # onnx_out = graph_to_onnx(graph,onnx_model_path,self.input_shape)
        # self.total_data[parameter_id] = (onnx_model_path, father_id, model_id)

        # from graph to json
        # json_model_path = os.path.join(self.path, str(model_id) + '.json')
        # json_out = graph_to_json(graph,json_model_path,self.input_shape)
        # self.total_data[parameter_id] = (json_model_path, father_id, model_id)

        # from graph to pickle file
        pickle_path = os.path.join(self.path, str(model_id) + ".graph")
        pickle_to_file(graph, pickle_path)
        self.total_data[parameter_id] = (pickle_path, father_id, model_id)

        return pickle_path

    def receive_trial_result(self, parameter_id, parameters, value):
        """ Record an observation of the objective function

        Arguments:           
            parameter_id : int
            parameters : dict of parameters
            value: final metrics of the trial, including reward     

        Raises:
            RuntimeError -- Received parameter_id not in total_data.
        """

        reward = self.extract_scalar_reward(value)

        if parameter_id not in self.total_data:
            raise RuntimeError("Received parameter_id not in total_data.")

        (_, father_id, model_id) = self.total_data[parameter_id]

        if self.optimize_mode is OptimizeMode.Maximize:
            reward = -reward

        # from onnx_model_path to gragh
        # onnx_graph = onnx.load(model_path)
        # graph = onnx_to_graph(onnx_graph,self.input_shape)

        # from json_model_path to gragh
        # json_graph = json.loads(model_path)
        # graph = json_to_graph(json_graph,self.input_shape)

        graph = self.bo.searcher.load_model_by_id(model_id)

        # to use the value and graph
        self.add_model(value, graph, model_id)
        self.update(father_id, graph, value, model_id)

        pickle_to_file(self, os.path.join(self.path, "searcher"))

    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""
        if self.verbose:
            logger.info("Initializing search.")
        for generator in self.generators:
            graph = generator(self.n_classes, self.input_shape).generate(
                self.default_model_len, self.default_model_width
            )
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())

        if self.verbose:
            logger.info("Initialization finished.")

    def generate(self):
        """Generate the next neural architecture.
        Returns:
            other_info: Anything to be saved in the training queue together with the architecture.
            generated_graph: An instance of Graph.
        """
        generated_graph, new_father_id = self.bo.generate(self.descriptors)
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](
                self.n_classes, self.input_shape
            ).generate(self.default_model_len, self.default_model_width)

        return new_father_id, generated_graph

    def update(self, other_info, graph, metric_value, model_id):
        """ Update the controller with evaluation result of a neural architecture.
        Args:
            other_info: Anything. In our case it is the father ID in the search tree.
            graph: An instance of Graph. The trained neural architecture.
            metric_value: The final evaluated metric value.
            model_id: An integer.
        """
        father_id = other_info
        self.bo.fit([graph.extract_descriptor()], [metric_value])
        self.bo.add_child(father_id, model_id)

    def add_model(self, metric_value, graph, model_id):
        """ add model to the history, x_queue and y_queue

        Arguments:
            metric_value: int --metric_value
            graph: dict -- graph
            model_id: int -- model_id

        Returns:
            model dict
        """

        if self.verbose:
            logger.info("Saving model.")

        pickle_to_file(graph, os.path.join(self.path, str(model_id) + ".graph"))

        # Update best_model text file
        ret = {"model_id": model_id, "metric_value": metric_value}
        self.history.append(ret)
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, "best_model.txt"), "w")
            file.write("best model: " + str(model_id))
            file.close()

        if self.verbose:
            idx = ["model_id", "metric_value"]
            header = ["Model ID", "Metric Value"]
            line = "|".join(x.center(24) for x in header)
            logger.info("+" + "-" * len(line) + "+")
            logger.info("|" + line + "|")

            if self.history:
                r = self.history[-1]
                logger.info("+" + "-" * len(line) + "+")
                line = "|".join(str(r[x]).center(24) for x in idx)
                logger.info("|" + line + "|")
            logger.info("+" + "-" * len(line) + "+")

        descriptor = graph.extract_descriptor()
        self.x_queue.append(descriptor)
        self.y_queue.append(metric_value)
        return ret

    def get_best_model_id(self):
        """ get the best model_id from history using the metric value

        Returns:
            int -- the best model_id
        """

        if self.metric.higher_better():
            return max(self.history, key=lambda x: x["metric_value"])["model_id"]
        return min(self.history, key=lambda x: x["metric_value"])["model_id"]

    def load_model_by_id(self, model_id):
        return pickle_from_file(os.path.join(self.path, str(model_id) + ".graph"))

    def load_best_model(self):
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        for item in self.history:
            if item["model_id"] == model_id:
                return item["metric_value"]
        return None
