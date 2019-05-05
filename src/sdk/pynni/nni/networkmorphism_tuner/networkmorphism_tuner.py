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

import logging
import os


from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward
from nni.networkmorphism_tuner.bayesian import BayesianOptimizer
from nni.networkmorphism_tuner.nn import CnnGenerator, MlpGenerator
from nni.networkmorphism_tuner.utils import Constant

from nni.networkmorphism_tuner.graph import graph_to_json, json_to_graph

logger = logging.getLogger("NetworkMorphism_AutoML")


class NetworkMorphismTuner(Tuner):
    """NetworkMorphismTuner is a tuner which using network morphism techniques."""

    def __init__(
            self,
            task="cv",
            input_width=32,
            input_channel=3,
            n_output_node=10,
            algorithm_name="Bayesian",
            optimize_mode="maximize",
            path="model_path",
            verbose=True,
            beta=Constant.BETA,
            t_min=Constant.T_MIN,
            max_model_size=Constant.MAX_MODEL_SIZE,
            default_model_len=Constant.MODEL_LEN,
            default_model_width=Constant.MODEL_WIDTH,
    ):
        """ initilizer of the NetworkMorphismTuner.

        Parameters
        ----------
        task : str
            task mode, such as "cv","common" etc. (default: {"cv"})
        input_width : int
            input sample shape (default: {32})
        input_channel : int
            input sample shape (default: {3})
        n_output_node : int
            output node number (default: {10})
        algorithm_name : str
            algorithm name used in the network morphism (default: {"Bayesian"})
        optimize_mode : str
            optimize mode "minimize" or "maximize" (default: {"minimize"})
        path : str
            default mode path to save the model file (default: {"model_path"})  
        verbose : bool
            verbose to print the log (default: {True})
        beta : float
            The beta in acquisition function. (default: {Constant.BETA})
        t_min : float
            The minimum temperature for simulated annealing. (default: {Constant.T_MIN})
        max_model_size : int
            max model size to the graph (default: {Constant.MAX_MODEL_SIZE})
        default_model_len : int
            default model length (default: {Constant.MODEL_LEN})
        default_model_width : int
            default model width (default: {Constant.MODEL_WIDTH})
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
        self.input_shape = (input_width, input_width, input_channel)

        self.t_min = t_min
        self.beta = beta
        self.algorithm_name = algorithm_name
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.json = None
        self.total_data = {}
        self.verbose = verbose
        self.model_count = 0

        self.bo = BayesianOptimizer(self, self.t_min, self.optimize_mode, self.beta)
        self.training_queue = []
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

        Parameters
        ----------
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

        # from graph to json
        json_model_path = os.path.join(self.path, str(model_id) + ".json")
        json_out = graph_to_json(graph, json_model_path)
        self.total_data[parameter_id] = (json_out, father_id, model_id)

        return json_out

    def receive_trial_result(self, parameter_id, parameters, value):
        """ Record an observation of the objective function.
    
        Parameters
        ----------
        parameter_id : int
        parameters : dict
        value : dict/float
            if value is dict, it should have "default" key.
        """
        reward = extract_scalar_reward(value)

        if parameter_id not in self.total_data:
            raise RuntimeError("Received parameter_id not in total_data.")

        (_, father_id, model_id) = self.total_data[parameter_id]

        graph = self.bo.searcher.load_model_by_id(model_id)

        # to use the value and graph
        self.add_model(reward, model_id)
        self.update(father_id, graph, reward, model_id)

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

        Returns
        -------
        other_info: any object
            Anything to be saved in the training queue together with the architecture.
        generated_graph: Graph
            An instance of Graph.
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

        Parameters
        ----------
        other_info: any object
            In our case it is the father ID in the search tree.
        graph: Graph
            An instance of Graph. The trained neural architecture.
        metric_value: float
            The final evaluated metric value.
        model_id: int
        """
        father_id = other_info
        self.bo.fit([graph.extract_descriptor()], [metric_value])
        self.bo.add_child(father_id, model_id)

    def add_model(self, metric_value, model_id):
        """ Add model to the history, x_queue and y_queue

        Parameters
        ----------
        metric_value : float
        graph : dict
        model_id : int

        Returns
        -------
        model : dict
        """
        if self.verbose:
            logger.info("Saving model.")

        # Update best_model text file
        ret = {"model_id": model_id, "metric_value": metric_value}
        self.history.append(ret)
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, "best_model.txt"), "w")
            file.write("best model: " + str(model_id))
            file.close()
        return ret

    def get_best_model_id(self):
        """ Get the best model_id from history using the metric value
        """

        if self.optimize_mode is OptimizeMode.Maximize:
            return max(self.history, key=lambda x: x["metric_value"])["model_id"]
        return min(self.history, key=lambda x: x["metric_value"])["model_id"]

    def load_model_by_id(self, model_id):
        """Get the model by model_id

        Parameters
        ----------
        model_id : int
            model index
        
        Returns
        -------
        load_model : Graph
            the model graph representation
        """

        with open(os.path.join(self.path, str(model_id) + ".json")) as fin:
            json_str = fin.read().replace("\n", "")

        load_model = json_to_graph(json_str)
        return load_model

    def load_best_model(self):
        """ Get the best model by model id

        Returns
        -------
        load_model : Graph
            the model graph representation
        """
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        """ Get the model metric valud by its model_id

        Parameters
        ----------
        model_id : int
            model index
        
        Returns
        -------
        float
             the model metric
        """
        for item in self.history:
            if item["model_id"] == model_id:
                return item["metric_value"]
        return None

    def import_data(self, data):
        pass

