import os
import sys
from pathlib import Path
import time

from nni.tuner import Tuner


class NasTuner(Tuner):
    def __init__(self):
        tuner_main(self)
        self.graphs = []
        self._param_id_to_graph = {}

    def generate_parameters(self, param_id, **kwargs):
        while not self.graphs:
            time.sleep(1)
        graph = self.graphs.pop(0)
        self._param_id_to_graph[param_id] = graph
        return graph.id

    def receive_trial_result(self, param_id, params, value, **kwargs):
        graph = self._param_id_to_graph.pop(param_id)
        graph.metrics = value

    def update_search_space(self, *args):
        pass


    def enqueue(self, graph):
        # this will be invoked from another thread
        self.graphs.append(graph)


def tuner_main(tuner):
    # add root of this git repo to path
    # this makes it impossible to "install" the sdk (FIXME)
    # the fix should go to NNI side
    path = Path(__file__).parents[2]
    os.chdir(path)
    sys.path.insert(0, str(path))
    from sdk.nni_integration.tuner_global import init
    init(tuner)
