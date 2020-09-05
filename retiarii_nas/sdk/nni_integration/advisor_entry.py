import os
import sys
from pathlib import Path
import time
import json_tricks
import logging

from nni.msg_dispatcher_base import MsgDispatcherBase
from nni.protocol import CommandType, send

# add root of this git repo to path
# this makes it impossible to "install" the sdk (FIXME)
# the fix should go to NNI side
path = Path(__file__).parents[2]
os.chdir(path)
sys.path.insert(0, str(path))
from sdk import compile_opt

_logger = logging.getLogger(__name__)

class NasAdvisor(MsgDispatcherBase):
    def __init__(self, command):
        super(NasAdvisor, self).__init__()
        self.command = command
        self.graphs: 'List[Graph]' = []
        self._param_id_to_opt_graph = {}
        self.param_id = 0
        # use this to track available resources, similar to credit
        self._available_resource = 0
        
        advisor_main(self)

    def handle_initialize(self, data):
        """callback for initializing the advisor
        Parameters
        ----------
        data: dict
            search space
        """
        self.handle_update_search_space(data)
        send(CommandType.Initialized, '')

    def handle_update_search_space(self, data):
        pass

    def handle_report_metric_data(self, data):
        opt_graph = self._param_id_to_opt_graph[data['parameter_id']]
        assert len(opt_graph.graphs) == 1
        graph = opt_graph.graphs[0]
        _logger.info('result addr: {}, {}'.format(graph, opt_graph.graphs[0]))
        graph.metrics = data['value']
        _logger.info('receive trial result: {}'.format(data['parameter_id']))

    def handle_request_trial_jobs(self, data):
        """
        this function can be seen as notified with available resource.
        thus, it includes resource allocation/planning

        Parameters
        ----------
        data : int
            number of trials to request
        """
        while not self.graphs:
            time.sleep(1)
        # currently the resource is only gpu, and data means released gpu number
        self._available_resource += data

        while self._available_resource > 0 and self.graphs:
            graphs = self.graphs[0]
            if not isinstance(graphs, list):
                graphs = [graphs]
            opt_graphs, left_graphs = compile_opt.optimize_graphs(graphs, self._available_resource)
            if left_graphs is None:
                self.graphs.pop(0)
            else:
                self.graphs[0] = left_graphs
            consumed_resource = 0
            for graph in opt_graphs:
                # opt_graph is OptGraph
                consumed_resource += graph.resource
                self.param_id += 1
                self._param_id_to_opt_graph[self.param_id] = graph
                trial = {'parameter_id': self.param_id, 'parameters': graph.opt_graph.dump(), 'parameter_source': 'algorithm'}
                send(CommandType.NewTrialJob, json_tricks.dumps(trial))
                _logger.info('send the trial')
            self._available_resource -= consumed_resource
            assert self._available_resource >= 0

    def handle_trial_end(self, data):
        pass

    def enqueue(self, graph):
        # this will be invoked from another thread
        _logger.info('enqueue one graph')
        self.graphs.append(graph)

    def handle_import_data(self, data):
        pass

def advisor_main(advisor):
    from sdk.nni_integration.advisor_global import init
    init(advisor)
