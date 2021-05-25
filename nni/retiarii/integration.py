# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, Callable

from nni.runtime.msg_dispatcher_base import MsgDispatcherBase
from nni.runtime.protocol import CommandType, send
from nni.utils import MetricType

from .graph import MetricData
from .integration_api import register_advisor
from .serializer import json_dumps, json_loads

_logger = logging.getLogger(__name__)


class RetiariiAdvisor(MsgDispatcherBase):
    """
    The class is to connect Retiarii components to NNI backend.

    It will function as the main thread when running a Retiarii experiment through NNI.
    Strategy will be launched as its thread, who will call APIs in execution engine. Execution
    engine will then find the advisor singleton and send payloads to advisor.

    When metrics are sent back, advisor will first receive the payloads, who will call the callback
    function (that is a member function in graph listener).

    The conversion advisor provides are minimum. It is only a send/receive module, and execution engine
    needs to handle all the rest.

    FIXME
        How does advisor exit when strategy exists?

    Attributes
    ----------
    send_trial_callback

    request_trial_jobs_callback

    trial_end_callback

    intermediate_metric_callback

    final_metric_callback
    """

    def __init__(self):
        super(RetiariiAdvisor, self).__init__()
        register_advisor(self)  # register the current advisor as the "global only" advisor
        self.search_space = None

        self.send_trial_callback: Callable[[dict], None] = None
        self.request_trial_jobs_callback: Callable[[int], None] = None
        self.trial_end_callback: Callable[[int, bool], None] = None
        self.intermediate_metric_callback: Callable[[int, MetricData], None] = None
        self.final_metric_callback: Callable[[int, MetricData], None] = None

        self.parameters_count = 0

    def handle_initialize(self, data):
        """callback for initializing the advisor
        Parameters
        ----------
        data: dict
            search space
        """
        self.handle_update_search_space(data)
        send(CommandType.Initialized, '')

    def send_trial(self, parameters):
        """
        Send parameters to NNI.

        Parameters
        ----------
        parameters : Any
            Any payload.

        Returns
        -------
        int
            Parameter ID that is assigned to this parameter,
            which will be used for identification in future.
        """
        self.parameters_count += 1
        new_trial = {
            'parameter_id': self.parameters_count,
            'parameters': parameters,
            'parameter_source': 'algorithm'
        }
        _logger.debug('New trial sent: %s', new_trial)
        send(CommandType.NewTrialJob, json_dumps(new_trial))
        if self.send_trial_callback is not None:
            self.send_trial_callback(parameters)  # pylint: disable=not-callable
        return self.parameters_count

    def mark_experiment_as_ending(self):
        send(CommandType.NoMoreTrialJobs, '')

    def handle_request_trial_jobs(self, num_trials):
        _logger.debug('Request trial jobs: %s', num_trials)
        if self.request_trial_jobs_callback is not None:
            self.request_trial_jobs_callback(num_trials)  # pylint: disable=not-callable

    def handle_update_search_space(self, data):
        _logger.debug('Received search space: %s', data)
        self.search_space = data

    def handle_trial_end(self, data):
        _logger.debug('Trial end: %s', data)
        self.trial_end_callback(json_loads(data['hyper_params'])['parameter_id'],  # pylint: disable=not-callable
                                data['event'] == 'SUCCEEDED')

    def handle_report_metric_data(self, data):
        _logger.debug('Metric reported: %s', data)
        if data['type'] == MetricType.REQUEST_PARAMETER:
            raise ValueError('Request parameter not supported')
        elif data['type'] == MetricType.PERIODICAL:
            self.intermediate_metric_callback(data['parameter_id'],  # pylint: disable=not-callable
                                              self._process_value(data['value']))
        elif data['type'] == MetricType.FINAL:
            self.final_metric_callback(data['parameter_id'],  # pylint: disable=not-callable
                                       self._process_value(data['value']))

    @staticmethod
    def _process_value(value) -> Any:  # hopefully a float
        value = json_loads(value)
        if isinstance(value, dict):
            if 'default' in value:
                return value['default']
            else:
                return value
        return value
