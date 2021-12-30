# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from typing import Any, Callable

import nni
from nni.common.serializer import PayloadTooLarge
from nni.runtime.msg_dispatcher_base import MsgDispatcherBase
from nni.runtime.protocol import CommandType, send
from nni.utils import MetricType

from .graph import MetricData
from .integration_api import register_advisor

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

    def _validate_placement_constraint(self, placement_constraint):
        if placement_constraint is None:
            raise ValueError('placement_constraint is None')
        if not 'type' in placement_constraint:
            raise ValueError('placement_constraint must have `type`')
        if not 'gpus' in placement_constraint:
            raise ValueError('placement_constraint must have `gpus`')
        if placement_constraint['type'] not in ['None', 'GPUNumber', 'Device']:
            raise ValueError('placement_constraint.type must be either `None`,. `GPUNumber` or `Device`')
        if placement_constraint['type'] == 'None' and len(placement_constraint['gpus']) > 0:
            raise ValueError('placement_constraint.gpus must be an empty list when type == None')
        if placement_constraint['type'] == 'GPUNumber':
            if len(placement_constraint['gpus']) != 1:
                raise ValueError('placement_constraint.gpus currently only support one host when type == GPUNumber')
            for e in placement_constraint['gpus']:
                if not isinstance(e, int):
                    raise ValueError('placement_constraint.gpus must be a list of number when type == GPUNumber')
        if placement_constraint['type'] == 'Device':
            for e in placement_constraint['gpus']:
                if not isinstance(e, tuple):
                    raise ValueError('placement_constraint.gpus must be a list of tuple when type == Device')
                if not (len(e) == 2 and isinstance(e[0], str) and isinstance(e[1], int)):
                    raise ValueError('placement_constraint.gpus`s tuple must be (str, int)')

    def send_trial(self, parameters, placement_constraint=None):
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
        if placement_constraint is None:
            placement_constraint = {
                'type': 'None',
                'gpus': []
            }
        self._validate_placement_constraint(placement_constraint)
        new_trial = {
            'parameter_id': self.parameters_count,
            'parameters': parameters,
            'parameter_source': 'algorithm',
            'placement_constraint': placement_constraint
        }
        _logger.debug('New trial sent: %s', new_trial)

        try:
            send_payload = nni.dump(new_trial, pickle_size_limit=int(os.getenv('PICKLE_SIZE_LIMIT', 64 * 1024)))
        except PayloadTooLarge:
            raise ValueError(
                'Serialization failed when trying to dump the model because payload too large (larger than 64 KB). '
                'This is usually caused by pickling large objects (like datasets) by mistake. '
                'See the full error traceback for details and https://nni.readthedocs.io/en/stable/NAS/Serialization.html '
                'for how to resolve such issue. '
            )

        # trial parameters can be super large, disable pickle size limit here
        # nevertheless, there could still be blocked by pipe / nni-manager
        send(CommandType.NewTrialJob, send_payload)

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
        self.trial_end_callback(nni.load(data['hyper_params'])['parameter_id'],  # pylint: disable=not-callable
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
        value = nni.load(value)
        if isinstance(value, dict):
            if 'default' in value:
                return value['default']
            else:
                return value
        return value
