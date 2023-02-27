# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import defaultdict

from nni import NoMoreTrialError
from nni.assessor import AssessResult

from .common import multi_phase_enabled
from .env_vars import dispatcher_env_vars
from .msg_dispatcher_base import MsgDispatcherBase
from .tuner_command_channel import CommandType
from ..common.serializer import dump, load
from ..utils import MetricType

_logger = logging.getLogger(__name__)

# Assessor global variables
_trial_history = defaultdict(dict)
'''key: trial job ID; value: intermediate results, mapping from sequence number to data'''

_ended_trials = set()
'''trial_job_id of all ended trials.
We need this because NNI manager may send metrics after reporting a trial ended.
TODO: move this logic to NNI manager
'''


def _sort_history(history):
    ret = []
    for i, _ in enumerate(history):
        if i in history:
            ret.append(history[i])
        else:
            break
    return ret


# Tuner global variables
_next_parameter_id = 0
_trial_params = {}
'''key: parameter ID; value: parameters'''
_customized_parameter_ids = set()


def _create_parameter_id():
    global _next_parameter_id
    _next_parameter_id += 1
    return _next_parameter_id - 1


def _pack_parameter(parameter_id, params, customized=False, trial_job_id=None, parameter_index=None):
    _trial_params[parameter_id] = params
    ret = {
        'parameter_id': parameter_id,
        'parameter_source': 'customized' if customized else 'algorithm',
        'parameters': params
    }
    if trial_job_id is not None:
        ret['trial_job_id'] = trial_job_id
    if parameter_index is not None:
        ret['parameter_index'] = parameter_index
    else:
        ret['parameter_index'] = 0
    return dump(ret)


class MsgDispatcher(MsgDispatcherBase):
    def __init__(self, command_channel_url, tuner, assessor=None):
        super().__init__(command_channel_url)
        self.tuner = tuner
        self.assessor = assessor
        if assessor is None:
            _logger.debug('Assessor is not configured')

    def load_checkpoint(self):
        self.tuner.load_checkpoint()
        if self.assessor is not None:
            self.assessor.load_checkpoint()

    def save_checkpoint(self):
        self.tuner.save_checkpoint()
        if self.assessor is not None:
            self.assessor.save_checkpoint()

    def handle_initialize(self, data):
        """Data is search space
        """
        _logger.info('Initial search space: %s', data)
        self.tuner.update_search_space(data)
        self.send(CommandType.Initialized, '')

    def send_trial_callback(self, id_, params):
        """For tuner to issue trial config when the config is generated
        """
        self.send(CommandType.NewTrialJob, _pack_parameter(id_, params))

    def handle_request_trial_jobs(self, data):
        # data: number or trial jobs
        ids = [_create_parameter_id() for _ in range(data)]
        _logger.debug("requesting for generating params of %s", ids)
        params_list = self.tuner.generate_multiple_parameters(ids, st_callback=self.send_trial_callback)

        for i, _ in enumerate(params_list):
            self.send(CommandType.NewTrialJob, _pack_parameter(ids[i], params_list[i]))
        # when parameters is None.
        if len(params_list) < len(ids):
            self.send(CommandType.NoMoreTrialJobs, _pack_parameter(ids[0], ''))

    def handle_update_search_space(self, data):
        _logger.info('New search space: %s', data)
        self.tuner.update_search_space(data)

    def handle_import_data(self, data):
        """Import additional data for tuning
        data: a list of dictionaries, each of which has at least two keys, 'parameter' and 'value'
        """
        for entry in data:
            entry['value'] = entry['value'] if type(entry['value']) is str else dump(entry['value'])
            entry['value'] = load(entry['value'])
        self.tuner.import_data(data)

    def handle_add_customized_trial(self, data):
        global _next_parameter_id
        # data: parameters
        previous_max_param_id = self.recover_parameter_id(data)
        _next_parameter_id = previous_max_param_id + 1

    def handle_report_metric_data(self, data):
        """
        data: a dict received from nni_manager, which contains:
              - 'parameter_id': id of the trial
              - 'value': metric value reported by nni.report_final_result()
              - 'type': report type, support {'FINAL', 'PERIODICAL'}
        """
        if self.is_created_in_previous_exp(data['parameter_id']):
            if data['type'] == MetricType.FINAL:
                # only deal with final metric using import data
                param = self.get_previous_param(data['parameter_id'])
                trial_data = [{'parameter': param, 'value': load(data['value'])}]
                self.handle_import_data(trial_data)
            return
        # metrics value is dumped as json string in trial, so we need to decode it here
        if 'value' in data:
            data['value'] = load(data['value'])
        if data['type'] == MetricType.FINAL:
            self._handle_final_metric_data(data)
        elif data['type'] == MetricType.PERIODICAL:
            if self.assessor is not None:
                self._handle_intermediate_metric_data(data)
        elif data['type'] == MetricType.REQUEST_PARAMETER:
            assert multi_phase_enabled()
            assert data['trial_job_id'] is not None
            assert data['parameter_index'] is not None
            param_id = _create_parameter_id()
            try:
                param = self.tuner.generate_parameters(param_id, trial_job_id=data['trial_job_id'])
            except NoMoreTrialError:
                param = None
            self.send(CommandType.SendTrialJobParameter, _pack_parameter(param_id, param, trial_job_id=data['trial_job_id'],
                                                                    parameter_index=data['parameter_index']))
        else:
            raise ValueError('Data type not supported: {}'.format(data['type']))

    def handle_trial_end(self, data):
        """
        data: it has three keys: trial_job_id, event, hyper_params
             - trial_job_id: the id generated by training service
             - event: the job's state
             - hyper_params: the hyperparameters generated and returned by tuner
        """
        id_ = load(data['hyper_params'])['parameter_id']
        if self.is_created_in_previous_exp(id_):
            # The end of the recovered trial is ignored
            return
        trial_job_id = data['trial_job_id']
        _ended_trials.add(trial_job_id)
        if trial_job_id in _trial_history:
            _trial_history.pop(trial_job_id)
            if self.assessor is not None:
                self.assessor.trial_end(trial_job_id, data['event'] == 'SUCCEEDED')
        if self.tuner is not None:
            self.tuner.trial_end(id_, data['event'] == 'SUCCEEDED')

    def _handle_final_metric_data(self, data):
        """Call tuner to process final results
        """
        id_ = data['parameter_id']
        value = data['value']
        if id_ is None or id_ in _customized_parameter_ids:
            if not hasattr(self.tuner, '_accept_customized'):
                self.tuner._accept_customized = False
            if not self.tuner._accept_customized:
                _logger.info('Customized trial job %s ignored by tuner', id_)
                return
            customized = True
        else:
            customized = False
        if id_ in _trial_params:
            self.tuner.receive_trial_result(id_, _trial_params[id_], value, customized=customized,
                                            trial_job_id=data.get('trial_job_id'))
        else:
            _logger.warning('Find unknown job parameter id %s, maybe something goes wrong.', id_)
            _logger.warning('_trial_params %s', _trial_params)

    def _handle_intermediate_metric_data(self, data):
        """Call assessor to process intermediate results
        """
        if data['type'] != MetricType.PERIODICAL:
            return
        if self.assessor is None:
            return

        trial_job_id = data['trial_job_id']
        if trial_job_id in _ended_trials:
            return

        history = _trial_history[trial_job_id]
        history[data['sequence']] = data['value']
        ordered_history = _sort_history(history)
        if len(ordered_history) < data['sequence']:  # no user-visible update since last time
            return

        try:
            result = self.assessor.assess_trial(trial_job_id, ordered_history)
        except Exception as e:
            _logger.error('Assessor error')
            _logger.exception(e)
            raise

        if isinstance(result, bool):
            result = AssessResult.Good if result else AssessResult.Bad
        elif not isinstance(result, AssessResult):
            msg = 'Result of Assessor.assess_trial must be an object of AssessResult, not %s'
            raise RuntimeError(msg % type(result))

        if result is AssessResult.Bad:
            _logger.debug('BAD, kill %s', trial_job_id)
            self.send(CommandType.KillTrialJob, dump(trial_job_id))
            # notify tuner
            _logger.debug('env var: NNI_INCLUDE_INTERMEDIATE_RESULTS: [%s]',
                          dispatcher_env_vars.NNI_INCLUDE_INTERMEDIATE_RESULTS)
            if dispatcher_env_vars.NNI_INCLUDE_INTERMEDIATE_RESULTS == 'true':
                self._earlystop_notify_tuner(data)
        else:
            _logger.debug('GOOD')

    def _earlystop_notify_tuner(self, data):
        """Send last intermediate result as final result to tuner in case the
        trial is early stopped.
        """
        _logger.debug('Early stop notify tuner data: [%s]', data)
        data['type'] = MetricType.FINAL
        data['value'] = dump(data['value'])
        self.enqueue_command(CommandType.ReportMetricData, data)
