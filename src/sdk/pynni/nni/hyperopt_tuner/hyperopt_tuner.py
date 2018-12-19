# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
hyperopt_tuner.py
'''

import copy
import logging

from enum import Enum, unique
import numpy as np

import hyperopt as hp
from nni.tuner import Tuner

logger = logging.getLogger('hyperopt_AutoML')


@unique
class OptimizeMode(Enum):
    '''
    Oprimize Mode class
    '''
    Minimize = 'minimize'
    Maximize = 'maximize'


ROOT = 'root'
TYPE = '_type'
VALUE = '_value'
INDEX = '_index'


def json2space(in_x, name=ROOT):
    '''
    Change json to search space in hyperopt.
    '''
    out_y = copy.deepcopy(in_x)
    if isinstance(in_x, dict):
        if TYPE in in_x.keys():
            _type = in_x[TYPE]
            name = name + '-' + _type
            _value = json2space(in_x[VALUE], name=name)
            if _type == 'choice':
                out_y = eval('hp.hp.'+_type)(name, _value)
            else:
                if _type in ['loguniform', 'qloguniform']:
                    _value[:2] = np.log(_value[:2])
                out_y = eval('hp.hp.' + _type)(name, *_value)
        else:
            out_y = dict()
            for key in in_x.keys():
                out_y[key] = json2space(in_x[key], name+'[%s]' % str(key))
    elif isinstance(in_x, list):
        out_y = list()
        for i, x_i in enumerate(in_x):
            out_y.append(json2space(x_i, name+'[%d]' % i))
    else:
        logger.info('in_x is not a dict or a list in json2space fuinction %s', str(in_x))
    return out_y


def json2parameter(in_x, parameter, name=ROOT):
    '''
    Change json to parameters.
    '''
    out_y = copy.deepcopy(in_x)
    if isinstance(in_x, dict):
        if TYPE in in_x.keys():
            _type = in_x[TYPE]
            name = name + '-' + _type
            if _type == 'choice':
                _index = parameter[name]
                out_y = {
                    INDEX: _index,
                    VALUE: json2parameter(in_x[VALUE][_index], parameter, name=name+'[%d]' % _index)
                }
            else:
                out_y = parameter[name]
        else:
            out_y = dict()
            for key in in_x.keys():
                out_y[key] = json2parameter(
                    in_x[key], parameter, name + '[%s]' % str(key))
    elif isinstance(in_x, list):
        out_y = list()
        for i, x_i in enumerate(in_x):
            out_y.append(json2parameter(x_i, parameter, name + '[%d]' % i))
    else:
        logger.info('in_x is not a dict or a list in json2space fuinction %s', str(in_x))
    return out_y


def json2vals(in_x, vals, out_y, name=ROOT):
    if isinstance(in_x, dict):
        if TYPE in in_x.keys():
            _type = in_x[TYPE]
            name = name + '-' + _type

            try:
                out_y[name] = vals[INDEX]
            # TODO - catch exact Exception
            except Exception:
                out_y[name] = vals

            if _type == 'choice':
                _index = vals[INDEX]
                json2vals(in_x[VALUE][_index], vals[VALUE],
                          out_y, name=name + '[%d]' % _index)
        else:
            for key in in_x.keys():
                json2vals(in_x[key], vals[key], out_y, name + '[%s]' % str(key))
    elif isinstance(in_x, list):
        for i, temp in enumerate(in_x):
            json2vals(temp, vals[i], out_y, name + '[%d]' % i)


def _split_index(params):
    result = {}
    for key in params:
        if isinstance(params[key], dict):
            value = params[key][VALUE]
        else:
            value = params[key]
        result[key] = value
    return result


class HyperoptTuner(Tuner):
    '''
    HyperoptTuner is a tuner which using hyperopt algorithm.
    '''
    
    def __init__(self, algorithm_name, optimize_mode):
        self.algorithm_name = algorithm_name
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.json = None
        self.total_data = {}
        self.rval = None

    def _choose_tuner(self, algorithm_name):
        if algorithm_name == 'tpe':
            return hp.tpe.suggest
        if algorithm_name == 'random_search':
            return hp.rand.suggest
        if algorithm_name == 'anneal':
            return hp.anneal.suggest
        raise RuntimeError('Not support tuner algorithm in hyperopt.')

    def update_search_space(self, search_space):
        '''
        Update search space definition in tuner by search_space in parameters.
        '''
        #assert self.json is None

        self.json = search_space
        search_space_instance = json2space(self.json)
        rstate = np.random.RandomState()
        trials = hp.Trials()
        domain = hp.Domain(None, search_space_instance,
                           pass_expr_memo_ctrl=None)
        algorithm = self._choose_tuner(self.algorithm_name)
        self.rval = hp.FMinIter(algorithm, domain, trials,
                                max_evals=-1, rstate=rstate, verbose=0)
        self.rval.catch_eval_exceptions = False

    def generate_parameters(self, parameter_id):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object.
        parameter_id : int
        '''
        rval = self.rval
        trials = rval.trials
        algorithm = rval.algo
        new_ids = rval.trials.new_trial_ids(1)
        rval.trials.refresh()
        random_state = rval.rstate.randint(2**31-1)
        new_trials = algorithm(new_ids, rval.domain, trials, random_state)
        rval.trials.refresh()
        vals = new_trials[0]['misc']['vals']
        parameter = dict()
        for key in vals:
            try:
                parameter[key] = vals[key][0].item()
            except Exception:
                parameter[key] = None

        # remove '_index' from json2parameter and save params-id
        total_params = json2parameter(self.json, parameter)
        self.total_data[parameter_id] = total_params
        params = _split_index(total_params)
        return params

    def receive_trial_result(self, parameter_id, parameters, value):
        '''
        Record an observation of the objective function
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial, including reward
        '''
        reward = self.extract_scalar_reward(value)
        # restore the paramsters contains '_index'
        if parameter_id not in self.total_data:
            raise RuntimeError('Received parameter_id not in total_data.')
        params = self.total_data[parameter_id]

        if self.optimize_mode is OptimizeMode.Maximize:
            reward = -reward

        rval = self.rval
        domain = rval.domain
        trials = rval.trials

        new_id = len(trials)

        rval_specs = [None]
        rval_results = [domain.new_result()]
        rval_miscs = [dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)]

        vals = params
        idxs = dict()

        out_y = dict()
        json2vals(self.json, vals, out_y)
        vals = out_y
        for key in domain.params:
            if key in [VALUE, INDEX]:
                continue
            if key not in vals or vals[key] is None or vals[key] == []:
                idxs[key] = vals[key] = []
            else:
                idxs[key] = [new_id]
                vals[key] = [vals[key]]

        self.miscs_update_idxs_vals(rval_miscs, idxs, vals,
                                    idxs_map={new_id: new_id},
                                    assert_all_vals_used=False)

        trial = trials.new_trial_docs([new_id], rval_specs, rval_results, rval_miscs)[0]
        trial['result'] = {'loss': reward, 'status': 'ok'}
        trial['state'] = hp.JOB_STATE_DONE
        trials.insert_trial_docs([trial])
        trials.refresh()

    def miscs_update_idxs_vals(self, miscs, idxs, vals,
                               assert_all_vals_used=True,
                               idxs_map=None):
        '''
        Unpack the idxs-vals format into the list of dictionaries that is
        `misc`.

        idxs_map: a dictionary of id->id mappings so that the misc['idxs'] can
            contain different numbers than the idxs argument. XXX CLARIFY
        '''
        if idxs_map is None:
            idxs_map = {}

        assert set(idxs.keys()) == set(vals.keys())

        misc_by_id = {m['tid']: m for m in miscs}
        for m in miscs:
            m['idxs'] = dict([(key, []) for key in idxs])
            m['vals'] = dict([(key, []) for key in idxs])

        for key in idxs:
            assert len(idxs[key]) == len(vals[key])
            for tid, val in zip(idxs[key], vals[key]):
                tid = idxs_map.get(tid, tid)
                if assert_all_vals_used or tid in misc_by_id:
                    misc_by_id[tid]['idxs'][key] = [tid]
                    misc_by_id[tid]['vals'][key] = [val]
