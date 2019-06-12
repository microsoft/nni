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
"""
hyperopt_tuner.py
"""

import copy
import logging

import hyperopt as hp
import numpy as np
from nni.tuner import Tuner
from nni.utils import NodeType, OptimizeMode, extract_scalar_reward, split_index, randint_to_quniform

logger = logging.getLogger('hyperopt_AutoML')


def json2space(in_x, name=NodeType.ROOT):
    """
    Change json to search space in hyperopt.

    Parameters
    ----------
    in_x : dict/list/str/int/float
        The part of json.
    name : str
        name could be NodeType.ROOT, NodeType.TYPE, NodeType.VALUE or NodeType.INDEX, NodeType.NAME.
    """
    out_y = copy.deepcopy(in_x)
    if isinstance(in_x, dict):
        if NodeType.TYPE in in_x.keys():
            _type = in_x[NodeType.TYPE]
            name = name + '-' + _type
            _value = json2space(in_x[NodeType.VALUE], name=name)
            if _type == 'choice':
                out_y = eval('hp.hp.choice')(name, _value)
            else:
                if _type in ['loguniform', 'qloguniform']:
                    _value[:2] = np.log(_value[:2])
                out_y = eval('hp.hp.' + _type)(name, *_value)
        else:
            out_y = dict()
            for key in in_x.keys():
                out_y[key] = json2space(in_x[key], name + '[%s]' % str(key))
    elif isinstance(in_x, list):
        out_y = list()
        for i, x_i in enumerate(in_x):
            if isinstance(x_i, dict):
                if NodeType.NAME not in x_i.keys():
                    raise RuntimeError(
                        '\'_name\' key is not found in this nested search space.'
                    )
            out_y.append(json2space(x_i, name + '[%d]' % i))
    return out_y


def json2parameter(in_x, parameter, name=NodeType.ROOT):
    """
    Change json to parameters.
    """
    out_y = copy.deepcopy(in_x)
    if isinstance(in_x, dict):
        if NodeType.TYPE in in_x.keys():
            _type = in_x[NodeType.TYPE]
            name = name + '-' + _type
            if _type == 'choice':
                _index = parameter[name]
                out_y = {
                    NodeType.INDEX:
                    _index,
                    NodeType.VALUE:
                    json2parameter(in_x[NodeType.VALUE][_index],
                                   parameter,
                                   name=name + '[%d]' % _index)
                }
            else:
                out_y = parameter[name]
        else:
            out_y = dict()
            for key in in_x.keys():
                out_y[key] = json2parameter(in_x[key], parameter,
                                            name + '[%s]' % str(key))
    elif isinstance(in_x, list):
        out_y = list()
        for i, x_i in enumerate(in_x):
            if isinstance(x_i, dict):
                if NodeType.NAME not in x_i.keys():
                    raise RuntimeError(
                        '\'_name\' key is not found in this nested search space.'
                    )
            out_y.append(json2parameter(x_i, parameter, name + '[%d]' % i))
    return out_y


def json2vals(in_x, vals, out_y, name=NodeType.ROOT):
    if isinstance(in_x, dict):
        if NodeType.TYPE in in_x.keys():
            _type = in_x[NodeType.TYPE]
            name = name + '-' + _type

            try:
                out_y[name] = vals[NodeType.INDEX]
            # TODO - catch exact Exception
            except Exception:
                out_y[name] = vals

            if _type == 'choice':
                _index = vals[NodeType.INDEX]
                json2vals(in_x[NodeType.VALUE][_index],
                          vals[NodeType.VALUE],
                          out_y,
                          name=name + '[%d]' % _index)
        else:
            for key in in_x.keys():
                json2vals(in_x[key], vals[key], out_y,
                          name + '[%s]' % str(key))
    elif isinstance(in_x, list):
        for i, temp in enumerate(in_x):
            # nested json
            if isinstance(temp, dict):
                if NodeType.NAME not in temp.keys():
                    raise RuntimeError(
                        '\'_name\' key is not found in this nested search space.'
                    )
                else:
                    json2vals(temp, vals[i], out_y, name + '[%d]' % i)
            else:
                json2vals(temp, vals[i], out_y, name + '[%d]' % i)


def _add_index(in_x, parameter):
    """
    change parameters in NNI format to parameters in hyperopt format(This function also support nested dict.).
    For example, receive parameters like:
        {'dropout_rate': 0.8, 'conv_size': 3, 'hidden_size': 512}
    Will change to format in hyperopt, like:
        {'dropout_rate': 0.8, 'conv_size': {'_index': 1, '_value': 3}, 'hidden_size': {'_index': 1, '_value': 512}}
    """
    if NodeType.TYPE not in in_x: # if at the top level
        out_y = dict()
        for key, value in parameter.items():
            out_y[key] = _add_index(in_x[key], value)
        return out_y
    elif isinstance(in_x, dict):
        value_type = in_x[NodeType.TYPE]
        value_format = in_x[NodeType.VALUE]
        if value_type == "choice":
            choice_name = parameter[0] if isinstance(parameter,
                                                     list) else parameter
            for pos, item in enumerate(
                    value_format):  # here value_format is a list
                if isinstance(
                        item,
                        list):  # this format is ["choice_key", format_dict]
                    choice_key = item[0]
                    choice_value_format = item[1]
                    if choice_key == choice_name:
                        return {
                            NodeType.INDEX: pos,
                            NodeType.VALUE: [
                                choice_name,
                                _add_index(choice_value_format, parameter[1])
                            ]
                        }
                elif choice_name == item:
                    return {NodeType.INDEX: pos, NodeType.VALUE: item}
        else:
            return parameter


class HyperoptTuner(Tuner):
    """
    HyperoptTuner is a tuner which using hyperopt algorithm.
    """

    def __init__(self, algorithm_name, optimize_mode='minimize'):
        """
        Parameters
        ----------
        algorithm_name : str
            algorithm_name includes "tpe", "random_search" and anneal".
        optimize_mode : str
        """
        self.algorithm_name = algorithm_name
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.json = None
        self.total_data = {}
        self.rval = None
        self.supplement_data_num = 0

    def _choose_tuner(self, algorithm_name):
        """
        Parameters
        ----------
        algorithm_name : str
            algorithm_name includes "tpe", "random_search" and anneal"
        """
        if algorithm_name == 'tpe':
            return hp.tpe.suggest
        if algorithm_name == 'random_search':
            return hp.rand.suggest
        if algorithm_name == 'anneal':
            return hp.anneal.suggest
        raise RuntimeError('Not support tuner algorithm in hyperopt.')

    def update_search_space(self, search_space):
        """
        Update search space definition in tuner by search_space in parameters.

        Will called when first setup experiemnt or update search space in WebUI.

        Parameters
        ----------
        search_space : dict
        """
        self.json = search_space
        randint_to_quniform(self.json)

        search_space_instance = json2space(self.json)
        rstate = np.random.RandomState()
        trials = hp.Trials()
        domain = hp.Domain(None,
                           search_space_instance,
                           pass_expr_memo_ctrl=None)
        algorithm = self._choose_tuner(self.algorithm_name)
        self.rval = hp.FMinIter(algorithm,
                                domain,
                                trials,
                                max_evals=-1,
                                rstate=rstate,
                                verbose=0)
        self.rval.catch_eval_exceptions = False

    def generate_parameters(self, parameter_id):
        """
        Returns a set of trial (hyper-)parameters, as a serializable object.

        Parameters
        ----------
        parameter_id : int

        Returns
        -------
        params : dict
        """
        total_params = self.get_suggestion(random_search=False)
        # avoid generating same parameter with concurrent trials because hyperopt doesn't support parallel mode
        if total_params in self.total_data.values():
            # but it can cause deplicate parameter rarely
            total_params = self.get_suggestion(random_search=True)
        self.total_data[parameter_id] = total_params
        params = split_index(total_params)
        return params

    def receive_trial_result(self, parameter_id, parameters, value):
        """
        Record an observation of the objective function

        Parameters
        ----------
        parameter_id : int
        parameters : dict
        value : dict/float
            if value is dict, it should have "default" key.
            value is final metrics of the trial.
        """
        reward = extract_scalar_reward(value)
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
            if key in [NodeType.VALUE, NodeType.INDEX]:
                continue
            if key not in vals or vals[key] is None or vals[key] == []:
                idxs[key] = vals[key] = []
            else:
                idxs[key] = [new_id]
                vals[key] = [vals[key]]

        self.miscs_update_idxs_vals(rval_miscs,
                                    idxs,
                                    vals,
                                    idxs_map={new_id: new_id},
                                    assert_all_vals_used=False)

        trial = trials.new_trial_docs([new_id], rval_specs, rval_results,
                                      rval_miscs)[0]
        trial['result'] = {'loss': reward, 'status': 'ok'}
        trial['state'] = hp.JOB_STATE_DONE
        trials.insert_trial_docs([trial])
        trials.refresh()

    def miscs_update_idxs_vals(self,
                               miscs,
                               idxs,
                               vals,
                               assert_all_vals_used=True,
                               idxs_map=None):
        """
        Unpack the idxs-vals format into the list of dictionaries that is
        `misc`.

        Parameters
        ----------
        idxs_map : dict
            idxs_map is a dictionary of id->id mappings so that the misc['idxs'] can
        contain different numbers than the idxs argument.
        """
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

    def get_suggestion(self, random_search=False):
        """get suggestion from hyperopt

        Parameters
        ----------
        random_search : bool
            flag to indicate random search or not (default: {False})

        Returns
        ----------
        total_params : dict
            parameter suggestion
        """

        rval = self.rval
        trials = rval.trials
        algorithm = rval.algo
        new_ids = rval.trials.new_trial_ids(1)
        rval.trials.refresh()
        random_state = rval.rstate.randint(2**31 - 1)
        if random_search:
            new_trials = hp.rand.suggest(new_ids, rval.domain, trials,
                                         random_state)
        else:
            new_trials = algorithm(new_ids, rval.domain, trials, random_state)
        rval.trials.refresh()
        vals = new_trials[0]['misc']['vals']
        parameter = dict()
        for key in vals:
            try:
                parameter[key] = vals[key][0].item()
            except (KeyError, IndexError):
                parameter[key] = None

        # remove '_index' from json2parameter and save params-id
        total_params = json2parameter(self.json, parameter)
        return total_params

    def import_data(self, data):
        """Import additional data for tuning

        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s" %
                        (_completed_num, len(data)))
            _completed_num += 1
            if self.algorithm_name == 'random_search':
                return
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info(
                    "Useless trial data, value is %s, skip this trial data." %
                    _value)
                continue
            self.supplement_data_num += 1
            _parameter_id = '_'.join(
                ["ImportData", str(self.supplement_data_num)])
            self.total_data[_parameter_id] = _add_index(in_x=self.json,
                                                        parameter=_params)
            self.receive_trial_result(parameter_id=_parameter_id,
                                      parameters=_params,
                                      value=_value)
        logger.info("Successfully import data to TPE/Anneal tuner.")
