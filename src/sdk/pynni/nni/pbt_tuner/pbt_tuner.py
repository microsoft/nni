# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
pbt_tuner.py
"""

import copy
import logging
import numpy as np
import os

import nni
from nni.tuner import Tuner
from nni.utils import NodeType, OptimizeMode, extract_scalar_reward

import nni.parameter_expressions as parameter_expressions

logger = logging.getLogger('pbt_tuner_AutoML')

# to be revised
all_checkpoint_dir = os.getenv('NNI_CHECKPOINT_DIRECTORY')


def json2space(x, oldy=None, name=NodeType.ROOT):
    """
    Change search space from json format to hyperopt format

    """
    y = list()
    if isinstance(x, dict):
        if NodeType.TYPE in x.keys():
            _type = x[NodeType.TYPE]
            name = name + '-' + _type
            if _type == 'choice':
                if oldy is not None:
                    _index = oldy[NodeType.INDEX]
                    y += json2space(x[NodeType.VALUE][_index],
                                    oldy[NodeType.VALUE], name=name+'[%d]' % _index)
                else:
                    y += json2space(x[NodeType.VALUE], None, name=name)
            y.append(name)
        else:
            for key in x.keys():
                y += json2space(x[key], oldy[key] if oldy else None, name+"[%s]" % str(key))
    elif isinstance(x, list):
        for i, x_i in enumerate(x):
            if isinstance(x_i, dict):
                if NodeType.NAME not in x_i.keys():
                    raise RuntimeError('\'_name\' key is not found in this nested search space.')
            y += json2space(x_i, oldy[i] if oldy else None, name + "[%d]" % i)
    return y


def json2parameter(x, is_rand, random_state, oldy=None, Rand=False, name=NodeType.ROOT):
    """
    Json to paramaters.

    """
    if isinstance(x, dict):
        if NodeType.TYPE in x.keys():
            _type = x[NodeType.TYPE]
            _value = x[NodeType.VALUE]
            name = name + '-' + _type
            Rand |= is_rand[name]
            if Rand is True:
                if _type == 'choice':
                    _index = random_state.randint(len(_value))
                    y = {
                        NodeType.INDEX: _index,
                        NodeType.VALUE: json2parameter(
                            x[NodeType.VALUE][_index],
                            is_rand,
                            random_state,
                            None,
                            Rand,
                            name=name+"[%d]" % _index
                        )
                    }
                else:
                    y = getattr(parameter_expressions, _type)(*(_value + [random_state]))
            else:
                y = copy.deepcopy(oldy)
        else:
            y = dict()
            for key in x.keys():
                y[key] = json2parameter(
                    x[key],
                    is_rand,
                    random_state,
                    oldy[key] if oldy else None,
                    Rand,
                    name + "[%s]" % str(key)
                )
    elif isinstance(x, list):
        y = list()
        for i, x_i in enumerate(x):
            if isinstance(x_i, dict):
                if NodeType.NAME not in x_i.keys():
                    raise RuntimeError('\'_name\' key is not found in this nested search space.')
            y.append(json2parameter(
                x_i,
                is_rand,
                random_state,
                oldy[i] if oldy else None,
                Rand,
                name + "[%d]" % i
            ))
    else:
        y = copy.deepcopy(x)
    return y


def exploit_and_explore(bot_trial_info, top_trial_info, factors, epoch):
    """
    Replace checkpoint of bot_trial with top, and perturb hyperparameters

    """
    bot_checkpoint_dir = bot_trial_info.checkpoint_dir
    top_hyper_parameters = top_trial_info.hyper_parameters
    hyper_parameters = copy.deepcopy(top_hyper_parameters)
    for key in hyper_parameters.keys():
        if key=='load_checkpoint_dir':
            hyper_parameters[key] = hyper_parameters['save_checkpoint_dir']
        elif key=='save_checkpoint_dir':
            hyper_parameters[key] = os.path.join(bot_checkpoint_dir, str(epoch))
        elif NodeType.INDEX in hyper_parameters[key]:
            continue
        else:
            perturb = np.random.choice(factors)
            hyper_parameters[key] *= perturb
    bot_trial_info.hyper_parameters = hyper_parameters
    bot_trial_info.clean_id()


class Trial_Info:
    """
    Information of each trial, refresh for each epoch

    """
    def __init__(self, checkpoint_dir=None, hyper_parameters=None, parameter_id=None, score=None):
        self.checkpoint_dir = checkpoint_dir
        self.hyper_parameters = hyper_parameters
        self.parameter_id = parameter_id
        self.score = score


    def clear_id(self):
        self.parameter_id = None


class PBTTuner(Tuner):
    def __init__(self, optimize_mode="maximize", population_size=10, factors=(1.2, 0.8), fraction=0.2):
        """
        Initialization

        Parameters
        ----------
        optimize_mode: str
            maximize or minimize
        population_size: int
            number if trials for each epoch
        factors: tuple
            factors for perturbation
        fraction: float
            fraction for selecting bottom and top trials
        """
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.population_size = population_size
        self.factors = factors
        self.fraction = fraction
        #defined in trial code
        #self.perturbation_interval = perturbation_interval

        self.population = None
        self.pos = -1
        self.param_ids = []
        self.running = {}
        self.finished = []
        self.credit = 0
        self.finished_trials = 0
        self.epoch = 0

        self.searchspace_json = None
        self.space = None

        self.send_trial_callback = None

        logger.info('pbt tuner initialization')


    def update_search_space(self, search_space):
        """
        Get search space

        Parameters
        ----------
        search_space : dict
            Search space
        """
        logger.info('update search space %s', search_space)
        self.searchspace_json = search_space
        self.space = json2space(self.searchspace_json)

        self.random_state = np.random.RandomState()
        self.population = []
        is_rand = dict()

        for item in self.space:
            is_rand[item] = True

        for i in range(self.population_size):
            hyper_parameters = json2parameter(
                self.searchspace_json, is_rand, self.random_state)
            checkpoint_dir = os.path.join(all_checkpoint_dir, str(i))
            hyper_parameters['load_checkpoint_dir'] = os.path.join(checkpoint_dir, str(self.epoch))
            hyper_parameters['save_checkpoint_dir'] = os.path.join(checkpoint_dir, str(self.epoch))
            self.population.append(Trial_Info(checkpoint_dir=checkpoint_dir,
                                               hyper_parameters=hyper_parameters))


    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """
        Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.

        Parameters
        ----------
        parameter_id_list : list of int
            Unique identifiers for each set of requested hyper-parameters.
            These will later be used in :meth:`receive_trial_result`.
        **kwargs
            Not used

        Returns
        -------
        list
            A list of newly generated configurations
        """
        result = []
        self.send_trial_callback = kwargs['st_callback']
        for parameter_id in parameter_id_list:
            had_exception = False
            try:
                logger.debug("generating param for %s", parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                #add credit
                had_exception = True
            if not had_exception:
                result.append(res)
        return result


    def generate_parameters(self, parameter_id, **kwargs):
        """
        Generate parameters, if no trial configration for now, self.credit plus 1 to send the config later

        Parameters
        ----------
        parameter_id : int
            Unique identifier for requested hyper-parameters.
            This will later be used in :meth:`receive_trial_result`.
        **kwargs
            Not used

        Returns
        -------
        dict
            One newly generated configuration

        """
        if self.pos == self.population_size-1:
            logger.debug('Credit added by one in parameters request')
            self.credit += 1
            self.param_ids.append(parameter_id)
            raise nni.NoMoreTrialError('no more parameters now.')
        trial_info = self.population[self.pos]
        trial_info.parameter_id = parameter_id
        self.running[parameter_id] = trial_info
        self.pos += 1
        logger.info('generate parameter : %s', trial_info.hyper_parameters)
        return trial_info.hyper_parameters


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Receive trial's result. if the number of finished trials equals self.population_size, start the next epoch to
        train the model.

        Parameters
        ----------
        parameter_id : int
            Unique identifier of used hyper-parameters, same with :meth:`generate_parameters`.
        parameters : dict
            Hyper-parameters generated by :meth:`generate_parameters`.
        value : dict
            Result from trial (the return value of :func:`nni.report_final_result`).
        """
        logger.info('get one trial result')
        value = extract_scalar_reward(value)
        if self.optimize_mode == OptimizeMode.Minimize:
            value = -value
        trial_info = self.running.pop(parameter_id, None)
        trial_info.score = value
        self.finished.append(trial_info)
        self.finished_trials += 1
        if self.finished_trials == self.population_size:
            logger.info('next epoch')
            self.epoch += 1
            self.population = []
            self.running = {}
            #exploit and explore
            self.finished = sorted(self.finished, key=lambda x: x['score'], reverse=True)
            cutoff = int(np.ceil(self.fraction * len(self.finished)))
            tops = self.finished[:cutoff]
            bottoms = self.finished[self.finished_trials - cutoff:]
            for bottom in bottoms:
                top = np.random.choice(tops)
                exploit_and_explore(bottom, top, self.factors, self.epoch)
            for trial in self.finished:
                if trial not in bottoms:
                    trial.clean_id()
                    trial.hyper_parameters['load_checkpoint_dir'] = trial.hyper_parameters['save_checkpoint_dir']
                    trial.hyper_parameters['save_checkpoint_dir'] = os.path.join(trial.checkpoint_dir, str(self.epoch))
            self.finished_trials = 0
            for _ in range(self.credit):
                self.credit -= 1
                parameter_id = self.param_ids.pop()
                trial_info = self.finished.pop()
                trial_info.parameter_id = parameter_id
                self.population.append(trial_info)
                self.running[parameter_id] = trial_info
                self.send_trial_callback(parameter_id, trial_info.hyper_parameters)


    def import_data(self, data):
        pass