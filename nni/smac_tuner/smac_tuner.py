# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
smac_tuner.py
"""

import logging
import sys

import numpy as np
from schema import Schema, Optional

from smac.facade.epils_facade import EPILS
from smac.facade.roar_facade import ROAR
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.utils.io.cmd_reader import CMDReader

from ConfigSpaceNNI import Configuration

import nni
from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward

from .convert_ss_to_scenario import generate_scenario

logger = logging.getLogger('smac_AutoML')

class SMACClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('config_dedup'): bool
        }).validate(kwargs)

class SMACTuner(Tuner):
    """
    This is a wrapper of [SMAC](https://github.com/automl/SMAC3) following NNI tuner interface.
    It only supports ``SMAC`` mode, and does not support the multiple instances of SMAC3 (i.e.,
    the same configuration is run multiple times).
    """
    def __init__(self, optimize_mode="maximize", config_dedup=False):
        """
        Parameters
        ----------
        optimize_mode : str
            Optimize mode, 'maximize' or 'minimize', by default 'maximize'
        config_dedup : bool
            If True, the tuner will not generate a configuration that has been already generated.
            If False, a configuration may be generated twice, but it is rare for relatively large search space.
        """
        self.logger = logger
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.total_data = {}
        self.optimizer = None
        self.smbo_solver = None
        self.first_one = True
        self.update_ss_done = False
        self.loguniform_key = set()
        self.categorical_dict = {}
        self.cs = None
        self.dedup = config_dedup

    def _main_cli(self):
        """
        Main function of SMAC for CLI interface. Some initializations of the wrapped SMAC are done
        in this function.

        Returns
        -------
        obj
            The object of the SMAC optimizer
        """
        self.logger.info("SMAC call: %s", " ".join(sys.argv))

        cmd_reader = CMDReader()
        args, _ = cmd_reader.read_cmd()

        root_logger = logging.getLogger()
        root_logger.setLevel(args.verbose_level)
        logger_handler = logging.StreamHandler(stream=sys.stdout)
        if root_logger.level >= logging.INFO:
            formatter = logging.Formatter("%(levelname)s:\t%(message)s")
        else:
            formatter = logging.Formatter(
                "%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                "%Y-%m-%d %H:%M:%S")
        logger_handler.setFormatter(formatter)
        root_logger.addHandler(logger_handler)
        # remove default handler
        root_logger.removeHandler(root_logger.handlers[0])

        # Create defaults
        rh = None
        initial_configs = None
        stats = None
        incumbent = None

        # Create scenario-object
        scen = Scenario(args.scenario_file, [])
        self.cs = scen.cs

        if args.mode == "SMAC":
            optimizer = SMAC(
                scenario=scen,
                rng=np.random.RandomState(args.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                stats=stats,
                restore_incumbent=incumbent,
                run_id=args.seed)
        elif args.mode == "ROAR":
            optimizer = ROAR(
                scenario=scen,
                rng=np.random.RandomState(args.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                run_id=args.seed)
        elif args.mode == "EPILS":
            optimizer = EPILS(
                scenario=scen,
                rng=np.random.RandomState(args.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                run_id=args.seed)
        else:
            optimizer = None

        return optimizer

    def update_search_space(self, search_space):
        """
        Convert search_space to the format that ``SMAC3`` could recognize, thus, not all the search space types
        are supported. In this function, we also do the initialization of `SMAC3`, i.e., calling ``self._main_cli``.

        NOTE: updating search space during experiment running is not supported.

        Parameters
        ----------
        search_space : dict
            The format could be referred to search space spec (https://nni.readthedocs.io/en/latest/Tutorial/SearchSpaceSpec.html).
        """
        self.logger.info('update search space in SMAC.')
        if not self.update_ss_done:
            self.categorical_dict = generate_scenario(search_space)
            if self.categorical_dict is None:
                raise RuntimeError('categorical dict is not correctly returned after parsing search space.')
            # TODO: this is ugly, we put all the initialization work in this method, because initialization relies
            #         on search space, also because update_search_space is called at the beginning.
            self.optimizer = self._main_cli()
            self.smbo_solver = self.optimizer.solver
            self.loguniform_key = {key for key in search_space.keys() if search_space[key]['_type'] == 'loguniform'}
            self.update_ss_done = True
        else:
            self.logger.warning('update search space is not supported.')

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Receive a trial's final performance result reported through :func:``nni.report_final_result`` by the trial.
        GridSearchTuner does not need trial's results.

        Parameters
        ----------
        parameter_id : int
            Unique identifier of used hyper-parameters, same with :meth:`generate_parameters`.
        parameters : dict
            Hyper-parameters generated by :meth:`generate_parameters`.
        value : dict
            Result from trial (the return value of :func:`nni.report_final_result`).

        Raises
        ------
        RuntimeError
            Received parameter id not in ``self.total_data``
        """
        reward = extract_scalar_reward(value)
        if self.optimize_mode is OptimizeMode.Maximize:
            reward = -reward

        if parameter_id not in self.total_data:
            raise RuntimeError('Received parameter_id not in total_data.')
        if self.first_one:
            self.smbo_solver.nni_smac_receive_first_run(self.total_data[parameter_id], reward)
            self.first_one = False
        else:
            self.smbo_solver.nni_smac_receive_runs(self.total_data[parameter_id], reward)

    def param_postprocess(self, challenger_dict):
        """
        Postprocessing for a set of hyperparameters includes:
            1. Convert the values of type ``loguniform`` back to their initial range.
            2. Convert ``categorical``: categorical values in search space are changed to list of numbers before,
               those original values will be changed back in this function.

        Parameters
        ----------
        challenger_dict : dict
            challenger dict

        Returns
        -------
        dict
            dict which stores copy of challengers
        """
        converted_dict = {}
        for key, value in challenger_dict.items():
            # convert to loguniform
            if key in self.loguniform_key:
                converted_dict[key] = np.exp(challenger_dict[key])
            # convert categorical back to original value
            elif key in self.categorical_dict:
                idx = challenger_dict[key]
                converted_dict[key] = self.categorical_dict[key][idx]
            else:
                converted_dict[key] = value
        return converted_dict

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Generate one instance of hyperparameters (i.e., one configuration).
        Get one from SMAC3's ``challengers``.

        Parameters
        ----------
        parameter_id : int
            Unique identifier for requested hyper-parameters. This will later be used in :meth:`receive_trial_result`.
        **kwargs
            Not used

        Returns
        -------
        dict
            One newly generated configuration
        """
        if self.first_one:
            init_challenger = self.smbo_solver.nni_smac_start()
            self.total_data[parameter_id] = init_challenger
            return self.param_postprocess(init_challenger.get_dictionary())
        else:
            challengers = self.smbo_solver.nni_smac_request_challengers()
            challengers_empty = True
            for challenger in challengers:
                challengers_empty = False
                if self.dedup:
                    match = [v for k, v in self.total_data.items() \
                             if v.get_dictionary() == challenger.get_dictionary()]
                    if match:
                        continue
                self.total_data[parameter_id] = challenger
                return self.param_postprocess(challenger.get_dictionary())
            assert challengers_empty is False, 'The case that challengers is empty is not handled.'
            self.logger.info('In generate_parameters: No more new parameters.')
            raise nni.NoMoreTrialError('No more new parameters.')

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """
        Generate mutiple instances of hyperparameters. If it is a first request,
        retrieve the instances from initial challengers. While if it is not, request
        new challengers and retrieve instances from the requested challengers.

        Parameters
        ----------
        parameter_id_list: list of int
            Unique identifiers for each set of requested hyper-parameters.
            These will later be used in :meth:`receive_trial_result`.
        **kwargs
            Not used

        Returns
        -------
        list
            a list of newly generated configurations
        """
        if self.first_one:
            params = []
            for one_id in parameter_id_list:
                init_challenger = self.smbo_solver.nni_smac_start()
                self.total_data[one_id] = init_challenger
                params.append(self.param_postprocess(init_challenger.get_dictionary()))
        else:
            challengers = self.smbo_solver.nni_smac_request_challengers()
            cnt = 0
            params = []
            for challenger in challengers:
                if cnt >= len(parameter_id_list):
                    break
                if self.dedup:
                    match = [v for k, v in self.total_data.items() \
                             if v.get_dictionary() == challenger.get_dictionary()]
                    if match:
                        continue
                self.total_data[parameter_id_list[cnt]] = challenger
                params.append(self.param_postprocess(challenger.get_dictionary()))
                cnt += 1
            if self.dedup and not params:
                self.logger.info('In generate_multiple_parameters: No more new parameters.')
        return params

    def import_data(self, data):
        """
        Import additional data for tuning.

        Parameters
        ----------
        data : list of dict
            Each of which has at least two keys, ``parameter`` and ``value``.
        """
        _completed_num = 0
        for trial_info in data:
            self.logger.info("Importing data, current processing progress %s / %s", _completed_num, len(data))
            # simply validate data format
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                self.logger.info("Useless trial data, value is %s, skip this trial data.", _value)
                continue
            _value = extract_scalar_reward(_value)
            # convert the keys in loguniform and categorical types
            valid_entry = True
            for key, value in _params.items():
                if key in self.loguniform_key:
                    _params[key] = np.log(value)
                elif key in self.categorical_dict:
                    if value in self.categorical_dict[key]:
                        _params[key] = self.categorical_dict[key].index(value)
                    else:
                        self.logger.info("The value %s of key %s is not in search space.", str(value), key)
                        valid_entry = False
                        break
            if not valid_entry:
                continue
            # start import this data entry
            _completed_num += 1
            config = Configuration(self.cs, values=_params)
            if self.optimize_mode is OptimizeMode.Maximize:
                _value = -_value
            if self.first_one:
                self.smbo_solver.nni_smac_receive_first_run(config, _value)
                self.first_one = False
            else:
                self.smbo_solver.nni_smac_receive_runs(config, _value)
        self.logger.info("Successfully import data to smac tuner, total data: %d, imported data: %d.", len(data), _completed_num)
