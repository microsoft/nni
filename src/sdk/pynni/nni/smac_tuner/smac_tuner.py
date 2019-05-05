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
smac_tuner.py
"""

from nni.tuner import Tuner
from nni.utils import extract_scalar_reward

import sys
import logging
import numpy as np
import json_tricks
from enum import Enum, unique
from .convert_ss_to_scenario import generate_scenario

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.roar_facade import ROAR
from smac.facade.epils_facade import EPILS

@unique
class OptimizeMode(Enum):
    """Oprimize Mode class"""
    Minimize = 'minimize'
    Maximize = 'maximize'

class SMACTuner(Tuner):
    """
    Parameters
    ----------
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    """
    def __init__(self, optimize_mode):
        """Constructor"""
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.total_data = {}
        self.optimizer = None
        self.smbo_solver = None
        self.first_one = True
        self.update_ss_done = False
        self.loguniform_key = set()
        self.categorical_dict = {}

    def _main_cli(self):
        """Main function of SMAC for CLI interface
        
        Returns
        -------
        instance
            optimizer
        """
        self.logger.info("SMAC call: %s" % (" ".join(sys.argv)))

        cmd_reader = CMDReader()
        args, _ = cmd_reader.read_cmd()

        root_logger = logging.getLogger()
        root_logger.setLevel(args.verbose_level)
        logger_handler = logging.StreamHandler(
                stream=sys.stdout)
        if root_logger.level >= logging.INFO:
            formatter = logging.Formatter(
                "%(levelname)s:\t%(message)s")
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
        """TODO: this is urgly, we put all the initialization work in this method, because initialization relies
        on search space, also because update_search_space is called at the beginning.
        NOTE: updating search space is not supported.

        Parameters
        ----------
        search_space:
            search space
        """
        if not self.update_ss_done:
            self.categorical_dict = generate_scenario(search_space)
            if self.categorical_dict is None:
                raise RuntimeError('categorical dict is not correctly returned after parsing search space.')
            self.optimizer = self._main_cli()
            self.smbo_solver = self.optimizer.solver
            self.loguniform_key = {key for key in search_space.keys() if search_space[key]['_type'] == 'loguniform'}
            self.update_ss_done = True
        else:
            self.logger.warning('update search space is not supported.')

    def receive_trial_result(self, parameter_id, parameters, value):
        """receive_trial_result
       
        Parameters
        ----------
        parameter_id: int
            parameter id
        parameters:
            parameters
        value:
            value
        
        Raises
        ------
        RuntimeError
            Received parameter id not in total_data
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

    def convert_loguniform_categorical(self, challenger_dict):
        """Convert the values of type `loguniform` back to their initial range
        Also, we convert categorical:
        categorical values in search space are changed to list of numbers before,
        those original values will be changed back in this function
        
        Parameters
        ----------
        challenger_dict: dict
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

    def generate_parameters(self, parameter_id):
        """generate one instance of hyperparameters
        
        Parameters
        ----------
        parameter_id: int
            parameter id
        
        Returns
        -------
        list
            new generated parameters
        """
        if self.first_one:
            init_challenger = self.smbo_solver.nni_smac_start()
            self.total_data[parameter_id] = init_challenger
            return self.convert_loguniform_categorical(init_challenger.get_dictionary())
        else:
            challengers = self.smbo_solver.nni_smac_request_challengers()
            for challenger in challengers:
                self.total_data[parameter_id] = challenger
                return self.convert_loguniform_categorical(challenger.get_dictionary())

    def generate_multiple_parameters(self, parameter_id_list):
        """generate mutiple instances of hyperparameters
        
        Parameters
        ----------
        parameter_id_list: list
            list of parameter id
        
        Returns
        -------
        list
            list of new generated parameters
        """
        if self.first_one:
            params = []
            for one_id in parameter_id_list:
                init_challenger = self.smbo_solver.nni_smac_start()
                self.total_data[one_id] = init_challenger
                params.append(self.convert_loguniform_categorical(init_challenger.get_dictionary()))
        else:
            challengers = self.smbo_solver.nni_smac_request_challengers()
            cnt = 0
            params = []
            for challenger in challengers:
                if cnt >= len(parameter_id_list):
                    break
                self.total_data[parameter_id_list[cnt]] = challenger
                params.append(self.convert_loguniform_categorical(challenger.get_dictionary()))
                cnt += 1
        return params

    def import_data(self, data):
        pass
