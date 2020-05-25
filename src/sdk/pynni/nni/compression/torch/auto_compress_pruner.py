# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import copy
from schema import And, Optional

from nni.utils import OptimizeMode

from .compressor import Pruner
from .simulated_annealing_pruner import SimulatedAnnealingPruner
from .admm_pruner import ADMMPruner
from .utils import CompressorSchema


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class AutoCompressPruner(Pruner):
    """
    This is a Pytorch implementation of AutoCompress pruning algorithm.

    For each round t:
        1. Generate sparsities distribution using SimualtedAnnealingPruner
        2. Perform ADMM-based structured pruning to generate pruning result, for the next round t

    Perform prurification step (the speedup process in nni)

    For more details, please refer to the paper: https://arxiv.org/abs/1907.03141.
    """

    def __init__(self, model, config_list, trainer, evaluator, iterations=3, optimize_mode='maximize', pruning_mode='channel',
                 # SimulatedAnnealing related
                 start_temperature=100, stop_temperature=20, cool_down_rate=0.9, perturbation_magnitude=0.35,
                 optimize_iterations=30, epochs=5, row=1e-4,  # ADMM related
                 experiment_data_dir='./'):
        """
        Parameters
        ----------
        model : pytorch model
            The model to be pruned
        config_list : list
            Supported keys:
                - sparsity : The final sparsity when the compression is done.
                - op_types : The operation type to prune.
        trainer : function
            function used for the first step of ADMM training
        evaluator : function
            function to evaluate the pruned model
        iterations : int
            number of overall iterations 
        optimize_mode : str
            optimize mode, 'maximize' or 'minimize', by default 'maximize'
        pruning_mode : str
            'channel' or 'fine_grained, by default 'channel'
        start_temperature : float
            Simualated Annealing related parameter
        stop_temperature : float
            Simualated Annealing related parameter
        cool_down_rate : float
            Simualated Annealing related parameter
        perturbation_magnitude : float
            initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature
        optimize_iteration : int
            ADMM optimize iterations
        epochs : int
            training epochs of the first optimization subproblem
        row : float
            penalty parameters for ADMM training
        experiment_data_dir : string
            PATH to save experiment data
        """
        # original model
        self._model_to_prune = copy.deepcopy(model)
        self._pruning_mode = pruning_mode

        super().__init__(model, config_list)

        self._trainer = trainer
        self._evaluator = evaluator
        self._iterations = iterations
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for SA algorithm
        self._start_temperature = start_temperature
        self._stop_temperature = stop_temperature
        self._cool_down_rate = cool_down_rate
        self._perturbation_magnitude = perturbation_magnitude

        # hyper parameters for ADMM algorithm
        self._optimize_iterations = optimize_iterations
        self._epochs = epochs
        self._row = row

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

    def _detect_modules_to_compress(self):
        """
        redefine this function, consider only the layers without dependencies
        """
        SimulatedAnnealingPruner._detect_modules_to_compress(self)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            Supported keys:
                - prune_iterations : The number of rounds for the iterative pruning.
                - sparsity : The final sparsity when the compression is done.
        """
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            Optional('op_types'): [str],
        }], model, _logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, **kwargs):
        return None

    def compress(self):
        """
        Compress the model with Simulated Annealing.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        _logger.info('Starting AutoCompress pruning...')

        sparsity_each_round = 1 - pow(1-self._sparsity, 1/self._iterations)

        for i in range(self._iterations):
            _logger.info('Pruning iteration: %d', i)
            _logger.info('Target sparsity this round: %s', 1-pow(1-sparsity_each_round, i+1))

            # TODO: check iterative pruning
            SApruner = SimulatedAnnealingPruner(
                model=copy.deepcopy(self._model_to_prune),
                config_list=[
                    {"sparsity": sparsity_each_round, "op_types": 'default'}],
                evaluator=self._evaluator,
                optimize_mode=self._optimize_mode,
                pruning_mode=self._pruning_mode,
                start_temperature=self._start_temperature,
                stop_temperature=self._stop_temperature,
                cool_down_rate=self._cool_down_rate,
                perturbation_magnitude=self._perturbation_magnitude,
                experiment_data_dir=self._experiment_data_dir)
            config_list = SApruner.compress(return_config_list=True)

            ADMMpruner = ADMMPruner(
                model=self._model_to_prune,
                config_list=config_list,
                trainer=self._trainer,
                optimize_iterations=self._optimize_iterations,
                epochs=self._epochs,
                row=self._row,
                pruning_mode=self._pruning_mode)
            ADMMpruner.compress()

            

        _logger.info('----------Compression finished--------------')

        return self.bound_model
