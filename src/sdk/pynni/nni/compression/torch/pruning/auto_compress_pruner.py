# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import copy
import torch
from schema import And, Optional

from nni.utils import OptimizeMode
from nni.compression.torch import ModelSpeedup

from ..compressor import Pruner
from ..utils.config_validation import CompressorSchema
from .simulated_annealing_pruner import SimulatedAnnealingPruner
from .admm_pruner import ADMMPruner


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class AutoCompressPruner(Pruner):
    """
    This is a Pytorch implementation of AutoCompress pruning algorithm.

    For each round, AutoCompressPruner prune the model for the same sparsity to achive the ovrall sparsity:
        1. Generate sparsities distribution using SimualtedAnnealingPruner
        2. Perform ADMM-based structured pruning to generate pruning result for the next round.
           Here we use 'speedup' to perform real pruning.

    For more details, please refer to the paper: https://arxiv.org/abs/1907.03141.
    """

    def __init__(self, model, config_list, trainer, evaluator, dummy_input,
                 num_iterations=3, optimize_mode='maximize', base_algo='l1',
                 # SimulatedAnnealing related
                 start_temperature=100, stop_temperature=20, cool_down_rate=0.9, perturbation_magnitude=0.35,
                 # ADMM related
                 admm_num_iterations=30, admm_training_epochs=5, row=1e-4,
                 experiment_data_dir='./'):
        """
        Parameters
        ----------
        model : pytorch model
            The model to be pruned
        config_list : list
            Supported keys:
                - sparsity : The target overall sparsity.
                - op_types : The operation type to prune.
        trainer : function
            function used for the first optimization subproblem.
            This function should include `model, optimizer, criterion, epoch, callback` as parameters,
            where callback should be inserted after loss.backward of the normal training process.
        evaluator : function
            function to evaluate the pruned model.
            This function should include `model` as the only parameter, and returns a scalar value.
        dummy_input : pytorch tensor
            The dummy input for ```jit.trace```, users should put it on right device before pass in
        num_iterations : int
            Number of overall iterations
        optimize_mode : str
            optimize mode, `maximize` or `minimize`, by default `maximize`
        base_algo : str
            base pruning algorithm. `level`, `l1` or `l2`, by default `l1`
        start_temperature : float
            Simualated Annealing related parameter
        stop_temperature : float
            Simualated Annealing related parameter
        cool_down_rate : float
            Simualated Annealing related parameter
        perturbation_magnitude : float
            Initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature
        admm_num_iterations : int
            Number of iterations of ADMM Pruner
        admm_training_epochs : int
            Training epochs of the first optimization subproblem of ADMMPruner
        row : float
            Penalty parameters for ADMM training
        experiment_data_dir : string
            PATH to store temporary experiment data
        """
        # original model
        self._model_to_prune = model
        self._base_algo = base_algo

        self._trainer = trainer
        self._evaluator = evaluator
        self._dummy_input = dummy_input
        self._num_iterations = num_iterations
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for SA algorithm
        self._start_temperature = start_temperature
        self._stop_temperature = stop_temperature
        self._cool_down_rate = cool_down_rate
        self._perturbation_magnitude = perturbation_magnitude

        # hyper parameters for ADMM algorithm
        self._admm_num_iterations = admm_num_iterations
        self._admm_training_epochs = admm_training_epochs
        self._row = row

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        """

        if self._base_algo == 'level':
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                Optional('op_types'): [str],
                Optional('op_names'): [str],
            }], model, _logger)
        elif self._base_algo in ['l1', 'l2']:
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                'op_types': ['Conv2d'],
                Optional('op_names'): [str]
            }], model, _logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, **kwargs):
        return None

    def compress(self):
        """
        Compress the model with AutoCompress.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        _logger.info('Starting AutoCompress pruning...')

        sparsity_each_round = 1 - pow(1-self._sparsity, 1/self._num_iterations)

        for i in range(self._num_iterations):
            _logger.info('Pruning iteration: %d', i)
            _logger.info('Target sparsity this round: %s',
                         1-pow(1-sparsity_each_round, i+1))

            # SimulatedAnnealingPruner
            _logger.info(
                'Generating sparsities with SimulatedAnnealingPruner...')
            SApruner = SimulatedAnnealingPruner(
                model=copy.deepcopy(self._model_to_prune),
                config_list=[
                    {"sparsity": sparsity_each_round, "op_types": ['Conv2d']}],
                evaluator=self._evaluator,
                optimize_mode=self._optimize_mode,
                base_algo=self._base_algo,
                start_temperature=self._start_temperature,
                stop_temperature=self._stop_temperature,
                cool_down_rate=self._cool_down_rate,
                perturbation_magnitude=self._perturbation_magnitude,
                experiment_data_dir=self._experiment_data_dir)
            config_list = SApruner.compress(return_config_list=True)
            _logger.info("Generated config_list : %s", config_list)

            # ADMMPruner
            _logger.info('Performing structured pruning with ADMMPruner...')
            ADMMpruner = ADMMPruner(
                model=copy.deepcopy(self._model_to_prune),
                config_list=config_list,
                trainer=self._trainer,
                num_iterations=self._admm_num_iterations,
                training_epochs=self._admm_training_epochs,
                row=self._row,
                base_algo=self._base_algo)
            ADMMpruner.compress()

            ADMMpruner.export_model(os.path.join(self._experiment_data_dir, 'model_admm_masked.pth'), os.path.join(
                self._experiment_data_dir, 'mask.pth'))

            # use speed up to prune the model before next iteration, because SimulatedAnnealingPruner & ADMMPruner don't take masked models
            self._model_to_prune.load_state_dict(torch.load(os.path.join(
                self._experiment_data_dir, 'model_admm_masked.pth')))

            masks_file = os.path.join(self._experiment_data_dir, 'mask.pth')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            _logger.info('Speeding up models...')
            m_speedup = ModelSpeedup(self._model_to_prune, self._dummy_input, masks_file, device)
            m_speedup.speedup_model()

            evaluation_result = self._evaluator(self._model_to_prune)
            _logger.info('Evaluation result of the pruned model in iteration %d: %s', i, evaluation_result)

        _logger.info('----------Compression finished--------------')

        os.remove(os.path.join(self._experiment_data_dir, 'model_admm_masked.pth'))
        os.remove(os.path.join(self._experiment_data_dir, 'mask.pth'))

        return self._model_to_prune

    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None, device=None):
        _logger.info("AutoCompressPruner export directly the pruned model without mask")

        torch.save(self._model_to_prune.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)

        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            if device is None:
                device = torch.device('cpu')
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self._model_to_prune, input_data.to(device), onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)
