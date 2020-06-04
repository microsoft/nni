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

    For each round t, AutoCompressPruner prune the model for the same sparsity to achive the ovrall sparsity:
        1. Generate sparsities distribution using SimualtedAnnealingPruner
        2. Perform ADMM-based structured pruning to generate pruning result for the next round.
           Here we use 'speedup' to perform real pruning.

    For more details, please refer to the paper: https://arxiv.org/abs/1907.03141.
    """

    def __init__(self, model, config_list, trainer, evaluator, dummy_input,
                 optimize_iterations=3, optimize_mode='maximize', pruning_mode='channel',
                 # SimulatedAnnealing related
                 start_temperature=100, stop_temperature=20, cool_down_rate=0.9, perturbation_magnitude=0.35,
                 # ADMM related
                 admm_optimize_iterations=30, admm_training_epochs=5, row=1e-4,
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
            Function used for the first step of ADMM pruning. This function should take 
            a pytorch model, optimizer, criterion, epoch, callback as parameters and train the model, no return is required.
        evaluator : function
            Function to evaluate the masked model.
            This function should take a pytorch model as the only parameter and return a scalar reward (accuracy, -loss... etc.)
        dummy_input : pytorch tensor
            The dummy input for ```jit.trace```, users should put it on right device before pass in
        optimize_iterations : int
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
        admm_optimize_iterations : int
            ADMM optimize iterations
        admm_training_epochs : int
            training epochs of the first optimization subproblem of ADMMPruner
        row : float
            penalty parameters for ADMM training
        experiment_data_dir : string
            PATH to save experiment data
        """
        # original model
        self._model_to_prune = model
        self._pruning_mode = pruning_mode

        self._trainer = trainer
        self._evaluator = evaluator
        self._dummy_input = dummy_input
        self._optimize_iterations = optimize_iterations
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for SA algorithm
        self._start_temperature = start_temperature
        self._stop_temperature = stop_temperature
        self._cool_down_rate = cool_down_rate
        self._perturbation_magnitude = perturbation_magnitude

        # hyper parameters for ADMM algorithm
        self._admm_optimize_iterations = admm_optimize_iterations
        self._admm_training_epochs = admm_training_epochs
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

        sparsity_each_round = 1 - pow(1-self._sparsity, 1/self._optimize_iterations)

        for i in range(self._optimize_iterations):
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
                pruning_mode=self._pruning_mode,
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
                optimize_iterations=self._admm_optimize_iterations,
                training_epochs=self._admm_training_epochs,
                row=self._row,
                pruning_mode=self._pruning_mode)
            ADMMpruner.compress()

            ADMMpruner.export_model(os.path.join(self._experiment_data_dir, 'model_admm_masked.pth'), os.path.join(
                self._experiment_data_dir, 'mask.pth'))

            # use speed up to do pruning before next iteration, because SimulatedAnnealingPruner & ADMMPruner don't take maked models
            self._model_to_prune.load_state_dict(torch.load(os.path.join(
                self._experiment_data_dir, 'model_admm_masked.pth')))

            masks_file = os.path.join(self._experiment_data_dir, 'mask.pth')
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

            _logger.info('Speeding up models...')
            m_speedup = ModelSpeedup(
                self._model_to_prune, self._dummy_input, masks_file, device)
            m_speedup.speedup_model()

            evaluation_result = self._evaluator(self._model_to_prune)
            _logger.info('Evaluation result of iteration %d: %s',
                         i, evaluation_result)

        _logger.info('----------Compression finished--------------')

        os.remove(os.path.join(
            self._experiment_data_dir, 'model_admm_masked.pth'))
        os.remove(os.path.join(self._experiment_data_dir, 'mask.pth'))

        return self._model_to_prune

    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None, device=None):
        _logger.info("AutoCompressPruner export only model, not mask")

        torch.save(self._model_to_prune.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)

        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            if device is None:
                device = torch.device('cpu')
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self._model_to_prune,
                              input_data.to(device), onnx_path)
            _logger.info(
                'Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)
