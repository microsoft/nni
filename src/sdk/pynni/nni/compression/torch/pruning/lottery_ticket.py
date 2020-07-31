# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import torch
from schema import And, Optional
from ..utils.config_validation import CompressorSchema
from ..compressor import Pruner
from .finegrained_pruning import LevelPrunerMasker

logger = logging.getLogger('torch pruner')

class LotteryTicketPruner(Pruner):
    """
    Parameters
    ----------
    model : pytorch model
        The model to be pruned
    config_list : list
        Supported keys:
            - prune_iterations : The number of rounds for the iterative pruning.
            - sparsity : The final sparsity when the compression is done.
    optimizer : pytorch optimizer
        The optimizer for the model
    lr_scheduler : pytorch lr scheduler
        The lr scheduler for the model if used
    reset_weights : bool
        Whether reset weights and optimizer at the beginning of each round.
    """
    def __init__(self, model, config_list, optimizer=None, lr_scheduler=None, reset_weights=True):
        # save init weights and optimizer
        self.reset_weights = reset_weights
        if self.reset_weights:
            self._model = model
            self._optimizer = optimizer
            self._model_state = copy.deepcopy(model.state_dict())
            self._optimizer_state = copy.deepcopy(optimizer.state_dict())
            self._lr_scheduler = lr_scheduler
            if lr_scheduler is not None:
                self._scheduler_state = copy.deepcopy(lr_scheduler.state_dict())

        super().__init__(model, config_list, optimizer)
        self.curr_prune_iteration = None
        self.prune_iterations = config_list[0]['prune_iterations']
        self.masker = LevelPrunerMasker(model, self)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            Supported keys:
                - prune_iterations : The number of rounds for the iterative pruning.
                - sparsity : The final sparsity when the compression is done.
        """
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            'prune_iterations': And(int, lambda n: n > 0),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)
        assert len(set([x['prune_iterations'] for x in config_list])) == 1, 'The values of prune_iterations must be equal in your config'

    def _calc_sparsity(self, sparsity):
        keep_ratio_once = (1 - sparsity) ** (1 / self.prune_iterations)
        curr_keep_ratio = keep_ratio_once ** self.curr_prune_iteration
        return max(1 - curr_keep_ratio, 0)

    def _calc_mask(self, wrapper, sparsity):
        weight = wrapper.module.weight.data
        if self.curr_prune_iteration == 0:
            mask = {'weight_mask': torch.ones(weight.shape).type_as(weight)}
        else:
            curr_sparsity = self._calc_sparsity(sparsity)
            mask = self.masker.calc_mask(sparsity=curr_sparsity, wrapper=wrapper)
        return mask

    def calc_mask(self, wrapper, **kwargs):
        """
        Generate mask for the given ``weight``.

        Parameters
        ----------
        wrapper : Module
            The layer to be pruned

        Returns
        -------
        tensor
            The mask for this weight, it is ```None``` because this pruner
            calculates and assigns masks in ```prune_iteration_start```,
            no need to do anything in this function.
        """
        return None

    def get_prune_iterations(self):
        """
        Return the range for iterations.
        In the first prune iteration, masks are all one, thus, add one more iteration

        Returns
        -------
        list
            A list for pruning iterations
        """
        return range(self.prune_iterations + 1)

    def prune_iteration_start(self):
        """
        Control the pruning procedure on updated epoch number.
        Should be called at the beginning of the epoch.
        """
        if self.curr_prune_iteration is None:
            self.curr_prune_iteration = 0
        else:
            self.curr_prune_iteration += 1
        assert self.curr_prune_iteration < self.prune_iterations + 1, 'Exceed the configured prune_iterations'

        modules_wrapper = self.get_modules_wrapper()
        modules_to_compress = self.get_modules_to_compress()
        for layer, config in modules_to_compress:
            module_wrapper = None
            for wrapper in modules_wrapper:
                if wrapper.name == layer.name:
                    module_wrapper = wrapper
                    break
            assert module_wrapper is not None

            sparsity = config.get('sparsity')
            mask = self._calc_mask(module_wrapper, sparsity)
            # TODO: directly use weight_mask is not good
            module_wrapper.weight_mask = mask['weight_mask']
            # there is no mask for bias

        # reinit weights back to original after new masks are generated
        if self.reset_weights:
            # should use this member function to reset model weights
            self.load_model_state_dict(self._model_state)
            self._optimizer.load_state_dict(self._optimizer_state)
            if self._lr_scheduler is not None:
                self._lr_scheduler.load_state_dict(self._scheduler_state)
