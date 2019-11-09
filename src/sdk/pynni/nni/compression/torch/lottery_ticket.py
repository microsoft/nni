import copy
import logging
import torch
import numpy as np
from .compressor import Pruner

_logger = logging.getLogger(__name__)


class LotteryTicketPruner(Pruner):
    """
    This is a Pytorch implementation of the paper "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks",
    following NNI model compression interface.

    1. Randomly initialize a neural network f(x;theta_0) (where theta_0 follows D_{theta}).
    2. Train the network for j iterations, arriving at parameters theta_j.
    3. Prune p% of the parameters in theta_j, creating a mask m.
    4. Reset the remaining parameters to their values in theta_0, creating the winning ticket f(x;m*theta_0).
    5. Repeat step 2, 3, and 4.
    """
    def __init__(self, model, config_list, optimizer, reset_weights=True):
        """
        Parameters
        ----------
        model : pytorch model
            The model to be pruned
        config_list : list
            Supported keys:
                - prune_iterations : The number of rounds for the iterative pruning.
                - epoch_per_iteration : The number of epochs for each round.
                - sparsity : The final sparsity when the compression is done.
        optimizer : pytorch optimizer
            The optimizer for the model
        reset_weights : bool
            Whether reset weights and optimizer at the beginning of each round.
        """
        super().__init__(model, config_list)
        self.prune_iterations = 1
        self.curr_prune_iterations = 0
        self.epoch_per_iteration = None

        # save init weights and optimizer
        self.reset_weights = reset_weights
        if self.reset_weights:
            self._model = model
            self._optimizer = optimizer
            self._model_state = copy.deepcopy(model.state_dict())
            self._optimizer_state = copy.deepcopy(optimizer.state_dict())

    def _print_masks(self, print_mask=False):
        torch.set_printoptions(threshold=1000)
        for op_name in self.mask_dict.keys():
            mask = self.mask_dict[op_name]
            print('op name: ', op_name)
            if print_mask:
                print('mask: ', mask)
            # calculate current sparsity
            mask_num = mask.sum().item()
            mask_size = mask.numel()
            print('sparsity: ', 1 - mask_num / mask_size)
        torch.set_printoptions(profile='default')

    def _calc_sparsity(self, sparsity):
        keep_ratio_once = (1 - sparsity) ** (1 / self.prune_iterations)
        curr_keep_ratio = keep_ratio_once ** self.curr_prune_iterations
        return max(1 - curr_keep_ratio, 0)

    def _calc_mask(self, weight, sparsity, op_name):
        if self.curr_prune_iterations == 0:
            mask = torch.ones(weight.shape).type_as(weight)
        else:
            curr_sparsity = self._calc_sparsity(sparsity)
            assert self.mask_dict.get(op_name) is not None
            curr_mask = self.mask_dict.get(op_name)
            w_abs = weight.abs() * curr_mask
            sorted_weights = np.sort(w_abs, None)
            index = np.around(curr_sparsity * sorted_weights.size).astype(int)
            index = min(index, sorted_weights.size - 1)
            threshold = sorted_weights[index]
            mask = torch.gt(w_abs, threshold).type_as(weight)
        return mask

    def calc_mask(self, layer, config, **kwargs):
        """
        Generate mask for the given ``weight``.

        Parameters
        ----------
        layer : LayerInfo
            The layer to be pruned
        config: dict
            Pruning configurations for this weight
        **kwargs
            Not used

        Returns
        -------
        tensor
            The mask for this weight
        """
        weight = layer.module.weight.data
        op_name = layer.name
        self.prune_iterations = config.get('prune_iterations')
        self.epoch_per_iteration = config.get('epoch_per_iteration')

        if self.mask_dict.get(op_name) is None:
            mask = torch.ones(weight.shape).type_as(weight)
            self.mask_dict.update({op_name: mask})
        else:
            mask = self.mask_dict[op_name]
        return mask

    def update_epoch(self, epoch):
        """
        Control the pruning procedure on updated epoch number.
        Should be called at the beginning of the epoch.

        Parameters
        ----------
        epoch: num
            The current epoch number provided by user call
        """
        if self.epoch_per_iteration is not None:
            self.curr_prune_iterations = int(epoch) // self.epoch_per_iteration

        if self.epoch_per_iteration \
            and int(epoch) % self.epoch_per_iteration == 0 \
            and self.curr_prune_iterations <= self.prune_iterations:
            modules_to_compress = self.detect_modules_to_compress()
            for layer, config in modules_to_compress:
                sparsity = config.get('sparsity')
                mask = self._calc_mask(layer.module.weight.data, sparsity, layer.name)
                self.mask_dict.update({layer.name: mask})
            self._print_masks()
            # reinit weights back to original after new masks are generated
            if self.reset_weights:
                self._model.load_state_dict(self._model_state)
                self._optimizer.load_state_dict(self._optimizer_state)
