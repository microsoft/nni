import copy
import logging
import torch
from .compressor import Pruner

_logger = logging.getLogger(__name__)


class LotteryTicketPruner(Pruner):
    """
    This is a Pytorch implementation of the paper "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks",
    following NNI model compression interface.

    Detail description...

    See Also
    --------
    :class:
    """
    def __init__(self, config_list):
        """
        config_list: supported keys:
            - sparsity
        """
        super().__init__(config_list)
        self.prune_iterations = 1
        self.curr_prune_iterations = 0
        self.epoch_per_iteration = None
        self.update_flags = {}
        self.mask_list = {}
        self.init_weights = {}

    def calc_mask(self, weight, config, op_name, **kwargs):
        """
        Generate mask for the given ``weight``.

        Parameters
        ----------
        weight: tensor
            The weight to be pruned
        config: dict
            Pruning configurations for this weight
        op_name: str
            The name of this operation???
        kwargs: dict
            ...

        Returns
        -------
        mask: tensor
            The mask for this weight
        """
        self.prune_iterations = config.get('prune_iterations')
        self.epoch_per_iteration = config.get('epoch_per_iteration')

        # keep the initial weights
        if self.init_weights.get(op_name) == None:
            self.init_weights.update({op_name: copy.deepcopy(weight)})

        if self.update_flags.get(op_name, True):
            if self.curr_prune_iterations == 0:
                mask = torch.ones(weight.shape).type_as(weight)
            else:
                sparsity = config.get('sparsity')
                sparsity_once = sparsity ** (1/self.prune_iterations)

                assert self.mask_list.get(op_name)
                curr_mask = self.mask_list.get(op_name)
                sorted_weights = np.sort(torch.abs(weight[curr_mask == 1]))
                index = np.around(sparsity * sorted_weights.size).astype(int)
                threshold = sorted_weights[index]

                w_abs = weight.abs() * curr_mask
                mask = torch.gt(w_abs, threshold).type_as(weight)

                # reinitiate the weights to the initial weights
                assert op_name in self.init_weights
                init_weight = self.init_weights.get(op_name)
                weight.data.copy_(init_weight.data)

            self.mask_list.update({op_name: mask})
            self.update_flags.update({op_name: False})
        else:
            mask = self.mask_list[op_name]
        return mask

    def update_epoch(self, epoch):
        """
        Control the pruning procedure on updated epoch number.

        Parameters
        ----------
        epoch: num
            The current epoch number provided by user call
        """
        if self.epoch_per_iteration \
            and int(epoch) % self.epoch_per_iteration == 0 \
            and self.curr_prune_iterations <= self.prune_iterations:
            for k in self.update_flags.keys():
                self.update_flags[k] = True
        self.curr_prune_iterations = int(epoch) // self.epoch_per_iteration

    def export_compressed_model(self):
        """
        Export two files, one is masks and the other is final weights.
        """