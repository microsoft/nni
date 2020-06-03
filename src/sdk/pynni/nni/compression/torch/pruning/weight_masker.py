# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class WeightMasker(object):
    def __init__(self, model, pruner, **kwargs):
        self.model = model
        self.pruner = pruner

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        """
        Calculate the mask of given layer.
        Parameters
        ----------
        sparsity: float
            pruning ratio,  preserved weight ratio is `1 - sparsity`
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict
            dictionary for storing masks
        """

        raise NotImplementedError('{} calc_mask is not implemented'.format(self.__class__.__name__))
