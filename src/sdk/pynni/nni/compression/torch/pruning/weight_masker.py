# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class WeightMasker(object):
    def __init__(self, model, pruner):
        self.model = model
        self.pruner = pruner

    def calc_mask(self, weight, bias=None, sparsity=1., **kwargs):
        """
        Calculate the mask of given layer.
        Parameters
        ----------
        weight : weight tensor
            module weights
        bias: bias tensor
            module bias
        sparsity: float
            pruning ratio,  preserved weight ratio is `1 - sparsity`
        kwargs: dict
            additional parameters passed from pruner
        Returns
        -------
        dict
            dictionary for storing masks
        """

        raise NotImplementedError('{} calc_mask is not implemented'.format(self.__class__.__name__))
