# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from nni.nas.pytorch.base_mutator import BaseMutator
from nni.nas.pytorch.mutables import LayerChoice


class MixedOp(nn.Module):
    """
    This class is to instantiate and manage info of one LayerChoice.
    It includes architecture weights and member functions for the weights.
    """

    def __init__(self, mutable, latency):
        """
        Parameters
        ----------
        mutable : LayerChoice
            A LayerChoice in user model
        latency : List
            performance cost for each op in mutable
        """
        super(MixedOp, self).__init__()
        self.latency = latency
        n_choices = len(mutable)
        self.path_alpha = nn.Parameter(
            torch.FloatTensor([1.0 / n_choices for i in range(n_choices)])
        )
        self.path_alpha.requires_grad = False
        self.temperature = 1.0

    def get_path_alpha(self):
        """Return the architecture parameter."""
        return self.path_alpha

    def get_weighted_latency(self):
        """Return the weighted perf_cost of current mutable."""
        soft_masks = self.probs_over_ops()
        weighted_latency = sum(m * l for m, l in zip(soft_masks, self.latency))
        return weighted_latency

    def set_temperature(self, temperature):
        """
        Set the annealed temperature for gumbel softmax.

        Parameters
        ----------
        temperature : float
            The annealed temperature for gumbel softmax
        """
        self.temperature = temperature

    def to_requires_grad(self):
        """Enable gradient calculation."""
        self.path_alpha.requires_grad = True

    def to_disable_grad(self):
        """Disable gradient calculation."""
        self.path_alpha.requires_grad = False

    def probs_over_ops(self):
        """Apply gumbel softmax to generate probability distribution."""
        return F.gumbel_softmax(self.path_alpha, self.temperature)

    def forward(self, mutable, x):
        """
        Define forward of LayerChoice.

        Parameters
        ----------
        mutable : LayerChoice
            this layer's mutable
        x : tensor
            inputs of this layer, only support one input

        Returns
        -------
        output: tensor
            output of this layer
        """
        candidate_ops = list(mutable)
        soft_masks = self.probs_over_ops()
        output = sum(m * op(x) for m, op in zip(soft_masks, candidate_ops))

        return output

    @property
    def chosen_index(self):
        """
        choose the op with max prob

        Returns
        -------
        int
            index of the chosen one
        """
        alphas = self.path_alpha.data.detach().cpu().numpy()
        index = int(np.argmax(alphas))
        return index


class FBNetMutator(BaseMutator):
    """
    This mutator initializes and operates all the LayerChoices of the supernet.
    It is for the related trainer to control the training flow of LayerChoices,
    coordinating with whole training process.
    """

    def __init__(self, model, lookup_table):
        """
        Init a MixedOp instance for each mutable i.e., LayerChoice.
        And register the instantiated MixedOp in corresponding LayerChoice.
        If does not register it in LayerChoice, DataParallel does'nt work then,
        for architecture weights are not included in the DataParallel model.
        When MixedOPs are registered, we use ```requires_grad``` to control
        whether calculate gradients of architecture weights.

        Parameters
        ----------
        model : pytorch model
            The model that users want to tune,
            it includes search space defined with nni nas apis
        lookup_table : class
            lookup table object to manage model space information,
            including candidate ops for each stage as the model space,
            input channels/output channels/stride/fm_size as the layer config,
            and the performance information for perf_cost accumulation.

        """
        super(FBNetMutator, self).__init__(model)
        self.mutable_list = []

        # Collect the op names of the candidate ops within each mutable
        ops_names_mutable = dict()
        left = 0
        right = 1
        for stage_name in lookup_table.layer_num:
            right = lookup_table.layer_num[stage_name]
            stage_ops = lookup_table.lut_ops[stage_name]
            ops_names = [op_name for op_name in stage_ops]

            for i in range(left, left + right):
                ops_names_mutable[i] = ops_names
            left = right

        # Create the mixed op
        for i, mutable in enumerate(self.undedup_mutables):
            ops_names = ops_names_mutable[i]
            latency_mutable = lookup_table.lut_perf[i]
            latency = [latency_mutable[op_name] for op_name in ops_names]
            self.mutable_list.append(mutable)
            mutable.registered_module = MixedOp(mutable, latency)

    def on_forward_layer_choice(self, mutable, *args, **kwargs):
        """
        Callback of layer choice forward. This function defines the forward
        logic of the input mutable. So mutable is only interface, its real
        implementation is defined in mutator.

        Parameters
        ----------
        mutable: LayerChoice
            forward logic of this input mutable
        args: list of torch.Tensor
            inputs of this mutable
        kwargs: dict
            inputs of this mutable

        Returns
        -------
        torch.Tensor
            output of this mutable, i.e., LayerChoice
        int
            index of the chosen op
        """
        # FIXME: return mask, to be consistent with other algorithms
        idx = mutable.registered_module.chosen_index
        return mutable.registered_module(mutable, *args, **kwargs), idx

    def num_arch_params(self):
        """
        The number of mutables, i.e., LayerChoice

        Returns
        -------
        int
            the number of LayerChoice in user model
        """
        return len(self.mutable_list)

    def get_architecture_parameters(self):
        """
        Get all the architecture parameters.

        yield
        -----
        PyTorch Parameter
            Return path_alpha of the traversed mutable
        """
        for mutable in self.undedup_mutables:
            yield mutable.registered_module.get_path_alpha()

    def get_weighted_latency(self):
        """
        Get the latency weighted by gumbel softmax coefficients.

        yield
        -----
        Tuple
            Return the weighted_latency of the traversed mutable
        """
        for mutable in self.undedup_mutables:
            yield mutable.registered_module.get_weighted_latency()

    def set_temperature(self, temperature):
        """
        Set the annealed temperature of the op for gumbel softmax.

        Parameters
        ----------
        temperature : float
            The annealed temperature for gumbel softmax
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.set_temperature(temperature)

    def arch_requires_grad(self):
        """
        Make architecture weights require gradient
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.to_requires_grad()

    def arch_disable_grad(self):
        """
        Disable gradient of architecture weights, i.e., does not
        calculate gradient for them.
        """
        for mutable in self.undedup_mutables:
            mutable.registered_module.to_disable_grad()

    def sample_final(self):
        """
        Generate the final chosen architecture.

        Returns
        -------
        dict
            the choice of each mutable, i.e., LayerChoice
        """
        result = dict()
        for mutable in self.undedup_mutables:
            assert isinstance(mutable, LayerChoice)
            index = mutable.registered_module.chosen_index
            # pylint: disable=not-callable
            result[mutable.key] = (
                F.one_hot(torch.tensor(index), num_classes=len(mutable))
                .view(-1)
                .bool(),
            )
        return result
