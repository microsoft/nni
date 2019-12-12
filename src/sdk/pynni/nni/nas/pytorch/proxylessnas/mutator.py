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

import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from nni.nas.pytorch.base_mutator import BaseMutator



class ProxylessNasMutator(BaseMutator):
    def __init__(self, model):
        """
        Init a MixedOp instance for each named mutable i.e., LayerChoice

        Parameters
        ----------
        model : pytorch model
            The model that users want to tune, it includes search space defined with nni nas apis
        """
        super(ProxylessNasMutator, self).__init__(model)
        #self.mixed_ops = {}
        self.cnt = 0
        for _, mutable, _ in self.named_mutables(distinct=False):
            #mo = MixedOp(mutable)
            #mutable.register_module(mo)
            #self.mixed_ops[mutable.key] = mo
            self.cnt += 1

    def on_forward_layer_choice(self, mutable, *inputs):
        """
        Callback of layer choice forward. Override if you are an advanced user.
        On default, this method calls :meth:`on_calc_layer_choice_mask` to get a mask on how to choose between layers
        (either by switch or by weights), then it will reduce the list of all tensor outputs with the policy speicified
        in `mutable.reduction`. It will also cache the mask with corresponding `mutable.key`.

        Parameters
        ----------
        mutable: LayerChoice
        inputs: list of torch.Tensor

        Returns
        -------
        torch.Tensor
        index of the chosen op
        """
        # FIXME: return mask, to be consistent with other algorithms
        #idx = self.mixed_ops[mutable.key].active_op_index
        #return self.mixed_ops[mutable.key](mutable, *inputs), idx
        return mutable.registered_module.active_op_index

    def reset_binary_gates(self):
        """
        For each LayerChoice, binarize based on alpha to only activate one op
        """
        for _, mutable, _ in self.named_mutables(distinct=False):
            #k = mutable.key
            #self.mixed_ops[k].binarize(mutable)
            mutable.registered_module.binarize(mutable)

    def set_chosen_op_active(self):
        """
        For each LayerChoice, set the op with highest alpha as the chosen op
        Usually used for validation.
        """
        #for k in self.mixed_ops.keys():
        #    self.mixed_ops[k].set_chosen_op_active()
        for _, mutable, _ in self.named_mutables(distinct=False):
            mutable.registered_module.set_chosen_op_active()

    def num_arch_params(self):
        """
        Returns
        -------
        The number of LayerChoice in user model
        """
        #return len(self.mixed_ops)
        return self.cnt

    def set_arch_param_grad(self):
        """
        For each LayerChoice, calculate gradients for architecture weights, i.e., alpha
        """
        for _, mutable, _ in self.named_mutables(distinct=False):
            #k = mutable.key
            #self.mixed_ops[k].set_arch_param_grad(mutable)
            mutable.registered_module.set_arch_param_grad(len(mutable))

    def get_architecture_parameters(self):
        """
        Return architecture weights of each LayerChoice, for arch optimizer
        """
        #for k in self.mixed_ops.keys():
        #    yield self.mixed_ops[k].get_AP_path_alpha()
        for _, mutable, _ in self.named_mutables(distinct=False):
            yield mutable.registered_module.get_AP_path_alpha()
