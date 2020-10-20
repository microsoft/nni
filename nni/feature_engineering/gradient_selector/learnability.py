# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

import time

import numpy as np
import scipy.special
import torch
import torch.nn as nn

from . import constants
from . import syssettings
from .fginitialize import ChunkDataLoader

torch.set_default_tensor_type(syssettings.torch.tensortype)
sparsetensor = syssettings.torch.sparse.tensortype


def def_train_opt(p):
    """
    Return the default optimizer.
    """
    return torch.optim.Adam(p, 1e-1, amsgrad=False)


def revcumsum(U):
    """
    Reverse cumulative sum for faster performance.
    """
    return U.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])


def triudr(X, r):

    Zr = torch.zeros_like(X, requires_grad=False)
    U = X * r
    Zr[:-1] = X[:-1] * revcumsum(U)[1:]

    return Zr


def triudl(X, l):

    Zl = torch.zeros_like(X, requires_grad=False)
    U = X * l
    Zl[1:] = X[1:] * (U.cumsum(dim=0)[:-1])

    return Zl


class ramp(torch.autograd.Function):
    """
    Ensures input is between 0 and 1
    """

    @staticmethod
    def forward(ctx, input_data):
        ctx.save_for_backward(input_data)
        return input_data.clamp(min=0, max=1)


    @staticmethod
    def backward(ctx, grad_output):
        input_data, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_data < 0] = 1e-2
        grad_input[input_data > 1] = -1e-2
        return grad_input


class safesqrt(torch.autograd.Function):
    """
    Square root without dividing by 0.
    """
    @staticmethod
    def forward(ctx, input_data):
        o = input_data.sqrt()
        ctx.save_for_backward(input_data, o)
        return o


    @staticmethod
    def backward(ctx, grad_output):
        _, o = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input *= 0.5 / (o + constants.EPSILON)
        return grad_input


class LearnabilityMB(nn.Module):
    """
    Calculates the learnability of a set of features.
    mini-batch version w/ "left" and "right" multiplies
    """


    def __init__(self, Nminibatch, D, coeff, groups=None, binary=False,
                 device=constants.Device.CPU):
        super(LearnabilityMB, self).__init__()

        a = coeff / scipy.special.binom(Nminibatch, np.arange(coeff.size) + 2)
        self.order = a.size
        # pylint: disable=E1102
        self.a = torch.tensor(a, dtype=torch.get_default_dtype(), requires_grad=False)
        self.binary = binary

        self.a = self.a.to(device)


    def ret_val(self, z):
        """
        Get the return value based on z.
        """

        if not self.binary:
            return 1 - z

        else:
            return 0.5 * (1 - safesqrt.apply(ramp.apply(z)))


    def forward(self, s, X, y):

        l = y.clone()
        r = y.clone()
        z = 0

        for i in range(self.order):
            if i % 2 == 0:
                Z = triudr(X, r)
                r = torch.mm(Z, s)
            else:
                Z = triudl(X, l)
                l = torch.mm(Z, s)
            if self.a[i] != 0:
                # same the computation if a[i] is 0
                p = torch.mm(l.t(), r)
                z += self.a[i] * p
        return self.ret_val(z)


class Solver(nn.Module):
    """
    Class that performs the main optimization.
    Keeps track of the current x and iterates through data to learn x given the penalty and order.
    """

    def __init__(self,
                 PreparedData,
                 order,
                 Nminibatch=None,
                 groups=None,
                 soft_groups=None,
                 x0=None,
                 C=1,
                 ftransform=torch.sigmoid,
                 get_train_opt=def_train_opt,
                 accum_steps=1,
                 rng=np.random.RandomState(0),
                 max_norm_clip=1.,
                 shuffle=True,
                 device=constants.Device.CPU,
                 verbose=1):
        """

        Parameters
        ----------
        PreparedData : Dataset of PrepareData class
        order : int
            What order of interactions to include. Higher orders
            may be more accurate but increase the run time. 12 is the maximum allowed order.
        Nminibatch : int
            Number of rows in a mini batch
        groups : array-like
            Optional, shape = [n_features]
            Groups of columns that must be selected as a unit
            e.g. [0, 0, 1, 2] specifies the first two columns are part of a group.
        soft_groups : array-like
            optional, shape = [n_features]
            Groups of columns come from the same source
            Used to encourage sparsity of number of sources selected
            e.g. [0, 0, 1, 2] specifies the first two columns are part of a group.
        x0 : torch.tensor
            Optional, initialization of x.
        C : float
            Penalty parameter.
        get_train_opt : function
            Function that returns a pytorch optimizer, Adam is the default
        accum_steps : int
            Number of steps
        rng : random state
        max_norm_clip : float
            Maximum allowable size of the gradient
        shuffle : bool
            Whether or not to shuffle data within the dataloader
        order : int
            What order of interactions to include. Higher orders
            may be more accurate but increase the run time. 12 is the maximum allowed order.
        penalty : int
            Constant that multiplies the regularization term.
        ftransform : function
            Function to transform the x. sigmoid is the default.
        device : str
            'cpu' to run on CPU and 'cuda' to run on GPU. Runs much faster on GPU
        verbose : int
            Controls the verbosity when fitting. Set to 0 for no printing
            1 or higher for printing every verbose number of gradient steps.
        """
        super(Solver, self).__init__()

        self.Ntrain, self.D = PreparedData.N, PreparedData.n_features
        if groups is not None:
            # pylint: disable=E1102
            groups = torch.tensor(groups, dtype=torch.long)
            self.groups = groups
        else:
            self.groups = None
        if soft_groups is not None:
            # pylint: disable=E1102
            soft_groups = torch.tensor(soft_groups, dtype=torch.long)
            self.soft_D = torch.unique(soft_groups).size()[0]
        else:
            self.soft_D = None
        self.soft_groups = soft_groups

        if Nminibatch is None:
            Nminibatch = self.Ntrain
        else:
            if Nminibatch > self.Ntrain:
                print('Minibatch larger than sample size.'
                      + (' Reducing from %d to %d.'
                         % (Nminibatch, self.Ntrain)))
                Nminibatch = self.Ntrain
        if Nminibatch > PreparedData.max_rows:
            print('Minibatch larger than mem-allowed.'
                  + (' Reducing from %d to %d.' % (Nminibatch,
                                                   PreparedData.max_rows)))
            Nminibatch = int(np.min([Nminibatch, PreparedData.max_rows]))
        self.Nminibatch = Nminibatch
        self.accum_steps = accum_steps

        if x0 is None:
            x0 = torch.zeros(self.D, 1, dtype=torch.get_default_dtype())
        self.ftransform = ftransform
        self.x = nn.Parameter(x0)
        self.max_norm = max_norm_clip

        self.device = device
        self.verbose = verbose

        self.multiclass = PreparedData.classification and PreparedData.n_classes and PreparedData.n_classes > 2
        if self.multiclass:
            self.n_classes = PreparedData.n_classes
        else:
            self.n_classes = None
        # whether to treat all classes equally
        self.balanced = PreparedData.balanced
        self.ordinal = PreparedData.ordinal

        if (hasattr(PreparedData, 'mappings')
                or PreparedData.storage_level == 'disk'):
            num_workers = PreparedData.num_workers
        elif PreparedData.storage_level == constants.StorageLevel.DENSE:
            num_workers = 0
        else:
            num_workers = 0

        if constants.Device.CUDA in device:
            pin_memory = False
        else:
            pin_memory = False

        if num_workers == 0:
            timeout = 0
        else:
            timeout = 60

        self.ds_train = ChunkDataLoader(
            PreparedData,
            batch_size=self.Nminibatch,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            timeout=timeout)
        self.f_train = LearnabilityMB(self.Nminibatch, self.D,
                                      constants.Coefficients.SLE[order],
                                      self.groups,
                                      binary=PreparedData.classification,
                                      device=self.device)
        self.opt_train = get_train_opt(torch.nn.ParameterList([self.x]))
        self.it = 0
        self.iters_per_epoch = int(np.ceil(len(self.ds_train.dataset)
                                           / self.ds_train.batch_size))
        self.f_train = self.f_train.to(device)
        # pylint: disable=E1102
        self.w = torch.tensor(
            C / (C + 1),
            dtype=torch.get_default_dtype(), requires_grad=False)
        self.w = self.w.to(device)


    def penalty(self, s):
        """
        Calculate L1 Penalty.
        """
        to_return = torch.sum(s) / self.D
        if self.soft_groups is not None:
            # if soft_groups, there is an additional penalty for using more
            # groups
            s_grouped = torch.zeros(self.soft_D, 1,
                                    dtype=torch.get_default_dtype(),
                                    device=self.device)
            for group in torch.unique(self.soft_groups):
                # groups should be indexed 0 to n_group - 1
                # TODO: consider other functions here
                s_grouped[group] = s[self.soft_groups == group].max()
            # each component of the penalty contributes .5
            # TODO: could make this a user given parameter
            to_return = (to_return + torch.sum(s_grouped) / self.soft_D) * .5
        return to_return


    def forward_and_backward(self, s, xsub, ysub, retain_graph=False):
        """
        Completes the forward operation and computes gradients for learnability and penalty.
        """
        f_train = self.f_train(s, xsub, ysub)
        pen = self.penalty(s).unsqueeze(0).unsqueeze(0)
        # pylint: disable=E1102
        grad_outputs = torch.tensor([[1]], dtype=torch.get_default_dtype(),
                                    device=self.device)
        g1, = torch.autograd.grad([f_train], [self.x], grad_outputs,
                                  retain_graph=True)
        # pylint: disable=E1102
        grad_outputs = torch.tensor([[1]], dtype=torch.get_default_dtype(),
                                    device=self.device)
        g2, = torch.autograd.grad([pen], [self.x], grad_outputs,
                                  retain_graph=retain_graph)
        return f_train, pen, g1, g2


    def combine_gradient(self, g1, g2):
        """
        Combine gradients from learnability and penalty

        Parameters
        ----------
        g1 : array-like
            gradient from learnability
        g2 : array-like
            gradient from penalty
        """
        to_return = ((1 - self.w) * g1 + self.w * g2) / self.accum_steps
        if self.groups is not None:
            # each column will get a gradient
            # but we can only up or down groups, so the gradient for the group
            # should be the average of the gradients of the columns
            to_return_grouped = torch.zeros_like(self.x)
            for group in torch.unique(self.groups):
                to_return_grouped[self.groups ==
                                  group] = to_return[self.groups == group].mean()
            to_return = to_return_grouped
        return to_return


    def combine_loss(self, f_train, pen):
        """
        Combine the learnability and L1 penalty.
        """
        return ((1 - self.w) * f_train.detach() + self.w * pen.detach()) \
            / self.accum_steps


    def transform_y_into_binary(self, ysub, target_class):
        """
        Transforms multiclass classification problems into a binary classification problem.
        """
        with torch.no_grad():
            ysub_binary = torch.zeros_like(ysub)
            if self.ordinal:
                # turn ordinal problems into n-1 classifications of is this
                # example less than rank k
                if target_class == 0:
                    return None

                ysub_binary[ysub >= target_class] = 1
                ysub_binary[ysub < target_class] = -1
            else:
                # turn multiclass problems into n binary classifications
                ysub_binary[ysub == target_class] = 1
                ysub_binary[ysub != target_class] = -1
        return ysub_binary


    def _get_scaling_value(self, ysub, target_class):
        """
        Returns the weight given to a class for multiclass classification.
        """
        if self.balanced:
            if self.ordinal:
                return 1 / (torch.unique(ysub).size()[0] - 1)

            return 1 / torch.unique(ysub).size()[0]
        else:
            if self.ordinal:
                this_class_proportion = torch.mean(ysub >= target_class)
                normalizing_constant = 0
                for i in range(1, self.n_classes):
                    normalizing_constant += torch.mean(ysub >= i)
                return this_class_proportion / normalizing_constant
            else:
                return torch.mean(ysub == target_class)


    def _skip_y_forward(self, y):
        """
        Returns boolean of whether to skip the currrent y if there is nothing to be learned from it.
        """
        if y is None:
            return True
        elif torch.unique(y).size()[0] < 2:
            return True
        else:
            return False


    def train(self, f_callback=None, f_stop=None):
        """
        Trains the estimator to determine which features to include.

        Parameters
        ----------
        f_callback : function
            Function that performs a callback
        f_stop: function
            Function that tells you when to stop
        """

        t = time.time()
        h = torch.zeros([1, 1], dtype=torch.get_default_dtype())
        h = h.to(self.device)
        # h_complete is so when we divide by the number of classes
        # we only do that for that minibatch if accumulating
        h_complete = h.clone()
        flag_stop = False
        dataloader_iterator = iter(self.ds_train)
        self.x.grad = torch.zeros_like(self.x)
        while not flag_stop:
            try:
                xsub, ysub = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.ds_train)
                xsub, ysub = next(dataloader_iterator)
            try:
                s = self.ftransform(self.x)
                s = s.to(self.device)
                if self.multiclass:
                    # accumulate gradients over each class, classes range from
                    # 0 to n_classes - 1
                    #num_classes_batch = torch.unique(ysub).size()[0]
                    for target_class in range(self.n_classes):
                        ysub_binary = self.transform_y_into_binary(
                            ysub, target_class)
                        if self._skip_y_forward(ysub_binary):
                            continue
                        # should should skip if target class is not included
                        # but that changes what we divide by
                        scaling_value = self._get_scaling_value(
                            ysub, target_class)
                        f_train, pen, g1, g2 = self.forward_and_backward(
                            s, xsub, ysub_binary, retain_graph=True)
                        self.x.grad += self.combine_gradient(
                            g1, g2) * scaling_value
                        h += self.combine_loss(f_train,
                                               pen) * scaling_value
                else:
                    if not self._skip_y_forward(ysub):
                        f_train, pen, g1, g2 = self.forward_and_backward(
                            s, xsub, ysub)
                        self.x.grad += self.combine_gradient(g1, g2)
                        h += self.combine_loss(f_train, pen)
                    else:
                        continue
                h_complete += h
                self.it += 1
                if torch.isnan(h):
                    raise constants.NanError(
                        'Loss is nan, something may be misconfigured')
                if self.it % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        torch.nn.ParameterList([self.x]),
                        max_norm=self.max_norm)
                    self.opt_train.step()

                    t = time.time() - t
                    if f_stop is not None:
                        flag_stop = f_stop(self, h, self.it, t)

                    if f_callback is not None:
                        f_callback(self, h, self.it, t)
                    elif self.verbose and (self.it // self.accum_steps) % self.verbose == 0:
                        epoch = int(self.it / self.iters_per_epoch)
                        print(
                            '[Minibatch: %6d/ Epoch: %3d/ t: %3.3f s] Loss: %0.3f' %
                            (self.it, epoch, t, h_complete / self.accum_steps))

                    if flag_stop:
                        break

                    self.opt_train.zero_grad()
                    h = 0
                    h_complete = 0
                    t = time.time()
            except KeyboardInterrupt:
                flag_stop = True
                break
