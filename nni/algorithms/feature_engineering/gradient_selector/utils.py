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


import numpy as np

class EMA():
    """
    maintains an exponential moving average
    """

    def __init__(self, f=np.nan, discount_factor=0.1, valid_after=None,
                 n_iters_relchange=3):

        self.f_ma = [f]
        self.fs = [f]
        self.gamma = discount_factor
        self.rel_change = [np.nan]
        if valid_after is None:
            self.valid_after = int(1/discount_factor)
        else:
            self.valid_after = valid_after
        self.n_iters_relchange = n_iters_relchange
        self.initialized = False

    def reset(self, f):

        self.f_ma = [f]
        self.fs = [f]
        self.rel_change = [np.nan]
        self.initialized = True

    def relchange(self):

        if self.num_updates() > np.max([self.valid_after,
                                        self.n_iters_relchange]):
            return np.max(self.rel_change[-self.n_iters_relchange:])
        else:
            return np.nan

    def update(self, f_new):

        if not self.initialized:
            self.reset(f_new)
        else:
            self.fs.append(f_new)
            self.f_ma.append(self.f_ma[-1]*(1-self.gamma) + self.gamma*f_new)
            if self.num_updates() > self.valid_after:
                self.rel_change.append(np.abs((self.f_ma[-1]-self.f_ma[-2])
                                              / self.f_ma[-2]))

    def num_updates(self):

        return len(self.f_ma)

    def __call__(self):

        if self.num_updates() > self.valid_after:
            return self.f_ma[-1]
        else:
            return np.nan
