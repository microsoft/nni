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
