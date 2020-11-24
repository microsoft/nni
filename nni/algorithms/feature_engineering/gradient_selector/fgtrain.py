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
import torch
from sklearn.feature_selection import SelectKBest, \
    f_classif, mutual_info_classif, f_regression, mutual_info_regression

from . import constants
from . import syssettings
from .learnability import Solver
from .utils import EMA

torch.set_default_tensor_type(syssettings.torch.tensortype)


def get_optim_f_stop(maxiter, maxtime, dftol_stop, freltol_stop,
                     minibatch=True):
    """
    Check stopping conditions.
    """

    discount_factor = 1. / 3

    total_t = [0.]
    df_store = [np.nan]
    it_store = [0]
    relchange_store = [np.nan]
    f_ma = EMA(discount_factor=discount_factor)
    df_ma = EMA(discount_factor=discount_factor)

    def f_stop(f0, v0, it, t):

        flag_stop = False

        total_t[-1] += t
        g = f0.x.grad.clone().cpu().detach()
        df = g.abs().max().numpy().squeeze()
        v = v0.clone().cpu().detach()
        f = v.numpy().squeeze()

        if it >= maxiter:
            flag_stop = True

        elif total_t[-1] >= maxtime:
            flag_stop = True

        f_ma.update(f)
        df_ma.update(df)
        rel_change = f_ma.relchange()

        if ((not minibatch) and (df < dftol_stop)) \
           or (minibatch and (df_ma() < dftol_stop)):
            flag_stop = True

        if rel_change < freltol_stop:
            flag_stop = True

        if not minibatch:
            df_store[-1] = df
        else:
            df_store[-1] = df_ma()
        relchange_store[-1] = rel_change
        it_store[-1] = it

        return flag_stop

    return f_stop, {'t': total_t, 'it': it_store, 'df': df_store,
                    'relchange': relchange_store}


def get_init(data_train, init_type='on', rng=np.random.RandomState(0), prev_score=None):
    """
    Initialize the 'x' variable with different settings
    """

    D = data_train.n_features
    value_off = constants.Initialization.VALUE_DICT[
        constants.Initialization.OFF]
    value_on = constants.Initialization.VALUE_DICT[
        constants.Initialization.ON]

    if prev_score is not None:
        x0 = prev_score
    elif not isinstance(init_type, str):
        x0 = value_off * np.ones(D)
        x0[init_type] = value_on
    elif init_type.startswith(constants.Initialization.RANDOM):
        d = int(init_type.replace(constants.Initialization.RANDOM, ''))
        x0 = value_off * np.ones(D)
        x0[rng.permutation(D)[:d]] = value_on
    elif init_type == constants.Initialization.SKLEARN:
        B = data_train.return_raw
        X, y = data_train.get_dense_data()
        data_train.set_return_raw(B)
        ix = train_sk_dense(init_type, X, y, data_train.classification)
        x0 = value_off * np.ones(D)
        x0[ix] = value_on
    elif init_type in constants.Initialization.VALUE_DICT:
        x0 = constants.Initialization.VALUE_DICT[init_type] * np.ones(D)
    else:
        raise NotImplementedError(
            'init_type {0} not supported yet'.format(init_type))
    # pylint: disable=E1102
    return torch.tensor(x0.reshape((-1, 1)),
                        dtype=torch.get_default_dtype())


def get_checkpoint(S, stop_conds, rng=None, get_state=True):
    """
    Save the necessary information into a dictionary
    """

    m = {}
    m['ninitfeats'] = S.ninitfeats
    m['x0'] = S.x0
    x = S.x.clone().cpu().detach()
    m['feats'] = np.where(x.numpy() >= 0)[0]
    m.update({k: v[0] for k, v in stop_conds.items()})
    if get_state:
        m.update({constants.Checkpoint.MODEL: S.state_dict(),
                  constants.Checkpoint.OPT: S.opt_train.state_dict(),
                  constants.Checkpoint.RNG: torch.get_rng_state(),
                  })
    if rng:
        m.update({'rng_state': rng.get_state()})

    return m


def _train(data_train, Nminibatch, order, C, rng, lr_train, debug, maxiter,
           maxtime, init, dftol_stop, freltol_stop, dn_log, accum_steps,
           path_save, shuffle, device=constants.Device.CPU,
           verbose=1,
           prev_checkpoint=None,
           groups=None,
           soft_groups=None):
    """
    Main training loop.
    """

    t_init = time.time()

    x0 = get_init(data_train, init, rng)
    if isinstance(init, str) and init == constants.Initialization.ZERO:
        ninitfeats = -1
    else:
        ninitfeats = np.where(x0.detach().numpy() > 0)[0].size

    S = Solver(data_train, order,
               Nminibatch=Nminibatch, x0=x0, C=C,
               ftransform=lambda x: torch.sigmoid(2 * x),
               get_train_opt=lambda p: torch.optim.Adam(p, lr_train),
               rng=rng,
               accum_steps=accum_steps,
               shuffle=shuffle,
               groups=groups,
               soft_groups=soft_groups,
               device=device,
               verbose=verbose)
    S = S.to(device)

    S.ninitfeats = ninitfeats
    S.x0 = x0

    if prev_checkpoint:
        S.load_state_dict(prev_checkpoint[constants.Checkpoint.MODEL])
        S.opt_train.load_state_dict(prev_checkpoint[constants.Checkpoint.OPT])
        torch.set_rng_state(prev_checkpoint[constants.Checkpoint.RNG])

    minibatch = S.Ntrain != S.Nminibatch

    f_stop, stop_conds = get_optim_f_stop(maxiter, maxtime, dftol_stop,
                                          freltol_stop, minibatch=minibatch)

    if debug:
        pass
    else:
        f_callback = None
    stop_conds['t'][-1] = time.time() - t_init

    S.train(f_stop=f_stop, f_callback=f_callback)

    return get_checkpoint(S, stop_conds, rng), S


def train_sk_dense(ty, X, y, classification):
    if classification:
        if ty.startswith('skf'):
            d = int(ty.replace('skf', ''))
            f_sk = f_classif
        elif ty.startswith('skmi'):
            d = int(ty.replace('skmi', ''))
            f_sk = mutual_info_classif
    else:
        if ty.startswith('skf'):
            d = int(ty.replace('skf', ''))
            f_sk = f_regression
        elif ty.startswith('skmi'):
            d = int(ty.replace('skmi', ''))
            f_sk = mutual_info_regression
    t = time.time()
    clf = SelectKBest(f_sk, k=d)
    clf.fit_transform(X, y.squeeze())
    ix = np.argsort(-clf.scores_)
    ix = ix[np.where(np.invert(np.isnan(clf.scores_[ix])))[0]][:d]
    t = time.time() - t
    return {'feats': ix, 't': t}
