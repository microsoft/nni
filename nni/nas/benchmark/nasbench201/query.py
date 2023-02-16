# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict

from nni.nas.benchmark.utils import load_benchmark
from .schema import Nb201TrialStats, Nb201TrialConfig, proxy


def query_nb201_trial_stats(arch, num_epochs, dataset, reduction=None, include_intermediates=False):
    """
    Query trial stats of NAS-Bench-201 given conditions.

    Parameters
    ----------
    arch : dict or None
        If a dict, it is in the format that is described in
        :class:`nni.nas.benchmark.nasbench201.Nb201TrialConfig`. Only trial stats
        matched will be returned. If none, all architectures in the database will be matched.
    num_epochs : int or None
        If int, matching results will be returned. Otherwise a wildcard.
    dataset : str or None
        If specified, can be one of the dataset available in :class:`nni.nas.benchmark.nasbench201.Nb201TrialConfig`.
        Otherwise a wildcard.
    reduction : str or None
        If 'none' or None, all trial stats will be returned directly.
        If 'mean', fields in trial stats will be averaged given the same trial config.
    include_intermediates : boolean
        If true, intermediate results will be returned.

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nasbench201.Nb201TrialStats` objects,
        where each of them has been converted into a dict.
    """

    if proxy.obj is None:
        proxy.initialize(load_benchmark('nasbench201'))

    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in Nb201TrialStats._meta.sorted_field_names:
            if field_name not in ['id', 'config', 'seed']:
                fields.append(fn.AVG(getattr(Nb201TrialStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(Nb201TrialStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = Nb201TrialStats.select(*fields, Nb201TrialConfig).join(Nb201TrialConfig)
    conditions = []
    if arch is not None:
        conditions.append(Nb201TrialConfig.arch == arch)
    if num_epochs is not None:
        conditions.append(Nb201TrialConfig.num_epochs == num_epochs)
    if dataset is not None:
        conditions.append(Nb201TrialConfig.dataset == dataset)
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(Nb201TrialStats.config)
    for trial in query:
        if include_intermediates:
            data = model_to_dict(trial)
            # exclude 'trial' from intermediates as it is already available in data
            data['intermediates'] = [
                {k: v for k, v in model_to_dict(t).items() if k != 'trial'} for t in trial.intermediates
            ]
            yield data
        else:
            yield model_to_dict(trial)
