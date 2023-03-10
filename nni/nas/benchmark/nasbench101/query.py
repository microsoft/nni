# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict

from nni.nas.benchmark.utils import load_benchmark
from .schema import Nb101TrialStats, Nb101TrialConfig, proxy
from .graph_util import hash_module, infer_num_vertices


def query_nb101_trial_stats(arch, num_epochs, isomorphism=True, reduction=None, include_intermediates=False):
    """
    Query trial stats of NAS-Bench-101 given conditions.

    Parameters
    ----------
    arch : dict or None
        If a dict, it is in the format that is described in
        :class:`nni.nas.benchmark.nasbench101.Nb101TrialConfig`. Only trial stats
        matched will be returned. If none, all architectures in the database will be matched.
    num_epochs : int or None
        If int, matching results will be returned. Otherwise a wildcard.
    isomorphism : boolean
        Whether to match essentially-same architecture, i.e., architecture with the
        same graph-invariant hash value.
    reduction : str or None
        If 'none' or None, all trial stats will be returned directly.
        If 'mean', fields in trial stats will be averaged given the same trial config.
    include_intermediates : boolean
        If true, intermediate results will be returned.

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nasbench101.Nb101TrialStats` objects,
        where each of them has been converted into a dict.
    """

    if proxy.obj is None:
        proxy.initialize(load_benchmark('nasbench101'))

    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in Nb101TrialStats._meta.sorted_field_names:
            if field_name not in ['id', 'config']:
                fields.append(fn.AVG(getattr(Nb101TrialStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(Nb101TrialStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = Nb101TrialStats.select(*fields, Nb101TrialConfig).join(Nb101TrialConfig)
    conditions = []
    if arch is not None:
        if isomorphism:
            num_vertices = infer_num_vertices(arch)
            conditions.append(Nb101TrialConfig.hash == hash_module(arch, num_vertices))
        else:
            conditions.append(Nb101TrialConfig.arch == arch)
    if num_epochs is not None:
        conditions.append(Nb101TrialConfig.num_epochs == num_epochs)
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(Nb101TrialStats.config)
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
