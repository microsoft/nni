# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict

from nni.nas.benchmark.utils import load_benchmark
from .schema import NdsTrialStats, NdsTrialConfig, proxy


def query_nds_trial_stats(model_family, proposer, generator, model_spec, cell_spec, dataset,
                          num_epochs=None, reduction=None, include_intermediates=False):
    """
    Query trial stats of NDS given conditions.

    Parameters
    ----------
    model_family : str or None
        If str, can be one of the model families available in :class:`nni.nas.benchmark.nds.NdsTrialConfig`.
        Otherwise a wildcard.
    proposer : str or None
        If str, can be one of the proposers available in :class:`nni.nas.benchmark.nds.NdsTrialConfig`. Otherwise a wildcard.
    generator : str or None
        If str, can be one of the generators available in :class:`nni.nas.benchmark.nds.NdsTrialConfig`. Otherwise a wildcard.
    model_spec : dict or None
        If specified, can be one of the model spec available in :class:`nni.nas.benchmark.nds.NdsTrialConfig`.
        Otherwise a wildcard.
    cell_spec : dict or None
        If specified, can be one of the cell spec available in :class:`nni.nas.benchmark.nds.NdsTrialConfig`.
        Otherwise a wildcard.
    dataset : str or None
        If str, can be one of the datasets available in :class:`nni.nas.benchmark.nds.NdsTrialConfig`. Otherwise a wildcard.
    num_epochs : float or None
        If int, matching results will be returned. Otherwise a wildcard.
    reduction : str or None
        If 'none' or None, all trial stats will be returned directly.
        If 'mean', fields in trial stats will be averaged given the same trial config.
    include_intermediates : boolean
        If true, intermediate results will be returned.

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nds.NdsTrialStats` objects,
        where each of them has been converted into a dict.
    """

    if proxy.obj is None:
        proxy.initialize(load_benchmark('nds'))

    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in NdsTrialStats._meta.sorted_field_names:
            if field_name not in ['id', 'config', 'seed']:
                fields.append(fn.AVG(getattr(NdsTrialStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(NdsTrialStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = NdsTrialStats.select(*fields, NdsTrialConfig).join(NdsTrialConfig)
    conditions = []
    for field_name in ['model_family', 'proposer', 'generator', 'model_spec', 'cell_spec',
                       'dataset', 'num_epochs']:
        if locals()[field_name] is not None:
            conditions.append(getattr(NdsTrialConfig, field_name) == locals()[field_name])
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(NdsTrialStats.config)
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
