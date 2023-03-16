# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict
from .schema import NlpTrialStats, NlpTrialConfig


def query_nlp_trial_stats(arch, dataset, reduction=None, include_intermediates=False):
    """
    Query trial stats of NLP benchmark given conditions, including config(arch + dataset) and training results after 50 epoch.

    Parameters
    ----------
    arch : dict or None
        If a dict, it is in the format that is described in
        :class:`nni.nas.benchmark.nlp.NlpTrialConfig`. Only trial stats matched will be returned.
        If none, all architectures in the database will be matched.
    dataset : str or None
        If specified, can be one of the dataset available in :class:`nni.nas.benchmark.nlp.NlpTrialConfig`.
        Otherwise a wildcard.
    reduction : str or None
        If 'none' or None, all trial stats will be returned directly.
        If 'mean', fields in trial stats will be averaged given the same trial config.
        Please note that some trial configs have multiple runs which make "reduction" meaningful, while some may not.
    include_intermediates : boolean
        If true, intermediate results will be returned.

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nlp.NlpTrialStats` objects,
        where each of them has been converted into a dict.
    """
    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in NlpTrialStats._meta.sorted_field_names:
            if field_name not in ['id', 'config']:
                fields.append(fn.AVG(getattr(NlpTrialStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(NlpTrialStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = NlpTrialStats.select(*fields, NlpTrialConfig).join(NlpTrialConfig)

    conditions = []
    if arch is not None:
        conditions.append(NlpTrialConfig.arch == arch)
    if dataset is not None:
        conditions.append(NlpTrialConfig.dataset == dataset)

    for trial in query.where(functools.reduce(lambda a, b: a & b, conditions)):
        if include_intermediates:
            data = model_to_dict(trial)
            # exclude 'trial' from intermediates as it is already available in data
            data['intermediates'] = [
                {k: v for k, v in model_to_dict(t).items() if k != 'trial'} for t in trial.intermediates
            ]
            yield data
        else:
            yield model_to_dict(trial)
