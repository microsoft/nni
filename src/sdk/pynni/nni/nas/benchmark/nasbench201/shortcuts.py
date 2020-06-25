import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict
from .model import Nb201ComputedStats, Nb201IntermediateStats, Nb201RunConfig


def query_nb201_computed_stats(arch, num_epochs, dataset, reduction=None):
    """
    Query computed stats of NAS-Bench-201 given conditions.

    Parameters
    ----------
    arch : dict or None
        If a dict, it is in the format that is described in
        :class:`nni.nas.benchmark.nasbench201.Nb201RunConfig`. Only computed stats
        matched will be returned. If none, architecture will be a wildcard.
    num_epochs : int or None
        If int, matching results will be returned. Otherwise a wildcard.
    dataset : str or None
        If specified, can be one of the dataset available in :class:`nni.nas.benchmark.nasbench201.Nb201RunConfig`.
        Otherwise a wildcard.
    reduction : str or None
        If 'none' or None, all computed stats will be returned directly.
        If 'mean', fields in computed stats will be averaged given the same run config.

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nasbench201.Nb201ComputedStats` objects,
        where each of them has been converted into a dict.
    """
    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in Nb201ComputedStats._meta.sorted_field_names:
            if field_name not in ['id', 'config', 'seed']:
                fields.append(fn.AVG(getattr(Nb201ComputedStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(Nb201ComputedStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = Nb201ComputedStats.select(*fields, Nb201RunConfig).join(Nb201RunConfig)
    conditions = []
    if arch is not None:
        conditions.append(Nb201RunConfig.arch == arch)
    if num_epochs is not None:
        conditions.append(Nb201RunConfig.num_epochs == num_epochs)
    if dataset is not None:
        conditions.append(Nb201RunConfig.dataset == dataset)
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(Nb201ComputedStats.config)
    for k in query:
        yield model_to_dict(k)
