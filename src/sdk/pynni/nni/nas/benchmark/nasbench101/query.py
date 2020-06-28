import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict
from .model import Nb101ComputedStats, Nb101IntermediateStats, Nb101RunConfig
from .graph_util import hash_module, infer_num_vertices


def query_nb101_computed_stats(arch, num_epochs, isomorphism=True, reduction=None):
    """
    Query computed stats of NAS-Bench-101 given conditions.

    Parameters
    ----------
    arch : dict or None
        If a dict, it is in the format that is described in
        :class:`nni.nas.benchmark.nasbench101.Nb101RunConfig`. Only computed stats
        matched will be returned. If none, architecture will be a wildcard.
    num_epochs : int or None
        If int, matching results will be returned. Otherwise a wildcard.
    isomorphism : boolean
        Whether to match essentially-same architecture, i.e., architecture with the
        same graph-invariant hash value.
    reduction : str or None
        If 'none' or None, all computed stats will be returned directly.
        If 'mean', fields in computed stats will be averaged given the same run config.

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nasbench101.Nb101ComputedStats` objects,
        where each of them has been converted into a dict.
    """
    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in Nb101ComputedStats._meta.sorted_field_names:
            if field_name not in ['id', 'config']:
                fields.append(fn.AVG(getattr(Nb101ComputedStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(Nb101ComputedStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = Nb101ComputedStats.select(*fields, Nb101RunConfig).join(Nb101RunConfig)
    conditions = []
    if arch is not None:
        if isomorphism:
            num_vertices = infer_num_vertices(arch)
            conditions.append(Nb101RunConfig.hash == hash_module(arch, num_vertices))
        else:
            conditions.append(Nb101RunConfig.arch == arch)
    if num_epochs is not None:
        conditions.append(Nb101RunConfig.num_epochs == num_epochs)
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(Nb101ComputedStats.config)
    for k in query:
        yield model_to_dict(k)
