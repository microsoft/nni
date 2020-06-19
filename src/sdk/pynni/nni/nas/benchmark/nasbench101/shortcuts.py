import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict
from .model import Nb101ComputedStats, Nb101IntermediateStats, Nb101RunConfig
from .graph_util import hash_module, infer_num_vertices


def query_nb101_computed_stats(arch, num_epochs, isomorphism=True, reduction=None):
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
