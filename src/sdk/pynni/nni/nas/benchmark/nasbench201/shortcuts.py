import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict
from .model import Nb201ComputedStats, Nb201IntermediateStats, Nb201RunConfig


def query_nb201_computed_stats(arch, num_epochs, dataset, reduction=None):
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
