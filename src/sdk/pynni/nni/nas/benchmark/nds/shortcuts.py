import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict
from .model import NdsComputedStats, NdsIntermediateStats, NdsRunConfig


def query_nds_computed_stats(model_family, proposer, generator, model_spec, cell_spec, dataset,
                             base_lr=None, weight_decay=None, num_epochs=None, reduction=None):
    """
    Query computed stats of NDS given conditions.

    Parameters
    ----------
    model_family : str or None
    proposer : str or None
    generator : str or None
    model_spec : dict or None
    dataset : str or None
    base_lr : float or None
    weight_decay : float or None
    num_epochs : float or None
    reduction : str or None

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nds.NdsComputedStats` objects,
        where each of them has been converted into a dict.
    """
    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in NdsComputedStats._meta.sorted_field_names:
            if field_name not in ['id', 'config', 'seed']:
                fields.append(fn.AVG(getattr(NdsComputedStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(NdsComputedStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = NdsComputedStats.select(*fields, NdsRunConfig).join(NdsRunConfig)
    conditions = []
    for field_name in ['model_family', 'proposer', 'generator', 'model_spec', 'cell_spec',
                       'dataset', 'base_lr', 'weight_decay', 'num_epochs']:
        if locals()[field_name] is not None:
            conditions.append(getattr(NdsRunConfig, field_name) == locals()[field_name])
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(NdsComputedStats.config)
    for k in query:
        yield model_to_dict(k)
