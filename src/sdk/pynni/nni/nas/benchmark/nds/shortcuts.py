import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict
from .model import NdsComputedStats, NdsIntermediateStats, NdsRunConfig


def query_nds_computed_stats(model_family, proposer, generator, model_spec, cell_spec, dataset,
                             num_epochs=None, reduction=None):
    """
    Query computed stats of NDS given conditions.

    Parameters
    ----------
    model_family : str or None
        If str, can be one of the model families available in :class:`nni.nas.benchmark.nds.NdsRunConfig`.
        Otherwise a wildcard.
    proposer : str or None
        If str, can be one of the proposers available in :class:`nni.nas.benchmark.nds.NdsRunConfig`. Otherwise a wildcard.
    generator : str or None
        If str, can be one of the generators available in :class:`nni.nas.benchmark.nds.NdsRunConfig`. Otherwise a wildcard.
    model_spec : dict or None
        If specified, can be one of the model spec available in :class:`nni.nas.benchmark.nds.NdsRunConfig`.
        Otherwise a wildcard.
    cell_spec : dict or None
        If specified, can be one of the cell spec available in :class:`nni.nas.benchmark.nds.NdsRunConfig`.
        Otherwise a wildcard.
    dataset : str or None
        If str, can be one of the datasets available in :class:`nni.nas.benchmark.nds.NdsRunConfig`. Otherwise a wildcard.
    num_epochs : float or None
        If int, matching results will be returned. Otherwise a wildcard.
    reduction : str or None
        If 'none' or None, all computed stats will be returned directly.
        If 'mean', fields in computed stats will be averaged given the same run config.

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
                       'dataset', 'num_epochs']:
        if locals()[field_name] is not None:
            conditions.append(getattr(NdsRunConfig, field_name) == locals()[field_name])
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(NdsComputedStats.config)
    for k in query:
        yield model_to_dict(k)
