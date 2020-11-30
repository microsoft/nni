import functools
import json

from peewee import fn
from playhouse.shortcuts import model_to_dict
from .model import NlpTrialStats, NlpTrialConfig, NlpIntermediateStats

def query_nlp_trial_stats(arch, dataset, reduction=None, include_intermediates=False):
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

    # print("begin query")
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
    # print("end query")    

