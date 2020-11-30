import functools
import json

from playhouse.shortcuts import model_to_dict

from .model import NlpTrialStats, NlpTrialConfig, NlpIntermediateStats

def query_nlp_trial_stats(arch, dataset, include_intermediates=False):
    fields = []
    fields.append(NlpTrialStats)
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

