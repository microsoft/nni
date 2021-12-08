from typing import Any, List
from ..graph import Model

def _unpack_if_only_one(ele: List[Any]):
    if len(ele) == 1:
        return ele[0]
    return ele

def get_mutation_dict(model: Model):
    return {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model.history}

def mutation_dict_to_summary(mutation: dict) -> dict:
    mutation_summary = {}
    for label, samples in mutation.items():
        # FIXME: this check might be wrong
        if not isinstance(samples, list):
            mutation_summary[label] = samples
        else:
            for i, sample in enumerate(samples):
                mutation_summary[f'{label}_{i}'] = sample
    return mutation_summary

def get_mutation_summary(model: Model) -> dict:
    mutation = get_mutation_dict(model)
    return mutation_dict_to_summary(mutation)
