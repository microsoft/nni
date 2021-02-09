import collections
from typing import Dict, Any, List
from ..graph import Model
from ..mutator import Mutator


def dry_run_for_search_space(model: Model, mutators: List[Mutator]) -> Dict[Any, List[Any]]:
    search_space = collections.OrderedDict()
    for mutator in mutators:
        recorded_candidates, model = mutator.dry_run(model)
        for i, candidates in recorded_candidates:
            search_space[(id(mutator), i)] = candidates
    return search_space
