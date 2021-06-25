import logging
from typing import Any, List, Optional

common_search_space_types = [
    'choice',
    'randint',
    'uniform',
    'quniform',
    'loguniform',
    'qloguniform',
    'normal',
    'qnormal',
    'lognormal',
    'qlognormal',
]

def validate_search_space(
        search_space: Any,
        support_types: Optional[List[str]] = None,
        raise_exception: bool = False  # for now, in case false positive
    ) -> bool:

    if not raise_exception:
        try:
            validate_search_space(search_space, support_types, True)
            return True
        except ValueError as e:
            logging.getLogger(__name__).error(e.args[0])
        return False

    if support_types is None:
        support_types = common_search_space_types

    if not isinstance(search_space, dict):
        raise ValueError('search space is not a dict')

    for name, spec in search_space.items():
        if '_type' not in spec or '_value' not in spec:
            raise ValueError(f'search space "{name}" does not have "_type" or "_value"')
        type_ = spec['_type']
        if type_ not in support_types:
            raise ValueError(f'search space "{name}" has unsupported type {type_}')
        args = spec['_value']
        if not isinstance(args, list):
            raise ValueError(f'search space "{name}"\'s value is not a list')

        if type_ == 'choice':
            continue

        if type_.startswith('q'):
            if len(args) != 3:
                raise ValueError(f'search space "{name}" ({type_}) must have 3 values')
        else:
            if len(args) != 2:
                raise ValueError(f'search space "{name}" ({type_}) must have 2 values')

        if type_ == 'randint':
            if not all(isinstance(arg, int) for arg in args):
                raise ValueError(f'search space "{name}" ({type_}) must have int values')
        else:
            if not all(isinstance(arg, (float, int)) for arg in args):
                raise ValueError(f'search space "{name}" ({type_}) must have float values')

        if 'normal' not in type_:
            if args[0] >= args[1]:
                raise ValueError(f'search space "{name}" ({type_}) must have high > low')
            if 'log' in type_ and args[0] <= 0:
                raise ValueError(f'search space "{name}" ({type_}) must have low > 0')
        else:
            if args[1] <= 0:
                raise ValueError(f'search space "{name}" ({type_}) must have sigma > 0')

    return True
