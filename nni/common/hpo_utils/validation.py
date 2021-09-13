# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
        raise ValueError(f'search space is a {type(search_space).__name__}, expect a dict : {repr(search_space)}')

    for name, spec in search_space.items():
        if not isinstance(spec, dict):
            raise ValueError(f'search space "{name}" is a {type(spec).__name__}, expect a dict : {repr(spec)}')
        if '_type' not in spec or '_value' not in spec:
            raise ValueError(f'search space "{name}" does not have "_type" or "_value" : {spec}')
        type_ = spec['_type']
        if type_ not in support_types:
            raise ValueError(f'search space "{name}" has unsupported type "{type_}" : {spec}')
        args = spec['_value']
        if not isinstance(args, list):
            raise ValueError(f'search space "{name}"\'s value is not a list : {spec}')

        if type_ == 'choice':
            if not all(isinstance(arg, (float, int, str)) for arg in args):
                # FIXME: need further check for each algorithm which types are actually supported
                # for now validation only prints warning so it doesn't harm
                if not isinstance(args[0], dict) or '_name' not in args[0]:  # not nested search space
                    raise ValueError(f'search space "{name}" (choice) should only contain numbers or strings : {spec}')
            continue

        if type_.startswith('q'):
            if len(args) != 3:
                raise ValueError(f'search space "{name}" ({type_}) must have 3 values : {spec}')
        else:
            if len(args) != 2:
                raise ValueError(f'search space "{name}" ({type_}) must have 2 values : {spec}')

        if type_ == 'randint':
            if not all(isinstance(arg, int) for arg in args):
                raise ValueError(f'search space "{name}" ({type_}) must have int values : {spec}')
        else:
            if not all(isinstance(arg, (float, int)) for arg in args):
                raise ValueError(f'search space "{name}" ({type_}) must have float values : {spec}')

        if 'normal' not in type_:
            if args[0] >= args[1]:
                raise ValueError(f'search space "{name}" ({type_}) must have high > low : {spec}')
            if 'log' in type_ and args[0] <= 0:
                raise ValueError(f'search space "{name}" ({type_}) must have low > 0 : {spec}')
        else:
            if args[1] <= 0:
                raise ValueError(f'search space "{name}" ({type_}) must have sigma > 0 : {spec}')

    return True
