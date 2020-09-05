#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import copy
import logging
import math

logger = logging.getLogger(__name__)


def py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def get_divisible_by(num, divisible_by, min_val=None):
    ret = int(num)
    if min_val is None:
        min_val = divisible_by
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((py2_round(num / divisible_by) or 1) * divisible_by)
        if ret < 0.9 * num:
            ret += divisible_by
    return ret


def filter_kwargs(func, kwargs, log_skipped=True):
    """ Filter kwargs based on signature of `func`
        Return arguments that matches `func`
    """
    import inspect

    sig = inspect.signature(func)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]

    if log_skipped:
        skipped_args = [x for x in kwargs.keys() if x not in filter_keys]
        if skipped_args:
            logger.warning(
                f"Arguments {skipped_args} skipped for op {func.__name__}"
            )

    filtered_dict = {
        filter_key: kwargs[filter_key]
        for filter_key in filter_keys
        if filter_key in kwargs
    }
    return filtered_dict


def filtered_func(func, **additional_args):
    """ Wrap `func` to take any input dict, arguments not used by `func` will be
          ignored
    """

    def ret_func(**kwargs):
        all_args = {**kwargs, **additional_args}
        filtered_args = filter_kwargs(func, all_args)
        return func(filtered_args)

    return ret_func


def unify_args(aargs):
    """ Return a dict of args """
    if aargs is None:
        return {}
    if isinstance(aargs, str):
        return {"name": aargs}
    assert isinstance(aargs, dict), f"args {aargs} must be a dict or a str"
    return aargs


def merge_unify_args(*args):
    from collections import ChainMap

    unified_args = [unify_args(x) for x in args]
    ret = dict(ChainMap(*unified_args))
    return ret


def update_dict(dest, src):
    """ Update the dict 'dest' recursively.
        Elements in src could be a callable function with signature
            f(key, curr_dest_val)
    """
    for key, val in src.items():
        if isinstance(val, collections.Mapping):
            # dest[key] could be None in the case of a dict
            cur_dest = dest.get(key, {}) or {}
            assert isinstance(cur_dest, dict), cur_dest
            dest[key] = update_dict(cur_dest, val)
        else:
            if callable(val) and key in dest:
                dest[key] = val(key, dest[key])
            else:
                dest[key] = val
    return dest


def merge(kwargs, **all_args):
    """ kwargs will override other arguments """
    return update_dict(all_args, kwargs)


def get_merged_dict(base, *new_dicts):
    ret = copy.deepcopy(base)
    for x in new_dicts:
        assert isinstance(x, dict)
        update_dict(ret, x)
    return ret


def format_dict_expanding_list_values(dic):
    """
    Formatting a dict into a multi-line string representation of its keys and
    values, if the value is a list, expand every element of that list into a new
    line with "-" in indentation (otherwise use space as indentation).
    Eg. {"aaa": [1, [2, 3]], "bbb": (1, 2, 3)} will become:
    aaa
    - 1
    - [2, 3]
    bbb
      (1, 2, 3)
    """
    dic = copy.deepcopy(dic)
    lines = []
    for k, v in dic.items():
        lines.append(k)
        if isinstance(v, list):
            for elem in v:
                lines.append("- {}".format(elem))
        else:
            lines.append("  {}".format(v))
    return "\n".join(lines)
