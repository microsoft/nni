# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import reduce
from typing import Any, overload


@overload
def get_nested_attr(__o: object, __name: str) -> Any:
    ...

@overload
def get_nested_attr(__o: object, __name: str, __default: Any) -> Any:
    ...

def get_nested_attr(__o: object, __name: str, *args) -> Any:
    """
    Get a nested named attribute from an object by a `.` separated name.
    rgetattr(x, 'y.z') is equivalent to getattr(getattr(x, 'y'), 'z') and x.y.z.
    """
    def _getattr(__o, __name):
        return getattr(__o, __name, *args)
    return reduce(_getattr, [__o] + __name.split('.'))  # type: ignore


def set_nested_attr(__obj: object, __name: str, __value: Any):
    """
    Set the nested named attribute on the given object to the specified value by a `.` separated name.
    set_nested_attr(x, 'y.z', v) is equivalent to setattr(getattr(x, 'y'), 'z', v) x.y.z = v.
    """
    pre, _, post = __name.rpartition('.')
    return setattr(get_nested_attr(__obj, pre) if pre else __obj, post, __value)


def has_nested_attr(__obj: object, __name: str) -> bool:
    """
    Determine whether a given object has an attribute with a `.` separated name.
    """
    pre, _, post = __name.rpartition('.')
    if pre:
        if has_nested_attr(__obj, pre):
            return has_nested_attr(get_nested_attr(__obj, pre), post)
        else:
            return False
    else:
        return hasattr(__obj, post)
