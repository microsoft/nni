# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for experiment config classes.
"""

import dataclasses

def is_missing(value):
    """
    Used to check whether a dataclass field has ever been assigned.

    If a field without default value has never been assigned, it will have a special value ``MISSING``.
    This function checks if the parameter is ``MISSING``.
    """
    # MISSING is not singleton and there is no official API to check it
    return isinstance(value, type(dataclasses.MISSING))
