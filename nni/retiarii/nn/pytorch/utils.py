from typing import Optional

from ...utils import uid, get_current_context


def generate_new_label(label: Optional[str]):
    if label is None:
        return '_mutation_' + str(uid('mutation'))
    return label


def get_fixed_value(label: str):
    ret = get_current_context('fixed')
    try:
        return ret[generate_new_label(label)]
    except KeyError:
        raise KeyError(f'Fixed context with {label} not found. Existing values are: {ret}')
