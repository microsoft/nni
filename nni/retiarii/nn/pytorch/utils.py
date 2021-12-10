from typing import Any, Optional, Tuple

from nni.retiarii.utils import ModelNamespace, get_current_context


def generate_new_label(label: Optional[str]):
    if label is None:
        return ModelNamespace.next_label()
    return label


def get_fixed_value(label: str) -> Any:
    ret = get_current_context('fixed')
    try:
        return ret[generate_new_label(label)]
    except KeyError:
        raise KeyError(f'Fixed context with {label} not found. Existing values are: {ret}')


def get_fixed_dict(label_prefix: str) -> Tuple[str, Any]:
    ret = get_current_context('fixed')
    try:
        label_prefix = generate_new_label(label_prefix)
        ret = {k: v for k, v in ret.items() if k.startswith(label_prefix + '/')}
        if not ret:
            raise KeyError
        return label_prefix, ret
    except KeyError:
        raise KeyError(f'Fixed context with prefix {label_prefix} not found. Existing values are: {ret}')
