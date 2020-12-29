def build_full_name(prefix, name, seq=None):
    if isinstance(name, list):
        name = '__'.join(name)
    if seq is None:
        return '{}__{}'.format(prefix, name)
    else:
        return '{}__{}{}'.format(prefix, name, str(seq))


def _convert_name(name: str) -> str:
    """
    Convert the names using separator '.' to valid variable name in code
    """
    return name.replace('.', '__')
