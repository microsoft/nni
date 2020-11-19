def build_full_name(prefix, name, seq=None):
    if seq is None:
        return '{}.{}'.format(prefix, name)
    else:
        return '{}.{}{}'.format(prefix, name, str(seq))