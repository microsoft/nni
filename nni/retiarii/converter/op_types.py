MODULE_EXCEPT_LIST = ['Sequential']


class Type:
    """Node Type class
    """
    Attr = 'Attr'
    Constant = 'Constant'
    ListConstruct = 'ListConstruct'
    LayerChoice = 'LayerChoice'
    InputChoice = 'InputChoice'
    ValueChoice = 'ValueChoice'
    Placeholder = 'Placeholder'

    MergedSlice = 'MergedSlice'

    # deal with aten op
    BasicOpsPT = {
        'aten::mean': 'Mean',
        'aten::relu': 'Relu',
        'aten::add': 'Add',
        'aten::__getitem__': 'getitem',
        'aten::append': 'Append',
        'aten::len': 'Len',
        'aten::slice': 'Slice',
        'aten::cat': 'Cat',
        'aten::size': 'Size',
        'aten::view': 'View'
    }

    BasicOpsTF = {}