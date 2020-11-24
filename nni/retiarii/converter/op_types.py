MODULE_EXCEPT_LIST = ['Sequential']
RETIARII_BASE_OPS = ['Placeholder']

class Type:
    """Node Type class
    """
    Attr = 'Attr'
    Constant = 'Constant'
    ListConstruct = 'ListConstruct'
    LayerChoice = 'LayerChoice'
    InputChoice = 'InputChoice'
    ValueChoice = 'ValueChoice'

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