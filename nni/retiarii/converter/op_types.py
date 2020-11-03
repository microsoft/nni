MODULE_EXCEPT_LIST = ['Sequential']
RETIARII_BASE_OPS = ['Placeholder']

class Type:
    """Node Type class
    """
    Attr = 'Attr'
    Constant = 'Constant'
    ListConstruct = 'ListConstruct'

    # deal with aten op
    BasicOpsPT = {
        'aten::mean': 'Mean',
        'aten::relu': 'Relu',
        'aten::add': 'Add'
    }

    BasicOpsTF = {}