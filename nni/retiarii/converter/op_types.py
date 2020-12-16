from enum import Enum

MODULE_EXCEPT_LIST = ['Sequential']


class OpTypeName(str, Enum):
    """
    op type to its type name str
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
    'aten::view': 'View',
    'aten::eq': 'Eq',
    'aten::add_': 'Add_'  # %out.3 : Tensor = aten::add_(%out.1, %connection.1, %4)
}

BasicOpsTF = {}
