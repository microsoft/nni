from enum import Enum

MODULE_EXCEPT_LIST = ['Sequential']


class OpTypeName(str, Enum):
    """
    op type to its type name str
    """
    Attr = 'Attr'
    Constant = 'Constant'
    ListConstruct = 'ListConstruct'
    ListUnpack = 'ListUnpack'
    TupleConstruct = 'TupleConstruct'
    TupleUnpack = 'TupleUnpack'
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
    'aten::Bool': 'Bool',
    'aten::empty': 'Empty',
    'aten::zeros': 'Zeros',
    'aten::chunk': 'Chunk',
    'aten::add_': 'Add_',  # %out.3 : Tensor = aten::add_(%out.1, %connection.1, %4)
    'aten::flatten': 'Flatten',
    'aten::sigmoid': 'Sigmoid',
    'aten::detach': 'Detach',
    'aten::le': 'Le',
    'aten::new_zeros': 'NewZeros',
    'aten::__not__': 'not',
    'aten::transpose': 'Transpose',
    'aten::contiguous': 'Contiguous'
}

BasicOpsTF = {}
