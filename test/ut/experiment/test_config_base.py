from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from nni.experiment.config.base import ConfigBase
from nni.experiment.config.utils import diff

# config classes

@dataclass(init=False)
class NestedChild(ConfigBase):
    msg: str
    int_field: int = 1

    def _canonicalize(self, parents):
        if '/' not in self.msg:
            self.msg = parents[0].msg + '/' + self.msg
        super()._canonicalize(parents)

    def _validate_canonical(self):
        super()._validate_canonical()
        if not self.msg.endswith('[2]'):
            raise ValueError('not end with [2]')

@dataclass(init=False)
class Child(ConfigBase):
    msg: str
    children: List[NestedChild]

    def _canonicalize(self, parents):
        if '/' not in self.msg:
            self.msg = parents[0].msg + '/' + self.msg
        super()._canonicalize(parents)

    def _validate_canonical(self):
        super()._validate_canonical()
        if not self.msg.endswith('[1]'):
            raise ValueError('not end with "[1]"')

@dataclass(init=False)
class TestConfig(ConfigBase):
    msg: str
    required_field: Optional[int]
    optional_field: Optional[int] = None
    multi_type_field: Union[int, List[int]]
    child: Optional[Child] = None

    def _canonicalize(self, parents):
        if isinstance(self.multi_type_field, int):
            self.multi_type_field = [self.multi_type_field]
        super()._canonicalize(parents)

# sample inputs

good = {
    'msg': 'a',
    'required_field': 10,
    'multi_type_field': 20,
    'child': {
        'msg': 'b[1]',
        'children': [{
            'msg': 'c[2]',
            'int_field': 30,
        }, {
            'msg': 'd[2]',
        }],
    },
}

missing = deepcopy(good)
missing.pop('required_field')

wrong_type = deepcopy(good)
wrong_type['optional_field'] = 0.5

nested_wrong_type = deepcopy(good)
nested_wrong_type['child']['children'][1]['int_field'] = 'str'

bad_value = deepcopy(good)
bad_value['child']['msg'] = 'b'

extra_field = deepcopy(good)
extra_field['hello'] = 'world'

bads = {
    'missing': missing,
    'wrong_type': wrong_type,
    'nested_wrong_type': nested_wrong_type,
    'bad_value': bad_value,
    'extra_field': extra_field,
}

# ground truth

_nested_child_1 = NestedChild()
_nested_child_1.msg = 'c[2]'
_nested_child_1.int_field = 30

_nested_child_2 = NestedChild()
_nested_child_2.msg = 'd[2]'
_nested_child_2.int_field = 1

_child = Child()
_child.msg = 'b[1]'
_child.children = [_nested_child_1, _nested_child_2]

good_config = TestConfig()
good_config.msg = 'a'
good_config.required_field = 10
good_config.optional_field = None
good_config.multi_type_field = 20
good_config.child = _child

_nested_child_1 = NestedChild()
_nested_child_1.msg = 'a/b[1]/c[2]'
_nested_child_1.int_field = 30

_nested_child_2 = NestedChild()
_nested_child_2.msg = 'a/b[1]/d[2]'
_nested_child_2.int_field = 1

_child = Child()
_child.msg = 'a/b[1]'
_child.children = [_nested_child_1, _nested_child_2]

good_canon_config = TestConfig()
good_canon_config.msg = 'a'
good_canon_config.required_field = 10
good_canon_config.optional_field = None
good_canon_config.multi_type_field = [20]
good_canon_config.child = _child

# test function

def test_good():
    config = TestConfig(**good)
    assert config == good_config
    config.validate()
    assert config.json() == good_canon_config.json()

def test_bad():
    for tag, bad in bads.items():
        exc = None
        try:
            config = TestConfig(**bad)
            config.validate()
        except Exception as e:
            exc = e
        assert exc is not None

def test_diff():
    config = TestConfig(**good)
    assert diff(config, good_config) == ''
    another = deepcopy(good)
    another['msg'] = 'another'
    assert 'another/b' in diff(config, TestConfig(**another))

if __name__ == '__main__':
    test_good()
    test_bad()
