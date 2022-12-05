import pytest

from nni.mutable import label_scope, auto_label
from nni.mutable.utils import reset_uid


def test_label_simple():
    reset_uid('global')

    with label_scope('param'):
        label1 = auto_label()
        label2 = auto_label()
        with label_scope() as scope:
            assert scope.activated
            label3 = auto_label()
            label4 = auto_label()
        assert not scope.activated
        label5 = auto_label()
        with label_scope('another'):
            label6 = auto_label('another')
        label7 = auto_label()

    assert label1 == 'param/1'
    assert label2 == 'param/2'
    assert label3 == 'param/3/1'
    assert label4 == 'param/3/2'
    assert label5 == 'param/4'
    assert label6 == 'param/another/another'
    assert label7 == 'param/5'

    with label_scope('model'):
        label1 = auto_label()       # model/1
        label2 = auto_label()       # model/2
        label3 = auto_label('foo')  # model/foo
        with label_scope():
            label4 = auto_label()   # model/3/1
            label5 = auto_label()   # model/3/2
        with label_scope('another'):
            label6 = auto_label()   # model/another/1
        with label_scope('model'):
            label7 = auto_label()   # model/model/1
    with label_scope('model'):
        label8 = auto_label()       # model/1, because the counter is reset
    with label_scope():
        label9 = auto_label()       # global/1/1

    assert label1 == 'model/1'
    assert label2 == 'model/2'
    assert label3 == 'model/foo'
    assert label4 == 'model/3/1'
    assert label5 == 'model/3/2'
    assert label6 == 'model/another/1'
    assert label7 == 'model/model/1'
    assert label8 == 'model/1'
    assert label9 == 'global/1/1'


def test_auto_label(caplog):
    reset_uid('global')

    label1 = auto_label('bar')
    assert 'recommend' not in caplog.text
    label2 = auto_label()
    assert 'recommend' in caplog.text
    with label_scope('foo'):
        label3 = auto_label()
    with label_scope():
        label4 = auto_label()
    with label_scope('another'):
        label5 = auto_label()
        label6 = auto_label('thing')
        label7 = auto_label()

    assert label1 == 'bar'
    assert label2 == 'global/1'
    assert label3 == 'foo/1'
    assert label4 == 'global/2/1'
    assert label5 == 'another/1'
    assert label6 == 'another/thing'
    assert label7 == 'another/2'


def test_label_reproducible():
    labels = []
    for _ in range(2):
        with label_scope('default'):
            label1 = auto_label('default')
            label2 = auto_label()
            labels.append(label2)
        assert label1 == 'default/default'
        assert label2 == 'default/1'


def test_label_validation():
    with pytest.raises(ValueError):
        auto_label('a/b')

    with pytest.raises(TypeError):
        auto_label(['abc'])

    auto_label('a_b')

    auto_label('123')

    with pytest.raises(TypeError):
        auto_label('hello', 'world')

    with pytest.raises(ValueError, match='not entered'):
        with label_scope('test'):
            auto_label('hello', label_scope('world'))


def test_scope_reenter():
    scope1 = label_scope('abc')
    scope2 = label_scope('def')
    with scope1:
        with scope2:
            assert scope1.name == 'abc'
            assert scope2.name == 'abc/def'
            assert auto_label() == 'abc/def/1'
    with scope2:
        with scope1:
            assert scope1.name == 'abc'
            assert scope2.name == 'abc/def'
            assert auto_label() == 'abc/1'

    scope3 = label_scope('abc')
    scope4 = label_scope('def')
    with scope3:
        with scope4:
            assert scope4 == scope2

    scope5 = label_scope('abc')
    scope6 = label_scope('def')
    with scope5:
        with scope6:
            with scope5:
                assert auto_label() == 'abc/1'


def test_label_idempotent():
    label = auto_label('abc')
    assert label == auto_label(label)
    with label_scope(label) as scope:
        label1 = auto_label()
        assert label1 == 'abc/1'
    with label_scope(scope) as scope:
        label2 = auto_label()
        assert label2 == 'abc/1'

    label = label_scope('def')
    with label:
        label1 = label.name
    with label_scope('ghi'):
        assert auto_label(label1) == label1
        assert auto_label('abc').startswith('ghi')


def test_label_replace():
    from nni.mutable.utils import label
    l = label(['model', '1'])
    assert isinstance(l, label)
    assert isinstance(l, str)
    assert l.replace('/', '__') == 'model__1'
