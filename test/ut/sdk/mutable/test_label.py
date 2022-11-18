import pytest

from nni.mutable import label_scope, auto_label
from nni.mutable.utils import reset_uid


def test_label_simple():
    with label_scope():
        label1 = auto_label()
        label2 = auto_label()
        with label_scope():
            label3 = auto_label()
            label4 = auto_label()
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
        label9 = auto_label()       # param/1

    assert label1 == 'model/1'
    assert label2 == 'model/2'
    assert label3 == 'model/foo'
    assert label4 == 'model/3/1'
    assert label5 == 'model/3/2'
    assert label6 == 'model/another/1'
    assert label7 == 'model/model/1'
    assert label8 == 'model/1'
    assert label9 == 'param/1'


def test_auto_label(caplog):
    reset_uid('global')

    label1 = auto_label('bar')          # bar, because the scope is global
    assert 'recommend' not in caplog.text
    label2 = auto_label()               # global/1, because label is not provided
    assert 'recommend' in caplog.text
    with label_scope('foo'):
        label3 = auto_label()           # foo/1, because in the scope "foo"
    with label_scope():
        label4 = auto_label()           # param/1, default key is used
    with label_scope('another'):
        label5 = auto_label()           # another/1
        label6 = auto_label('thing')    # another/thing
        label7 = auto_label()           # another/2

    assert label1 == 'bar'
    assert label2 == 'global/1'
    assert label3 == 'foo/1'
    assert label4 == 'param/1'
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


def test_label_validation(caplog):
    with pytest.raises(ValueError):
        auto_label('a/b')
    with pytest.raises(ValueError):
        auto_label('a_b')

    with pytest.raises(TypeError):
        auto_label(['abc'])

    auto_label('123')
    assert 'only digits' in caplog.text

    with pytest.raises(TypeError):
        auto_label('hello', 'world')

    with pytest.raises(ValueError, match='not entered'):
        with label_scope('test'):
            auto_label('hello', label_scope('world'))
