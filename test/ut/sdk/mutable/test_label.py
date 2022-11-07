from nni.mutable import LabelNamespace, auto_label
from nni.mutable.utils import reset_uid


def test_label_simple():
    with LabelNamespace():
        label1 = auto_label()
        label2 = auto_label()
        with LabelNamespace():
            label3 = auto_label()
            label4 = auto_label()
        label5 = auto_label()
        with LabelNamespace('another'):
            label6 = auto_label('another')
        label7 = auto_label()

    assert label1 == 'param_1'
    assert label2 == 'param_2'
    assert label3 == 'param_3_1'
    assert label4 == 'param_3_2'
    assert label5 == 'param_4'
    assert label6 == 'another_1'
    assert label7 == 'param_5'

    with LabelNamespace('param'):
        label1 = auto_label()       # param_1
        label2 = auto_label()       # param_2
        with LabelNamespace('param'):
            label3 = auto_label()   # param_3_1
            label4 = auto_label()   # param_3_2
        with LabelNamespace('another'):
            label5 = auto_label('another')   # another_1
        with LabelNamespace('param'):
            label6 = auto_label()   # param_4_1
    with LabelNamespace('param'):
        label7 = auto_label()       # param_1, because the counter is reset

    assert label1 == 'param_1'
    assert label2 == 'param_2'
    assert label3 == 'param_3_1'
    assert label4 == 'param_3_2'
    assert label5 == 'another_1'
    assert label6 == 'param_4_1'
    assert label7 == 'param_1'


def test_auto_label(caplog):
    reset_uid('global')

    label1 = auto_label()               # global_1
    with LabelNamespace('param'):
        label2 = auto_label('param')    # param_1, because in the namespace "param"
    with LabelNamespace():
        label3 = auto_label()           # param_1, default key is used
    with LabelNamespace('another'):
        label4 = auto_label('another')  # another_1
        label5 = auto_label()           # global_2

    assert label1 == 'global_1'
    assert 'recommend' in caplog.text
    assert label2 == 'param_1'
    assert label3 == 'param_1'
    assert label4 == 'another_1'
    assert label5 == 'global_2'


def test_label_reproducible():
    labels = []
    for _ in range(2):
        with LabelNamespace('default'):
            label1 = auto_label('default')
            label2 = auto_label()
            labels.append(label2)
        assert label1 == 'default_1'
    assert len(set(labels)) == 2
