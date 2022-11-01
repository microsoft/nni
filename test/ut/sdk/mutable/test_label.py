from nni.mutable import LabelNamespace, auto_label


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


def test_label_reproducible():
    labels = []
    for _ in range(2):
        with LabelNamespace('default'):
            label1 = auto_label('default')
            label2 = auto_label()
            labels.append(label2)
        assert label1 == 'default_1'
    assert len(set(labels)) == 2
