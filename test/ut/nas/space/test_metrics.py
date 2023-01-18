import pytest

import nni
from nni.nas.space import Metrics

def test_metrics():
    metrics = Metrics()
    metrics.add_intermediate(0.6)
    metrics.add_intermediate({'default': 0.7})
    metrics.add_intermediate({'default': 0.8, 'accuracy': 0.9})
    metrics.final = 1
    assert metrics.intermediates == [0.6, 0.7, 0.8]
    assert metrics.final == 1.0 and isinstance(metrics.final, float)
    metrics.final = {'default': 1.1}
    assert metrics.final == 1.1

    with pytest.raises(ValueError, match='default'):
        metrics.final = {'hello': 1.1}
    with pytest.raises(ValueError, match='not a number'):
        metrics.final = 'hello'

    assert nni.load(nni.dump(metrics)) == metrics

    metrics.clear()
    assert metrics.intermediates == []
    assert metrics.final == None

def test_metrics_non_strict():
    metrics = Metrics(strict=False)

    metrics.add_intermediate(0.6)
    metrics.add_intermediate({'default': 0.7})
    metrics.add_intermediate({'default': 0.8, 'accuracy': 0.9})
    metrics.add_intermediate([3, 4, 5])
    metrics.final = (3, 5)
    assert metrics.intermediates == [0.6, {'default': 0.7}, {'default': 0.8, 'accuracy': 0.9}, [3, 4, 5]]
    assert metrics.final == (3, 5)
