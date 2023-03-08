import pytest

from nni.mutable import *
from nni.nas.strategy.utils import *


def test_dedup():
    deduper = DeduplicationHelper()
    assert deduper.dedup(1)
    assert deduper.dedup(2)
    assert not deduper.dedup(1)

    assert deduper.dedup({'a': 1, 'b': {'c': 3, 'd': [2, 3, 4]}})
    assert not deduper.dedup({'a': 1, 'b': {'c': 3, 'd': [2, 3, 4]}})

    deduper.remove({'a': 1, 'b': {'c': 3, 'd': [2, 3, 4]}})
    assert deduper.dedup({'a': 1, 'b': {'c': 3, 'd': [2, 3, 4]}})

    deduper.reset()
    assert deduper.dedup(1)

    deduper = DeduplicationHelper(True)
    assert deduper.dedup(1)
    with pytest.raises(DuplicationError):
        deduper.dedup(1)


def test_retry():
    helper = RetrySamplingHelper(10)
    assert helper.retry(lambda: 1) == 1

    class _keep_trying_and_will_success:
        def __init__(self):
            self._count = 0

        def __call__(self):
            self._count += 1
            if self._count < 5:
                raise KeyError()
            return 1

    with pytest.raises(KeyError):
        helper.retry(_keep_trying_and_will_success())

    helper = RetrySamplingHelper(10, KeyError)
    assert helper.retry(_keep_trying_and_will_success()) == 1

    helper = RetrySamplingHelper(3, KeyError)
    assert helper.retry(_keep_trying_and_will_success()) is None

    helper = RetrySamplingHelper(3, KeyError, raise_last=True)
    with pytest.raises(KeyError):
        helper.retry(_keep_trying_and_will_success())

