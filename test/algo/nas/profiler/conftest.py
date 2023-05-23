import pytest

from nni.mutable import frozen_context


@pytest.fixture(autouse=True)
def context():
    with frozen_context():
        yield
