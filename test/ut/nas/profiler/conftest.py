import pytest
from packaging.version import Version

from nni.mutable import frozen_context


@pytest.fixture(autouse=True)
def context():
    with frozen_context():
        yield

@pytest.fixture(autouse=True)
def skip_for_legacy_pytorch():
    import torch
    if Version(torch.__version__) < Version('1.11.0'):
        pytest.skip('PyTorch version is too old, skip this test.')
