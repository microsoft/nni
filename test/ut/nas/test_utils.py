import tempfile
import pytest
from pathlib import Path

import torch

from nni.nas.utils import *

@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

def test_default_serializer():
    assert isinstance(get_default_serializer(), TorchSerializer)
    set_default_serializer(JsonSerializer())
    assert isinstance(get_default_serializer(), JsonSerializer)
    set_default_serializer(TorchSerializer())


def test_torch_serializer(tempdir, caplog):
    s = TorchSerializer()
    s.save(1, tempdir / 'test.ckpt')
    assert (tempdir / 'test.ckpt.torch').exists()

    assert s.load(tempdir / 'test.ckpt') == 1

    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test')
    assert 'does not match' in caplog.text

    caplog.clear()

    assert s.load(tempdir / 'test.ckpt.torch') == 1

    s.save(torch.randn(5), tempdir / 'test.ckpt')
    assert s.load(tempdir / 'test.ckpt').shape == (5,)


def test_json_serializer(tempdir, caplog):
    s = JsonSerializer()
    s.save({1: 5, 2: [3, 4]}, tempdir / 'test.ckpt')
    assert (tempdir / 'test.ckpt.json').exists()
    assert s.load(tempdir / 'test.ckpt') == {'1': 5, '2': [3, 4]}

    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test')
    assert 'does not match' in caplog.text


def test_mixed_serializer(tempdir, caplog):
    s = TorchSerializer()
    s.save(1, tempdir / 'test.ckpt')

    s = JsonSerializer()
    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test.ckpt')

    assert 'which could be loaded' in caplog.text

    s.save(2, tempdir / 'test.ckpt')
    assert s.load(tempdir / 'test.ckpt') == 2

    assert TorchSerializer().load(tempdir / 'test.ckpt') == 1
