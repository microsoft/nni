import pytest
import sys

from nni.common.version import version_dump, version_check


def test_version_dump():
    dump_ver = version_dump()
    assert len(dump_ver) >= 9
    print(dump_ver)


def test_version_check():
    version_check(version_dump(), raise_error=True)
    version_check({'python': sys.version_info[:2]}, raise_error=True)

    with pytest.warns(UserWarning):
        version_check({'nni': (99999, 99999)})

    with pytest.raises(RuntimeError):
        version_check({'python': (2, 7)}, raise_error=True)
