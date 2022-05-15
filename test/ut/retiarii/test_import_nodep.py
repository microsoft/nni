import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('framework', ['torch', 'tensorflow'])
def test_import_without_framework(framework):
    subprocess.run([sys.executable, Path(__file__).parent / '_import_nodep.py', framework])
