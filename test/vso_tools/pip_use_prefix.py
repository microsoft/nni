"""
 0. Create a directory whose name is sys.argv[1].
 1. Create a "pip-install" script that uses this directory as prefix.
 2. Set PYTHONPATH to use this prefix, via VSO variable.
"""

import os
from pathlib import Path
import sys

prefix = Path(sys.argv[1])
prefix.mkdir(parents=True, exist_ok=True)
prefix = prefix.resolve()

if sys.platform == 'win32':
    script = f'python -m pip install --prefix {prefix} %*'
    Path('pip-install.cmd').write_text(script + '\n')
else:
    script = f'python -m pip install --prefix {prefix} "$@"'
    Path('pip-install').write_text('#!/bin/bash\n' + script + '\n')
    os.chmod('pip-install', 0o775)

version = f'{sys.version_info.major}.{sys.version_info.minor}'
path = str(prefix / f'lib/python{version}/site-packages')
if os.getenv('PYTHONPATH'):
    path = os.getenv('PYTHONPATH') + os.pathsep + path
print('##vso[task.setvariable variable=PYTHONPATH]' + path)
