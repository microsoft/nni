"""
 0. Create a directory whose name is sys.argv[1].
 1. Create a "pip-install" script that uses this directory as prefix.
 2. Set PYTHONPATH to use this prefix, via VSO variable.
"""

import os
from pathlib import Path
import sys
import typing

prefix = Path(sys.argv[1])
prefix.mkdir(parents=True, exist_ok=True)
prefix = prefix.resolve()

if sys.platform == 'win32':
    # Do not use CMD script. It will magically stop pipeline step.
    script = f'python -m pip install --prefix {prefix} --no-compile --prefer-binary $args'
    Path('pip-install.ps1').write_text(script + '\n')
else:
    script = f'python -m pip install --prefix {prefix} --no-compile --prefer-binary "$@"'
    Path('pip-install').write_text('#!/bin/bash\n' + script + '\n')
    os.chmod('pip-install', 0o775)

if sys.platform == 'win32':
    site_path = prefix / 'Lib/site-packages'
else:
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    site_path = prefix / f'lib/python{version}/site-packages'
paths = [
    Path(typing.__file__).parent,  # With PYTHONPATH, the backport version of typing will mask stdlib version.
    site_path,
    os.getenv('PYTHONPATH'),
]
path = os.pathsep.join(str(p) for p in paths if p)

print('PYTHONPATH:', path)
print('##vso[task.setvariable variable=PYTHONPATH]' + path)
