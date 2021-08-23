import sys

from _common import build_wheel, run_command

if len(sys.argv) <= 2:
    extra_dep = ''
else:
    extra_dep = f'[{sys.argv[2]}]'

wheel = build_wheel()
run_command(f'{sys.executable} -m pip install {wheel}{extra_dep}')
