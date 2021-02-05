import sys

from _common import run_command

name = sys.argv[1]
run_command(f'docker container stop {name}')
run_command(f'docker container rm {name}')
