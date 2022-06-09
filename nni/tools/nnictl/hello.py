# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Provide ``nnictl hello`` command to generate quickstart example.
"""

from pathlib import Path
import shutil

from colorama import Fore

import nni_assets

def create_example(_args):
    example_path = Path(nni_assets.__path__[0], 'hello_hpo')
    try:
        shutil.copytree(example_path, 'nni_hello_hpo')
    except PermissionError:
        print(Fore.RED + 'Permission denied. Please run the command in a writable directory.' + Fore.RESET)
        exit(1)
    except FileExistsError:
        print('File exists. Please run "python nni_hello_hpo/main.py" to start the example.')
        exit(1)
    print('A hyperparameter optimization example has been created at "nni_hello_hpo" directory.')
    print('Please run "python nni_hello_hpo/main.py" to try it out.')
