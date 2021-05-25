# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entrypoint for trials.
"""
import argparse
from pathlib import Path
import re

from .auto_compress_engine import AutoCompressEngine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trial entry for auto compress.')
    parser.add_argument('--module_file_name', required=True, dest='module_file_name', help='the path of auto compress module file')
    parser.add_argument('--module_class_name', required=True, dest='module_class_name', help='the name of auto compress module')
    args = parser.parse_args()

    module_name = Path(args.module_file_name).as_posix()
    module_name = re.sub(re.escape('.py') + '$', '', module_name).replace('/', '.') + '.' + args.module_class_name
    AutoCompressEngine.trial_execute_compress(module_name)
