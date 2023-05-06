# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import sys

import nni

def main():
    data = {}
    data['nniVersion'] = nni.__version__
    data['nniPath'] = nni.__path__[0]  # type: ignore
    data['pythonVersion'] = '{}.{}.{}-{}-{}'.format(*sys.version_info)
    data['pythonPath'] = sys.executable
    print(json.dumps(data))

if __name__ == '__main__':
    main()
