import os
import sys

invalid_files = []

copyright_headers = [
    '# Copyright (c) Microsoft Corporation.\n# Licensed under the MIT license.',
    '# Copyright (c) Microsoft Corporation. All rights reserved.\n#\n# MIT License',
]

whitelist = [
    'nni/version.py',
    'nni/algorithms/hpo/bohb_advisor/config_generator.py',
    'nni/compression/quantization_speedup/trt_pycuda.py',
]

for root, dirs, files in os.walk('nni'):
    for file in files:
        if not file.endswith('.py'):
            continue
        full_path = os.path.join(root, file)
        if full_path in whitelist:
            continue
        content = open(full_path).read()
        if not content.strip():
            # empty file
            continue
        if not any(content.startswith(header) for header in copyright_headers):
            invalid_files.append(full_path)

if invalid_files:
    print("The following files doesn't have a copyright text header.\n")
    for file in invalid_files:
        print('    ' + file)
    print('\nPlease add the following text at the beginning of the file.\n')
    print('# Copyright (c) Microsoft Corporation.\n# Licensed under the MIT license.')
    sys.exit(1)
