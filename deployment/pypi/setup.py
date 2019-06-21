# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

import setuptools
import platform
from os import walk, path

os_type = platform.system()
if os_type == 'Linux':
    os_name = 'POSIX :: Linux'
elif os_type == 'Darwin':
    os_name = 'MacOS'
elif os_type == 'Windows':
    os_name = 'Microsoft :: Windows'
else:
    raise NotImplementedError('current platform {} not supported'.format(os_type))

data_files = [('bin', ['node-{}-x64/bin/node'.format(os_type.lower())])]
if os_type == 'Windows':
    data_files = [('.\Scripts', ['node-{}/node.exe'.format(os_type.lower())])]

for (dirpath, dirnames, filenames) in walk('./nni'):
    files = [path.normpath(path.join(dirpath, filename)) for filename in filenames]
    data_files.append((path.normpath(dirpath), files))

with open('../../README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'nni',
    version = '999.0.0-developing',
    author = 'Microsoft NNI team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence package',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',
    packages = setuptools.find_packages('../../tools') + setuptools.find_packages('../../src/sdk/pynni', exclude=['tests']),
    package_dir = {
        'nni_annotation': '../../tools/nni_annotation',
        'nni_cmd': '../../tools/nni_cmd',
        'nni_trial_tool': '../../tools/nni_trial_tool',
        'nni_gpu_tool': '../../tools/nni_gpu_tool',
        'nni': '../../src/sdk/pynni/nni'
    },
    package_data = {'nni': ['**/requirements.txt']},
    python_requires = '>=3.5',
    install_requires = [
        'schema',
        'ruamel.yaml',
        'psutil',
        'requests',
        'astor',
        'PythonWebHDFS',
        'hyperopt',
        'json_tricks',
        'numpy',
        'scipy',
        'coverage',
        'colorama',
        'sklearn'
    ],
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: ' + os_name
    ],
    data_files = data_files,
    entry_points = {
        'console_scripts' : [
            'nnictl = nni_cmd.nnictl:parse_args'
        ]
    }
)
