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


import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setup(
    name = 'nni',
    version = '999.0.0-developing',
    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence project',
    long_description = read('README.md'),
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',

    packages = find_packages('src/sdk/pynni', exclude=['tests']) + find_packages('tools'),
    package_dir = {
        'nni': 'src/sdk/pynni/nni',
        'nni_annotation': 'tools/nni_annotation',
        'nni_cmd': 'tools/nni_cmd',
        'nni_trial_tool':'tools/nni_trial_tool',
        'nni_gpu_tool':'tools/nni_gpu_tool'
    },
    package_data = {'nni': ['**/requirements.txt']},
    python_requires = '>=3.5',
    install_requires = [
        'astor',
        'hyperopt',
        'json_tricks',
        'numpy',
        'psutil',
        'ruamel.yaml',
        'requests',
        'scipy',
        'schema',
        'PythonWebHDFS',
        'colorama',
        'sklearn'
    ],

    entry_points = {
        'console_scripts' : [
            'nnictl = nni_cmd.nnictl:parse_args'
        ]
    }
)
