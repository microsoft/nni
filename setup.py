# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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

    packages = find_packages('src/sdk/pynni', exclude=['tests']) + find_packages('src/sdk/pycli') + find_packages('tools'),
    package_dir = {
        'nni': 'src/sdk/pynni/nni',
        'nnicli': 'src/sdk/pycli/nnicli',
        'nni_annotation': 'tools/nni_annotation',
        'nni_cmd': 'tools/nni_cmd',
        'nni_trial_tool':'tools/nni_trial_tool',
        'nni_gpu_tool':'tools/nni_gpu_tool'
    },
    package_data = {'nni': ['**/requirements.txt']},
    python_requires = '>=3.5',
    install_requires = [
        'astor',
        'hyperopt==0.1.2',
        'json_tricks',
        'numpy',
        'psutil',
        'ruamel.yaml',
        'requests',
        'scipy',
        'schema',
        'PythonWebHDFS',
        'colorama',
        'scikit-learn>=0.20,<0.22'
    ],

    entry_points = {
        'console_scripts' : [
            'nnictl = nni_cmd.nnictl:parse_args'
        ]
    }
)
