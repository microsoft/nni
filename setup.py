# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from setuptools import setup


version = '999.0.0-developing'


def _find_python_packages():
    packages = []
    for dirpath, dirnames, filenames in os.walk('nni'):
        if '/__pycache__' not in dirpath:
            packages.append(dirpath.replace('/', '.'))
    return sorted(packages) + ['nni_node']

def _find_node_files():
    files = []
    for dirpath, dirnames, filenames in os.walk('nni_node'):
        for filename in filenames:
            files.append((dirpath + '/' + filename)[len('nni_node/'):])
    files.remove('__init__.py')
    return sorted(files)


setup(
    name = 'nni',
    version = version,
    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence project',
    long_description = open('README.md', encoding='utf-8').read(),
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',

    packages = _find_python_packages(),
    package_data = {
        'nni': ['**/requirements.txt'],
        'nni_node': _find_node_files()
    },

    python_requires = '>=3.6',
    install_requires = [
        'astor',
        'hyperopt==0.1.2',
        'json_tricks',
        'netifaces',
        'numpy',
        'psutil',
        'ruamel.yaml',
        'requests',
        'responses',
        'scipy',
        'schema',
        'PythonWebHDFS',
        'colorama',
        'scikit-learn>=0.23.2',
        'pkginfo',
        'websockets'
    ],

    entry_points = {
        'console_scripts' : [
            'nnictl = nni.nni_cmd.nnictl:parse_args'
        ]
    }
)
