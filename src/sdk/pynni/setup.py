# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import setuptools

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setuptools.setup(
    name = 'nni-sdk',
    version = '999.0.0-developing',
    packages = setuptools.find_packages(exclude=['tests']),

    python_requires = '>=3.6',
    install_requires = [
        'hyperopt==0.1.2',
        'json_tricks',
        'numpy',
        'scipy',
        'coverage'
    ],
    package_data = {'nni': ['**/requirements.txt']},

    test_suite = 'tests',

    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'Python SDK for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',

    long_description = read('README.md')
)
