# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name = 'demo-tuner',
    version = '0.1',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.6',
    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'NNI control for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni'
)
