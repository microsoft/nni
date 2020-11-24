# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name='nnicli',
    version='999.0.0-developing',
    packages=setuptools.find_packages(),

    python_requires='>=3.6',
    install_requires=[
        'requests'
    ],

    author='Microsoft NNI Team',
    author_email='nni@microsoft.com',
    description='nnicli for Neural Network Intelligence project',
    license='MIT',
    url='https://github.com/Microsoft/nni',
)
