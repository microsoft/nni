# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name = 'nni-tool',
    version = '999.0.0-developing',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.6',
    install_requires = [
        'requests',
        'responses',
        'ruamel.yaml',
        'psutil',
        'astor',
        'schema',
        'PythonWebHDFS',
        'colorama',
        'netifaces',
        'websockets'
    ],

    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'NNI control for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',
    entry_points = {
        'console_scripts' : [
            'nnictl = nni_cmd.nnictl:parse_args'
        ]
    }
)
