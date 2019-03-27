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
import setuptools

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setuptools.setup(
    name = 'nni-sdk',
    version = '999.0.0-developing',
    packages = setuptools.find_packages(exclude=['tests']),

    python_requires = '>=3.5',
    install_requires = [
        'hyperopt',
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
