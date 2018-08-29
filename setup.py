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
from setuptools.command.install import install
from subprocess import Popen

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class InstallWithMakefile(install):
    """Custom install command."""
    def makeInstall(self):
        cmds = ["make", "pip-install"]
        process = Popen(cmds)
        if process.wait() != 0:
            print("Error: make install failed")
            exit(0)

    def run(self):
        self.makeInstall()
        install.run(self)


setup(
    name = 'NNITOOL',
    version = '0.0.1',
    packages = find_packages('src/sdk/pynni', exclude=['tests']) + find_packages('tools'),
    package_dir = {
        'nni': 'src/sdk/pynni/nni',
        'annotation': 'tools',
        'nnicmd': 'tools'
    },
    python_requires = '>=3.5',
    install_requires = [
        'json_tricks',
        'numpy',
        'pymc3',
        'scipy',
        'requests',
        'pyyaml',
        'psutil',
        'astor'
    ],
    dependency_links = [
        'git+https://github.com/hyperopt/hyperopt.git',
    ],

    cmdclass={
        'install': InstallWithMakefile
    },
    entry_points={
        'console_scripts': ['nnictl = nnicmd.nnictl:parse_args']
    },

    author = 'Microsoft NNI Team',
    author_email = 'NNIUsers@microsoft.com',
    description = 'Neural Network Intelligence project',
    long_description = read('docs/NNICTLDOC.md'),
    license = 'MIT',
    url = 'https://msrasrg.visualstudio.com/NeuralNetworkIntelligence'

    
)
