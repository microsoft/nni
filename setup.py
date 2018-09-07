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

class CustomInstallCommand(install):
    '''a customized install class in pip module'''
    def makeInstall(self):
        '''execute make pip-install command'''
        cmds = ['make', 'pip-install']
        process = Popen(cmds)
        if process.wait() != 0:
            print('Error: Make Install Failed')
            exit(-1)

    def writeEnvironmentVariables(self, variable_name):
        '''write an environment variable into ~/.bashrc'''
        paths = os.getenv("PATH").split(':')
        bin_path = os.path.join(os.getenv('HOME'),'.local/'+variable_name+'/bin')
        
        if bin_path not in paths:
            bashrc_path = os.path.join(os.getenv('HOME'), '.bashrc')
            process = Popen('echo export PATH=' + bin_path + ':\$PATH >> ' + bashrc_path, shell=True)
            if process.wait() != 0:
                print('Error: Write Environment Variables Failed')
                exit(-1)

    def run(self):
        install.run(self)
        self.makeInstall()
        self.writeEnvironmentVariables('node')
        self.writeEnvironmentVariables('yarn')

setup(
    name = 'NNI',
    version = '0.0.1',
    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence project',
    long_description = read('docs/NNICTLDOC.md'),
    license = 'MIT',
    url = 'https://msrasrg.visualstudio.com/NeuralNetworkIntelligence',

    packages = find_packages('src/sdk/pynni', exclude=['tests']) + find_packages('tools'),
    package_dir = {
        'annotation': 'tools/nni_annotation',
        'nni': 'src/sdk/pynni/nni',
        'nnicmd': 'tools/nnicmd'
    },
    python_requires = '>=3.5',
    install_requires = [
        'astor',
        'json_tricks',
        'numpy',
        'psutil',
        'pymc3',
        'pyyaml',
        'requests',
        'scipy'
        
    ],
    dependency_links = [
        'git+https://github.com/hyperopt/hyperopt.git',
    ],

    cmdclass={
        'install': CustomInstallCommand
    },
    entry_points={
        'console_scripts': ['nnictl = nnicmd.nnictl:parse_args']
    }
)
