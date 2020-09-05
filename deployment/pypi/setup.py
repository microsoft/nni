# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools
from setuptools.command.install import install
import platform
import os
from os import walk, path

os_type = platform.system()
if os_type == 'Linux':
    os_name = 'POSIX :: Linux'
elif os_type == 'Darwin':
    os_name = 'MacOS'
elif os_type == 'Windows':
    os_name = 'Microsoft :: Windows'
else:
    raise NotImplementedError('current platform {} not supported'.format(os_type))

version = '1.8'

class AutoCompletion(install):
    def run(self):
        COMP_URL = 'https://raw.githubusercontent.com/microsoft/nni/v{}/tools/bash-completion'.format(version)
        if os_type == 'Linux':
            HOME = os.environ.get('HOME')
            if not os.geteuid(): # run as root
                BASH_COMP_PREFIX = '/usr/share/bash-completion/completions'
            else:
                if not HOME:
                    install.run(self)
                    return # can not get $HOME, abort.
                BASH_COMP_PREFIX = os.path.join(HOME, '.bash_completion.d')
            # install auto completion script to /usr/share/bash-completion/completions/nnictl for root or 
            #  ~/.bash_completion.d/nnictl for normal user
            BASH_COMP_SCRIPT = os.path.join(BASH_COMP_PREFIX, 'nnictl')
            os.system('mkdir -p {} && wget -O {} {} && chmod 644 {}'.format(BASH_COMP_PREFIX,
                                                                            BASH_COMP_SCRIPT,
                                                                            COMP_URL,
                                                                            BASH_COMP_SCRIPT))
            # not root and completion not installed
            if os.geteuid():
                NEED_SOURCE = os.path.join(HOME, '.bash_completion')
                if not os.popen('(source {} ; command -v _nnictl) 2>/dev/null'.format(NEED_SOURCE)).read():
                    os.system("echo '[[ -f {} ]] && source {}' >> {}".format(BASH_COMP_SCRIPT,
                                                                             BASH_COMP_SCRIPT,
                                                                             NEED_SOURCE))
        install.run(self)

data_files = [('bin', ['node-{}-x64/bin/node'.format(os_type.lower())])]
if os_type == 'Windows':
    data_files = [('.\Scripts', ['node-{}/node.exe'.format(os_type.lower())])]

for (dirpath, dirnames, filenames) in walk('./nni'):
    files = [path.normpath(path.join(dirpath, filename)) for filename in filenames]
    data_files.append((path.normpath(dirpath), files))

with open('../../README.md', 'r', encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'nni',
    version = version,
    author = 'Microsoft NNI team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence package',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',
    packages = setuptools.find_packages('../../tools') \
        + setuptools.find_packages('../../src/sdk/pynni', exclude=['tests']) \
        + setuptools.find_packages('../../src/sdk/pycli'),
    package_dir = {
        'nni_annotation': '../../tools/nni_annotation',
        'nni_cmd': '../../tools/nni_cmd',
        'nni_trial_tool': '../../tools/nni_trial_tool',
        'nni_gpu_tool': '../../tools/nni_gpu_tool',
        'nni': '../../src/sdk/pynni/nni',
        'nnicli': '../../src/sdk/pycli/nnicli'
    },
    package_data = {'nni': ['**/requirements.txt']},
    python_requires = '>=3.6',
    install_requires = [
        'schema',
        'ruamel.yaml',
        'psutil',
        'requests',
        'astor',
        'PythonWebHDFS',
        'hyperopt==0.1.2',
        'json_tricks',
        'netifaces',
        'numpy',
        'scipy',
        'coverage',
        'colorama',
        'scikit-learn>=0.23.2',
        'pkginfo',
        'websockets'
    ],
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: ' + os_name
    ],
    data_files = data_files,
    entry_points = {
        'console_scripts' : [
            'nnictl = nni_cmd.nnictl:parse_args'
        ]
    },
    cmdclass = {
        'install': AutoCompletion
    }
)
