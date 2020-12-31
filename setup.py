# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script for installation and distribution.

You can use environment variable `NNI_RELEASE` to set release version.

If release version is not set, default to a development build whose version string will be `999.dev0`.


## Development ##

Build and install for development:

  $ python setup.py develop

Uninstall:

  $ pip uninstall nni

Remove generated files: (use "--all" to remove toolchain and built wheel)

  $ python setup.py clean [--all]

Build TypeScript modules without install:

  $ python setup.py build_ts


## Release ##

Build wheel package:

  $ NNI_RELEASE=2.0 python setup.py build_ts
  $ NNI_RELEASE=2.0 python setup.py bdist_wheel -p manylinux1_x86_64

Where "2.0" is version string and "manylinux1_x86_64" is platform.
The platform may also be "macosx_10_9_x86_64" or "win_amd64".

`build_ts` must be manually invoked before `bdist_wheel`,
or setuptools cannot locate JS files which should be packed into wheel.
"""

from distutils.cmd import Command
from distutils.command.build import build
from distutils.command.clean import clean
import glob
import os
import shutil
import sys

import setuptools
from setuptools.command.develop import develop

import setup_ts


dependencies = [
    'astor',
    'hyperopt==0.1.2',
    'json_tricks',
    'netifaces',
    'psutil',
    'ruamel.yaml',
    'requests',
    'responses',
    'schema',
    'PythonWebHDFS',
    'colorama',
    'scikit-learn>=0.23.2',
    'pkginfo',
    'websockets',
    'filelock',
    'prettytable',
    'dataclasses ; python_version < "3.7"',
    'numpy < 1.19.4 ; sys_platform == "win32"',
    'numpy < 1.20 ; sys_platform != "win32" and python_version < "3.7"',
    'numpy ; sys.platform != "win32" and python_version >= "3.7"',
    'scipy < 1.6 ; python_version < "3.7"',
    'scipy ; python_version >= "3.7"',
]

release = os.environ.get('NNI_RELEASE')

def _setup():
    setuptools.setup(
        name = 'nni',
        version = release or '999.dev0',
        description = 'Neural Network Intelligence project',
        long_description = open('README.md', encoding='utf-8').read(),
        long_description_content_type = 'text/markdown',
        url = 'https://github.com/Microsoft/nni',
        author = 'Microsoft NNI Team',
        author_email = 'nni@microsoft.com',
        license = 'MIT',
        classifiers = [
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3 :: Only',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],

        packages = _find_python_packages(),
        package_data = {
            'nni': _find_requirements_txt() + _find_default_config(),  # setuptools issue #1806
            'nni_node': _find_node_files()  # note: this does not work before building
        },

        python_requires = '>=3.6',
        install_requires = dependencies,
        extras_require = {
            'SMAC': ['ConfigSpaceNNI', 'smac4nni'],
            'BOHB': ['ConfigSpace==0.4.7', 'statsmodels==0.12.0'],
            'PPOTuner': ['enum34', 'gym']
        },
        setup_requires = ['requests'],

        entry_points = {
            'console_scripts' : [
                'nnictl = nni.tools.nnictl.nnictl:parse_args'
            ]
        },

        cmdclass = {
            'build': Build,
            'build_ts': BuildTs,
            'clean': Clean,
            'develop': Develop,
        }
    )


def _find_python_packages():
    packages = []
    for dirpath, dirnames, filenames in os.walk('nni'):
        if '/__pycache__' not in dirpath and '/.mypy_cache' not in dirpath and '/default_config' not in dirpath:
            packages.append(dirpath.replace('/', '.'))
    return sorted(packages) + ['nni_node']

def _find_requirements_txt():
    requirement_files = []
    for dirpath, dirnames, filenames in os.walk('nni'):
        if 'requirements.txt' in filenames:
            requirement_files.append(os.path.join(dirpath[len('nni/'):], 'requirements.txt'))
    return requirement_files

def _find_default_config():
    return ['runtime/default_config/' + name for name in os.listdir('nni/runtime/default_config')]

def _find_node_files():
    if not os.path.exists('nni_node'):
        if release and 'build_ts' not in sys.argv and 'clean' not in sys.argv:
            sys.exit('ERROR: To build a release version, run "python setup.py build_ts" first')
        return []
    files = []
    for dirpath, dirnames, filenames in os.walk('nni_node'):
        for filename in filenames:
            files.append(os.path.join(dirpath[len('nni_node/'):], filename))
    if '__init__.py' in files:
        files.remove('__init__.py')
    return sorted(files)

def _using_conda_or_virtual_environment():
    return sys.prefix != sys.base_prefix or os.path.isdir(os.path.join(sys.prefix, 'conda-meta'))

class BuildTs(Command):
    description = 'build TypeScript modules'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        setup_ts.build(release)

class Build(build):
    def run(self):
        if not release:
            sys.exit('Please set environment variable "NNI_RELEASE=<release_version>"')
        if os.path.islink('nni_node/main.js'):
            sys.exit('A development build already exists. Please uninstall NNI and run "python3 setup.py clean --all".')
        open('nni/version.py', 'w').write(f"__version__ = '{release}'")
        super().run()

class Develop(develop):
    user_options = develop.user_options + [
        ('no-user', None, 'Prevent automatically adding "--user"'),
        ('skip-ts', None, 'Prevent building TypeScript modules')
    ]

    boolean_options = develop.boolean_options + ['no-user', 'skip-ts']

    def initialize_options(self):
        super().initialize_options()
        self.no_user = None
        self.skip_ts = None

    def finalize_options(self):
        # if `--user` or `--no-user` is explicitly set, do nothing
        # otherwise activate `--user` if using system python
        if not self.user and not self.no_user:
            self.user = not _using_conda_or_virtual_environment()
        super().finalize_options()

    def run(self):
        open('nni/version.py', 'w').write("__version__ = '999.dev0'")
        if not self.skip_ts:
            setup_ts.build(release=None)
        super().run()

class Clean(clean):
    def finalize_options(self):
        self._all = self.all
        self.all = True  # always use `clean --all`
        super().finalize_options()

    def run(self):
        super().run()
        setup_ts.clean(self._all)
        _clean_temp_files()
        shutil.rmtree('nni.egg-info', ignore_errors=True)
        if self._all:
            shutil.rmtree('dist', ignore_errors=True)


def _clean_temp_files():
    for pattern in _temp_files:
        for path in glob.glob(pattern):
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)

_temp_files = [
    # unit test
    'test/model_path/',
    'test/temp.json',
    'test/ut/sdk/*.pth',
    'test/ut/tools/annotation/_generated/'
]


if __name__ == '__main__':
    _setup()
