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
    'websockets',
    'prettytable'
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
            'nni': ['**/requirements.txt'],
            'nni_node': _find_node_files()  # note: this does not work before building
        },

        python_requires = '>=3.6',
        install_requires = dependencies,
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
        if '/__pycache__' not in dirpath:
            packages.append(dirpath.replace('/', '.'))
    return sorted(packages) + ['nni_node']

def _find_node_files():
    if not os.path.exists('nni_node'):
        if release and 'built_ts' not in sys.argv:
            sys.exit('ERROR: To build a release version, run "python setup.py built_ts" first')
        return []
    files = []
    for dirpath, dirnames, filenames in os.walk('nni_node'):
        for filename in filenames:
            files.append((dirpath + '/' + filename)[len('nni_node/'):])
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
        assert release, 'Please set environment variable "NNI_RELEASE=<release_version>"'
        assert os.path.isfile('nni_node/main.js'), 'Please run "build_ts" before "build"'
        assert not os.path.islink('nni_node/main.js'), 'This is a development build'
        super().run()

class Develop(develop):
    user_options = develop.user_options + [
        ('no-user', None, 'Prevent automatically adding "--user"')
    ]

    boolean_options = develop.boolean_options + ['no-user']

    def initialize_options(self):
        super().initialize_options()
        self.no_user = None

    def finalize_options(self):
        # if `--user` or `--no-user` is explicitly set, do nothing
        # otherwise activate `--user` if using system python
        if not self.user and not self.no_user:
            self.user = not _using_conda_or_virtual_environment()
        super().finalize_options()

    def run(self):
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


_setup()
