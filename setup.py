# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script for installation and distribution.

You can use environment variable `NNI_RELEASE` to set release version.

If release version is not set, default to a development build whose version string will be `999.0.0`.


## Development ##

Build and install for development:

  $ python setup.py develop

Uninstall:

  $ pip uninstall nni

Remove generated files:

  $ python setup.py clean

Build TypeScript modules without install:

  $ python setup.py build_ts

The version string of development build will be "999.0.0".


## Release ##

Build wheel package:

  $ python setup.py clean
  $ NNI_RELEASE=2.0 python setup.py build
  $ NNI_RELEASE=2.0 python setup.py bdist_wheel -p <platform>

Where <platform> should be one of:

  - manylinux1_x86_64
  - macosx_10_9_x86_64
  - win_amd64

Remember to invoke `build` explicitly before `bdist_wheel`,
or it cannot find out JavaScript files.
"""

from distutils.cmd import Command
from distutils.command.build import build
from distutils.command.clean import clean
import os
import shutil

import setuptools
from setuptools.command.develop import develop

import build_ts


release = os.environ.get('NNI_RELEASE')

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
    'websockets'
]


def _setup():
    setuptools.setup(
        name = 'nni',
        version = release or '999.0.0',
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
                'nnictl = nni.nni_cmd.nnictl:parse_args'
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
    files = []
    for dirpath, dirnames, filenames in os.walk('nni_node'):
        for filename in filenames:
            files.append((dirpath + '/' + filename)[len('nni_node/'):])
    files.remove('__init__.py')
    return sorted(files)


class BuildTs(Command):
    description = 'build TypeScript modules'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build_ts.build(release)

class Build(build):
    def run(self):
        assert release, 'Please set environment variable `NNI_RELEASE=<release_version>`'
        build_ts.build(release)
        super().run()

class Develop(develop):
    def finalize_options(self):
        self.user = True  # always use `develop --user`
        super().finalize_options()

    def run(self):
        build_ts.build(release=None)
        super().run()

class Clean(clean):
    def finalize_options(self):
        self.all = True  # always use `clean --all`
        super().finalize_options()

    def run(self):
        super().run()
        build_ts.clean()
        shutil.rmtree('nni.egg-info', ignore_errors=True)


_setup()
