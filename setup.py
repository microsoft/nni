# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script for installation and distribution.

You can use environment variable `NNI_RELEASE` to set release version.

If release version is not set, default to a development build whose version string will be `999.dev0`.


## Prepare Environment ##

Install development dependencies:

  $ pip install -U -r dependencies/setup.txt
  $ pip install -r dependencies/develop.txt


## Development ##

Build and install for development:

  $ python setup.py develop

Uninstall:

  $ pip uninstall nni

Remove generated files: (use "--all" to remove built wheel)

  $ python setup.py clean [--all]

Compile TypeScript modules without re-install:

  $ python setup.py build_ts


## Release ##

Build wheel package:

  $ NNI_RELEASE=2.0 python setup.py build_ts
  $ NNI_RELEASE=2.0 python setup.py bdist_wheel -p manylinux1_x86_64

for jupyterlab 2.x package:
  $ JUPYTER_LAB_VERSION=2.3.1 NNI_RELEASE=2.0 python setup.py build_ts
  $ JUPYTER_LAB_VERSION=2.3.1 NNI_RELEASE=2.0 python setup.py bdist_wheel -p manylinux1_x86_64

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

release = os.environ.get('NNI_RELEASE')

#def _get_jupyter_lab_version():
#    try:
#        import jupyterlab
#        return jupyterlab.__version__
#    except ImportError:
#        return '3.x'

#jupyter_lab_major_version = _get_jupyter_lab_version().split('.')[0]

#def check_jupyter_lab_version():
#    environ_version = os.environ.get('JUPYTER_LAB_VERSION')
#
#    jupyter_lab_version = _get_jupyter_lab_version()
#
#    if environ_version:
#        if jupyter_lab_version.split('.')[0] != environ_version.split('.')[0]:
#            sys.exit(f'ERROR: To build a jupyter lab extension, run "JUPYTER_LAB_VERSION={jupyter_lab_version}", current: {environ_version} ')
#    elif jupyter_lab_version.split('.')[0] != '3':
#        sys.exit(f'ERROR: To build a jupyter lab extension, run "JUPYTER_LAB_VERSION={jupyter_lab_version}" first for nondefault version(3.x)')

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
            'nni_assets': _find_asset_files(),
            'nni_node': _find_node_files()  # note: this does not work before building
        },

        data_files = _get_data_files(),

        python_requires = '>=3.7',
        install_requires = _read_requirements_txt('dependencies/required.txt'),
        extras_require = {
            'Anneal': _read_requirements_txt('dependencies/required_extra.txt', 'Anneal'),
            'SMAC': _read_requirements_txt('dependencies/required_extra.txt', 'SMAC'),
            'BOHB': _read_requirements_txt('dependencies/required_extra.txt', 'BOHB'),
            'PPOTuner': _read_requirements_txt('dependencies/required_extra.txt', 'PPOTuner'),
            'DNGO': _read_requirements_txt('dependencies/required_extra.txt', 'DNGO'),
            'all': _read_requirements_txt('dependencies/required_extra.txt'),
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

def _get_data_files():
    data_files = []
#    if jupyter_lab_major_version == '2':
#        extension_file = glob.glob("nni_node/jupyter-extension/extensions/nni-jupyter-extension*.tgz")
#        data_files = [('share/jupyter/lab/extensions', extension_file)]
    return data_files

def _find_python_packages():
    packages = []
    for dirpath, dirnames, filenames in os.walk('nni'):
        if '/__pycache__' not in dirpath and '/.mypy_cache' not in dirpath and '/default_config' not in dirpath:
            packages.append(dirpath.replace('/', '.'))
    return sorted(packages) + ['nni_assets', 'nni_node']

def _find_requirements_txt():
    requirement_files = []
    for dirpath, dirnames, filenames in os.walk('nni'):
        if 'requirements.txt' in filenames:
            requirement_files.append(os.path.join(dirpath[len('nni/'):], 'requirements.txt'))
    return requirement_files

def _find_default_config():
    return ['runtime/default_config/' + name for name in os.listdir('nni/runtime/default_config')]

def _find_asset_files():
    files = []
    for dirpath, dirnames, filenames in os.walk('nni_assets'):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.py':
                files.append(os.path.join(dirpath[len('nni_assets/'):], filename))
    return sorted(files)

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

def _read_requirements_txt(file_path, section=None):
    with open(file_path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]  # remove whitespaces and empty lines
    if section is None:
        return [line for line in lines if not line.startswith('#')]
    selected_lines = []
    started = False
    for line in lines:
        if started:
            if line.startswith('#'):
                return selected_lines
            else:
                selected_lines.append(line)
        elif line.startswith('# ' + section):
            started = True
    return selected_lines

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
        #check_jupyter_lab_version()
        setup_ts.build(release)

class Build(build):
    def run(self):
        if not release:
            sys.exit('Please set environment variable "NNI_RELEASE=<release_version>"')

        #check_jupyter_lab_version()

        if os.path.islink('nni_node/main.js'):
            sys.exit('A development build already exists. Please uninstall NNI and run "python3 setup.py clean".')
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
        setup_ts.clean()
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

    # example
    'nni_assets/**/data/',
]


if __name__ == '__main__':
    _setup()
