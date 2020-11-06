# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script for building TypeScript modules.
This script is called by `setup.py` and common users should avoid using this directly.

It compiles TypeScript source files in `ts` directory,
and copies (or links) JavaScript output as well as dependencies to `nni_node`.

You can set environment `GLOBAL_TOOLCHAIN=1` to use global node and yarn, if you know what you are doing.
"""

from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from zipfile import ZipFile


node_version = 'v10.23.0'
yarn_version = 'v1.22.10'


def build(release):
    """
    Compile TypeScript modules and copy or symlink to nni_node directory.

    `release` is the version number without leading letter "v".

    If `release` is None or empty, this is a development build and uses symlinks on Linux/macOS;
    otherwise this is a release build and copies files instead.
    On Windows it always copies files because creating symlink requires extra privilege.
    """
    if release or not os.environ.get('GLOBAL_TOOLCHAIN'):
        download_toolchain()
    prepare_nni_node()
    compile_ts()
    if release or sys.platform == 'win32':
        copy_nni_node(release)
    else:
        symlink_nni_node()

def clean(clean_all=False):
    """
    Remove TypeScript-related intermediate files.
    Python intermediate files are not touched here.
    """
    shutil.rmtree('nni_node', ignore_errors=True)

    for file_or_dir in generated_files:
        path = Path(file_or_dir)
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    if clean_all:
        shutil.rmtree('toolchain', ignore_errors=True)


if sys.platform == 'linux' or sys.platform == 'darwin':
    node_executable = 'node'
    node_spec = f'node-{node_version}-{sys.platform}-x64'
    node_download_url = f'https://nodejs.org/dist/{node_version}/{node_spec}.tar.xz'
    node_extractor = lambda data: tarfile.open(fileobj=BytesIO(data), mode='r:xz')
    node_executable_in_tarball = 'bin/node'

    yarn_executable = 'yarn'
    yarn_download_url = f'https://github.com/yarnpkg/yarn/releases/download/{yarn_version}/yarn-{yarn_version}.tar.gz'

    path_env_seperator = ':'

elif sys.platform == 'win32':
    node_executable = 'node.exe'
    node_spec = f'node-{node_version}-win-x64'
    node_download_url = f'https://nodejs.org/dist/{node_version}/{node_spec}.zip'
    node_extractor = lambda data: ZipFile(BytesIO(data))
    node_executable_in_tarball = 'node.exe'

    yarn_executable = 'yarn.cmd'
    yarn_download_url = f'https://github.com/yarnpkg/yarn/releases/download/{yarn_version}/yarn-{yarn_version}.tar.gz'

    path_env_seperator = ';'

else:
    raise RuntimeError('Unsupported system')


def download_toolchain():
    """
    Download and extract node and yarn.
    """
    if Path('toolchain/node', node_executable_in_tarball).is_file():
        return

    Path('toolchain').mkdir(exist_ok=True)
    import requests  # place it here so setup.py can install it before importing

    _print(f'Downloading node.js from {node_download_url}')
    resp = requests.get(node_download_url)
    resp.raise_for_status()
    _print('Extracting node.js')
    tarball = node_extractor(resp.content)
    tarball.extractall('toolchain')
    shutil.rmtree('toolchain/node', ignore_errors=True)
    Path('toolchain', node_spec).rename('toolchain/node')

    _print(f'Downloading yarn from {yarn_download_url}')
    resp = requests.get(yarn_download_url)
    resp.raise_for_status()
    _print('Extracting yarn')
    tarball = tarfile.open(fileobj=BytesIO(resp.content), mode='r:gz')
    tarball.extractall('toolchain')
    shutil.rmtree('toolchain/yarn', ignore_errors=True)
    Path(f'toolchain/yarn-{yarn_version}').rename('toolchain/yarn')


def prepare_nni_node():
    """
    Create clean nni_node diretory, then copy node runtime to it.
    """
    shutil.rmtree('nni_node', ignore_errors=True)
    Path('nni_node').mkdir()

    Path('nni_node/__init__.py').write_text('"""NNI node.js modules."""\n')

    node_src = Path('toolchain/node', node_executable_in_tarball)
    node_dst = Path('nni_node', node_executable)
    shutil.copy(node_src, node_dst)


def compile_ts():
    """
    Use yarn to download dependencies and compile TypeScript code.
    """
    _print('Building NNI manager')
    _yarn('ts/nni_manager')
    _yarn('ts/nni_manager', 'build')
    # todo: I don't think these should be here
    shutil.rmtree('ts/nni_manager/dist/config', ignore_errors=True)
    shutil.copytree('ts/nni_manager/config', 'ts/nni_manager/dist/config')

    _print('Building web UI')
    _yarn('ts/webui')
    _yarn('ts/webui', 'build')

    _print('Building NAS UI')
    _yarn('ts/nasui')
    _yarn('ts/nasui', 'build')


def symlink_nni_node():
    """
    Create symlinks to compiled JS files.
    If you manually modify and compile TS source files you don't need to install again.
    """
    _print('Creating symlinks')

    for path in Path('ts/nni_manager/dist').iterdir():
        _symlink(path, Path('nni_node', path.name))
    _symlink('ts/nni_manager/package.json', 'nni_node/package.json')
    _symlink('ts/nni_manager/node_modules', 'nni_node/node_modules')

    _symlink('ts/webui/build', 'nni_node/static')

    Path('nni_node/nasui').mkdir(exist_ok=True)
    _symlink('ts/nasui/build', 'nni_node/nasui/build')
    _symlink('ts/nasui/server.js', 'nni_node/nasui/server.js')


def copy_nni_node(version):
    """
    Copy compiled JS files to nni_node.
    This is meant for building release package, so you need to provide version string.
    The version will written to `package.json` in nni_node directory,
    while `package.json` in ts directory will be left unchanged.
    """
    _print('Copying files')

    # copytree(..., dirs_exist_ok=True) is not supported by Python 3.6
    for path in Path('ts/nni_manager/dist').iterdir():
        if path.is_file():
            shutil.copyfile(path, Path('nni_node', path.name))
        else:
            shutil.copytree(path, Path('nni_node', path.name))

    package_json = json.load(open('ts/nni_manager/package.json'))
    if version:
        while len(version.split('.')) < 3:  # node.js semver requires at least three parts
            version = version + '.0'
        package_json['version'] = version
    json.dump(package_json, open('nni_node/package.json', 'w'), indent=2)

    _yarn('ts/nni_manager', '--prod', '--cwd', str(Path('nni_node').resolve()))

    shutil.copytree('ts/webui/build', 'nni_node/static')

    Path('nni_node/nasui').mkdir(exist_ok=True)
    shutil.copytree('ts/nasui/build', 'nni_node/nasui/build')
    shutil.copyfile('ts/nasui/server.js', 'nni_node/nasui/server.js')


_yarn_env = dict(os.environ)
# `Path('nni_node').resolve()` does not work on Windows if the directory not exists
_yarn_env['PATH'] = str(Path().resolve() / 'nni_node') + path_env_seperator + os.environ['PATH']
_yarn_path = Path().resolve() / 'toolchain/yarn/bin' / yarn_executable

def _yarn(path, *args):
    if os.environ.get('GLOBAL_TOOLCHAIN'):
        subprocess.run(['yarn', *args], cwd=path, check=True)
    else:
        subprocess.run([str(_yarn_path), *args], cwd=path, check=True, env=_yarn_env)


def _symlink(target_file, link_location):
    target = Path(target_file)
    link = Path(link_location)
    relative = os.path.relpath(target, link.parent)
    link.symlink_to(relative, target.is_dir())


def _print(*args):
    if sys.platform == 'win32':
        print(*args)
    else:
        print('\033[1;36m#', *args, '\033[0m')


generated_files = [
    'ts/nni_manager/dist',
    'ts/nni_manager/node_modules',
    'ts/webui/build',
    'ts/webui/node_modules',
    'ts/nasui/build',
    'ts/nasui/node_modules',

    # unit test
    'ts/nni_manager/exp_profile.json',
    'ts/nni_manager/metrics.json',
    'ts/nni_manager/trial_jobs.json',
]
