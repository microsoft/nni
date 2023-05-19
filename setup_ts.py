# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script for building TypeScript modules.
This script is called by `setup.py` and common users should avoid using this directly.

It compiles TypeScript source files in `ts` directory,
and copies (or links) JavaScript output as well as dependencies to `nni_node`.

You can set environment `GLOBAL_TOOLCHAIN=1` to use global node and npm, if you know what you are doing.
"""

from io import BytesIO
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import traceback
from zipfile import ZipFile


node_version = 'v18.15.0'

def _print(*args, color='cyan'):
    color_code = {'yellow': 33, 'cyan': 36}[color]
    if sys.platform == 'win32':
        print(*args, flush=True)
    else:
        print(f'\033[1;{color_code}m#', *args, '\033[0m', flush=True)

#def _get_jupyter_lab_version():
#    try:
#        import jupyterlab
#        return jupyterlab.__version__
#    except ImportError:
#        return '3.x'

def _get_glibc_minor_version():  # type: () -> int | None
    try:
        from pip._internal.utils.glibc import glibc_version_string
        glibc_version = glibc_version_string()
        if glibc_version is None:
            return None
        glibc_major, glibc_minor = map(int, glibc_version.split('.'))
        if glibc_major < 2:
            raise RuntimeError('Unsupported glibc version: ' + glibc_version)
        elif glibc_major == 2:
            _print(f'Detected glibc version: {glibc_version}')
            return glibc_minor
        return None
    except ImportError:
        _print('Unsupported pip version. Assuming glibc not found.', color='yellow')
        return None

def _get_node_downloader():
    if platform.machine() == 'x86_64':
        glibc_minor = _get_glibc_minor_version()
        if glibc_minor is None or glibc_minor >= 28:
            _arch = 'x64'
        elif glibc_minor >= 27:
            _print('Detected deprecated glibc version < 2.28. Please upgrade as soon as possible.', color='yellow')
            _arch = 'glibc-2.27'
        else:
            _print('glibc version is too low. We will try to use the node version compiled with glibc 2.23, '
                   'but it might not work.', color='yellow')
            _print('Please check your glibc version by running `ldd --version`, and upgrade it if necessary.',
                   color='yellow')
            _arch = 'glibc-2.23'
    else:
        _arch = platform.machine()

    if _arch.startswith('glibc'):
        node_legacy_version = 'v18.12.1'  # We might not upgrade node version for legacy builds every time.
        node_spec = f'node-{node_legacy_version}-{sys.platform}-x64'
        node_download_url = f'https://nni.blob.core.windows.net/cache/toolchain/node-{node_legacy_version}-{sys.platform}-{_arch}.tar.gz'
        node_extractor = lambda data: tarfile.open(fileobj=BytesIO(data), mode='r:gz')
    else:
        node_spec = f'node-{node_version}-{sys.platform}-' + _arch
        node_download_url = f'https://nodejs.org/dist/{node_version}/{node_spec}.tar.xz'
        node_extractor = lambda data: tarfile.open(fileobj=BytesIO(data), mode='r:xz')
    return node_download_url, node_spec, node_extractor

#jupyter_lab_major_version = _get_jupyter_lab_version().split('.')[0]

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
    #update_package()
    compile_ts(release)
    if release or sys.platform == 'win32':
        copy_nni_node(release)
    else:
        symlink_nni_node()
    #restore_package()

def clean():
    """
    Remove TypeScript-related intermediate files.
    Python intermediate files are not touched here.
    """
    shutil.rmtree('nni_node', ignore_errors=True)
    shutil.rmtree('toolchain', ignore_errors=True)

    for file_or_dir in generated_files:
        path = Path(file_or_dir)
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


if sys.platform == 'linux' or sys.platform == 'darwin':
    node_executable = 'node'
    node_download_url, node_spec, node_extractor = _get_node_downloader()
    node_executable_in_tarball = 'bin/node'

    npm_executable = 'bin/npm'

    path_env_separator = ':'

elif sys.platform == 'win32':
    node_executable = 'node.exe'
    node_spec = f'node-{node_version}-win-x64'
    node_download_url = f'https://nodejs.org/dist/{node_version}/{node_spec}.zip'
    node_extractor = lambda data: ZipFile(BytesIO(data))
    node_executable_in_tarball = 'node.exe'

    npm_executable = 'npm.cmd'

    path_env_separator = ';'

else:
    raise RuntimeError('Unsupported system')


def download_toolchain():
    """
    Download and extract node.
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

#def update_package():
#    if jupyter_lab_major_version == '2':
#        package_json = json.load(open('ts/jupyter_extension/package.json'))
#        json.dump(package_json, open('ts/jupyter_extension/.package_default.json', 'w'), indent=2)
#
#        package_json['scripts']['build'] = 'tsc && jupyter labextension link .'
#        package_json['dependencies']['@jupyterlab/application'] = '^2.3.0'
#        package_json['dependencies']['@jupyterlab/launcher'] = '^2.3.0'
#
#        package_json['jupyterlab']['outputDir'] = 'build'
#        json.dump(package_json, open('ts/jupyter_extension/package.json', 'w'), indent=2)
#        print(f'updated package.json with {json.dumps(package_json, indent=2)}')

#def restore_package():
#    if jupyter_lab_major_version == '2':
#        package_json = json.load(open('ts/jupyter_extension/.package_default.json'))
#        print(f'stored package.json with {json.dumps(package_json, indent=2)}')
#        json.dump(package_json, open('ts/jupyter_extension/package.json', 'w'), indent=2)
#        os.remove('ts/jupyter_extension/.package_default.json')

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


def compile_ts(release):
    """
    Use npm to download dependencies and compile TypeScript code.
    """
    _print('Building NNI manager')
    _npm('ts/nni_manager', 'install')
    _npm('ts/nni_manager', 'run', 'build')
    # todo: I don't think these should be here
    shutil.rmtree('ts/nni_manager/dist/config', ignore_errors=True)
    shutil.copytree('ts/nni_manager/config', 'ts/nni_manager/dist/config')

    _print('Building web UI')
    _npm('ts/webui', 'install')
    if release:
        _npm('ts/webui', 'run', 'release')
    else:
        _npm('ts/webui', 'run', 'build')

    #_print('Building JupyterLab extension')
    #try:
    #    _yarn('ts/jupyter_extension')
    #    _yarn('ts/jupyter_extension', 'build')
    #except Exception:
    #    if release:
    #        raise
    #    _print('Failed to build JupyterLab extension, skip for develop mode', color='yellow')
    #    _print(traceback.format_exc(), color='yellow')


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

    #if jupyter_lab_major_version == '2':
    #    _symlink('ts/jupyter_extension/build', 'nni_node/jupyter-extension')
    #    _symlink(os.path.join(sys.exec_prefix, 'share/jupyter/lab/extensions'), 'nni_node/jupyter-extension/extensions')
    #elif Path('ts/jupyter_extension/dist').exists():
    #    _symlink('ts/jupyter_extension/dist', 'nni_node/jupyter-extension')


def copy_nni_node(version):
    """
    Copy compiled JS files to nni_node.
    This is meant for building release package, so you need to provide version string.
    The version will written to `package.json` in nni_node directory,
    while `package.json` in ts directory will be left unchanged.
    """
    _print('Copying files')

    if sys.version_info >= (3, 8):
        shutil.copytree('ts/nni_manager/dist', 'nni_node', dirs_exist_ok=True)
    else:
        for item in os.listdir('ts/nni_manager/dist'):
            subsrc = os.path.join('ts/nni_manager/dist', item)
            subdst = os.path.join('nni_node', item)
            if os.path.isdir(subsrc):
                shutil.copytree(subsrc, subdst)
            else:
                shutil.copy2(subsrc, subdst)
    shutil.copyfile('ts/nni_manager/package-lock.json', 'nni_node/package-lock.lock')
    Path('nni_node/nni_manager.tsbuildinfo').unlink()

    package_json = json.load(open('ts/nni_manager/package.json'))
    if version:
        while len(version.split('.')) < 3:  # node.js semver requires at least three parts
            version = version + '.0'
        package_json['version'] = version
    json.dump(package_json, open('nni_node/package.json', 'w'), indent=2)

    if sys.platform == 'win32':
        # On Windows, manually install node-gyp for sqlite3.
        _npm('ts/nni_manager', 'install', '--global', 'node-gyp')

    # reinstall without development dependencies
    prod_path = Path('nni_node').resolve()
    _npm(str(prod_path), 'install', '--omit', 'dev')

    shutil.copytree('ts/webui/build', 'nni_node/static')

    #if jupyter_lab_major_version == '2':
    #    shutil.copytree('ts/jupyter_extension/build', 'nni_node/jupyter-extension/build')
    #    shutil.copytree(os.path.join(sys.exec_prefix, 'share/jupyter/lab/extensions'), 'nni_node/jupyter-extension/extensions')
    #elif version or Path('ts/jupyter_extension/dist').exists():
    #    shutil.copytree('ts/jupyter_extension/dist', 'nni_node/jupyter-extension')


_npm_env = dict(os.environ)
# `Path('nni_node').resolve()` does not work on Windows if the directory not exists
_npm_env['PATH'] = str(Path().resolve() / 'nni_node') + path_env_separator + os.environ['PATH']
_npm_path = Path().resolve() / 'toolchain/node' / npm_executable

def _npm(path, *args):
    _print('npm ' + ' '.join(args) + f' (path: {path})')
    if os.environ.get('GLOBAL_TOOLCHAIN'):
        subprocess.run(['npm', *args], cwd=path, check=True)
    else:
        subprocess.run([str(_npm_path), *args], cwd=path, check=True, env=_npm_env)


def _symlink(target_file, link_location):
    target = Path(target_file)
    link = Path(link_location)
    relative = os.path.relpath(target, link.parent)
    link.symlink_to(relative, target.is_dir())


generated_files = [
    'ts/nni_manager/dist',
    'ts/nni_manager/node_modules',
    'ts/webui/build',
    'ts/webui/node_modules',

    # unit test
    'ts/nni_manager/.nyc_output',
    'ts/nni_manager/coverage',
    'ts/nni_manager/exp_profile.json',
    'ts/nni_manager/metrics.json',
    'ts/nni_manager/trial_jobs.json',
]
