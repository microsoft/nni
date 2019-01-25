import setuptools
import platform
from os import walk, path

os_type = platform.system()
if os_type == 'Linux':
    os_name = 'POSIX :: Linux'
elif os_type == 'Darwin':
    os_name = 'MacOS'
else:
    raise NotImplementedError('current platform {} not supported'.format(os_type))

data_files = [('bin', ['node-{}-x64/bin/node'.format(os_type.lower())])]
for (dirpath, dirnames, filenames) in walk('./nni'):
    files = [path.normpath(path.join(dirpath, filename)) for filename in filenames]
    data_files.append((path.normpath(dirpath), files))

with open('../../README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'nni',
    version = '999.0.0-developing',
    author = 'Microsoft NNI team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence package',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',
    packages = setuptools.find_packages('../../tools'),
    package_dir = {
        'nni_annotation': '../../tools/nni_annotation',
        'nni_cmd': '../../tools/nni_cmd',
        'nni_trial_tool':'../../tools/nni_trial_tool'
    },
    python_requires = '>=3.5',
    install_requires = [
        'nni-sdk',
        'schema',
        'pyyaml',
        'psutil',
        'requests',
        'astor',
        'pyhdfs'
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
    }
)