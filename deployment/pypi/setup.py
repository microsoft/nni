import setuptools
from os import walk, path

data_files = [('bin', ['node-linux-x64/bin/node'])]
for (dirpath, dirnames, filenames) in walk('./nni'):
    files = [path.normpath(path.join(dirpath, filename)) for filename in filenames]
    data_files.append((path.normpath(dirpath), files))

with open('../../README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'nni',
    version = '0.3.4',
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
        'Operating System :: POSIX :: Linux'
    ],
    data_files = data_files,
    entry_points = {
        'console_scripts' : [
            'nnictl = nni_cmd.nnictl:parse_args'
        ]
    }
)