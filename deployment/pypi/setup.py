import setuptools
from os import walk, path

data_files = [('bin', ['node-linux-x64/bin/node'])]
for (dirpath, dirnames, filenames) in walk('./nni_pkg'):
    files = [path.normpath(path.join(dirpath, filename)) for filename in filenames]
    data_files.append((path.normpath(dirpath), files))

with open('../../README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'nni-pkg',
    version = '0.3.0',
    author = 'Microsoft NNI team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence package',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',
    packages = setuptools.find_packages(),
    python_requires = '>=3.5',
    install_requires = [
        'nni',
        'schema',
        'pyyaml',
        'psutil',
        'requests',
        'paramiko'
    ],
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux'
    ],
    data_files = data_files,
    entry_points = {
        'console_scripts' : [
            'nnictl = nnicmd.nnictl:parse_args'
        ]
    }
)