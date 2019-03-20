import setuptools

setuptools.setup(
    name = 'nni-tool',
    version = 'v0.1-521-gf803acb',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.5',
    install_requires = [
        'requests',
        'pyyaml',
        'psutil',
        'astor',
        'schema',
        'PythonWebHDFS'
    ],

    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'NNI control for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',
    entry_points = {
        'console_scripts' : [
            'nnictl = nni_cmd.nnictl:parse_args'
        ]
    }
)
