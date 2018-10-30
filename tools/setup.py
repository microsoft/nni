import setuptools

setuptools.setup(
    name = 'nnictl',
    version = '0.2.0',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.5',
    install_requires = [
        'requests',
        'pyyaml',
        'psutil',
        'astor',
        'schema',
        'pyhdfs',
        'paramiko'
    ],

    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'NNI control for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',
)
