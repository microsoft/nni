import setuptools

setuptools.setup(
    name = 'nnictl',
    version = '0.0.1',
    packages = setuptools.find_packages(),

    python_requires = '>=3.5',
    install_requires = [
        'requests',
        'pyyaml',
        'psutil',
        'astor',
        'schema'
    ],

    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'NNI control for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://msrasrg.visualstudio.com/NeuralNetworkIntelligence',
)
