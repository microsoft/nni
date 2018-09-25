import setuptools

setuptools.setup(
    # NNI Training Service(nnits) package
    name = 'nnits-tool',
    version = '0.0.1',
    packages = setuptools.find_packages(),

    python_requires = '>=3.5',
    install_requires = [
        'requests',
        'psutil'
    ],

    author = 'Microsoft NNI Team',
    author_email = 'nni@microsoft.com',
    description = 'NNI Training Service Tool for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni'
)