from setuptools import setup, find_packages

setup(
    name="nnitest",
    version="0.0.1",
    author = 'Microsoft NNI team',
    author_email = 'nni@microsoft.com',
    description = 'Neural Network Intelligence package',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni',
    packages=find_packages('nnitest'),
    long_description="",
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent"
    ],
)
