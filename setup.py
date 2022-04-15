"""
use the following command to dev install

python3 setup.py develop

"""
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

rootdir = os.path.dirname(os.path.realpath(__file__))

version = "0.0.1"

ext_modules = []

setup(
    name='sparta',
    version=version,
    description='Deployment tool',
    author='MSRA',
    author_email="Ningxin.Zheng@microsoft.com",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
