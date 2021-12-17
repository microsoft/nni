# fix setuptools.distutils import in torch
import os
from torch import utils

file_name = os.path.join(os.path.dirname(utils.__file__), 'tensorboard/__init__.py')
dummy_file_name = os.path.join(os.path.dirname(file_name), '__dummy_init__.py')
if os.path.exists(file_name):
    with open(file_name, 'r') as fr, open(dummy_file_name, 'w') as fw:
        origin_text = fr.read()
        patched_text = origin_text.replace('from setuptools import distutils', '', 1)
        patched_text = patched_text.replace('LooseVersion = distutils.version.LooseVersion', 'from distutils.version import LooseVersion', 1)
        patched_text = patched_text.replace('del distutils', '', 1)
        fw.write(patched_text)

if os.path.exists(dummy_file_name):
    os.remove(file_name)
    os.rename(dummy_file_name, file_name)
