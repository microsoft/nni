# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import shutil
import json

from . import code_generator
from . import search_space_generator
from . import specific_code_generator


__all__ = ['generate_search_space', 'expand_annotations']

slash = '/'
if sys.platform == "win32":
    slash = '\\'

def generate_search_space(code_dir):
    """Generate search space from Python source code.
    Return a serializable search space object.
    code_dir: directory path of source files (str)
    """
    code_dir = str(code_dir)
    search_space = {}

    if code_dir.endswith(slash):
        code_dir = code_dir[:-1]

    for subdir, _, files in os.walk(code_dir):
        # generate module name from path
        if subdir == code_dir:
            package = ''
        else:
            assert subdir.startswith(code_dir + slash), subdir
            prefix_len = len(code_dir) + 1
            package = subdir[prefix_len:].replace(slash, '.') + '.'

        for file_name in files:
            if file_name.endswith('.py'):
                path = os.path.join(subdir, file_name)
                module = package + file_name[:-3]
                search_space.update(_generate_file_search_space(path, module))

    return search_space

def _generate_file_search_space(path, module):
    with open(path) as src:
        try:
            search_space, code = search_space_generator.generate(module, src.read())
        except Exception as exc:  # pylint: disable=broad-except
            if exc.args:
                raise RuntimeError(path + ' ' + '\n'.join(exc.args))
            else:
                raise RuntimeError('Failed to generate search space for %s: %r' % (path, exc))
    with open(path, 'w') as dst:
        dst.write(code)
    return search_space


def expand_annotations(src_dir, dst_dir, exp_id='', trial_id='', nas_mode=None):
    """Expand annotations in user code.
    Return dst_dir if annotation detected; return src_dir if not.
    src_dir: directory path of user code (str)
    dst_dir: directory to place generated files (str)
    nas_mode: the mode of NAS given that NAS interface is used
    """
    src_dir, dst_dir = str(src_dir), str(dst_dir)

    if src_dir[-1] == slash:
        src_dir = src_dir[:-1]

    if dst_dir[-1] == slash:
        dst_dir = dst_dir[:-1]

    annotated = False

    for src_subdir, dirs, files in os.walk(src_dir):
        assert src_subdir.startswith(src_dir)
        dst_subdir = src_subdir.replace(src_dir, dst_dir, 1)
        os.makedirs(dst_subdir, exist_ok=True)

        # generate module name from path
        if src_subdir == src_dir:
            package = ''
        else:
            assert src_subdir.startswith(src_dir + slash), src_subdir
            prefix_len = len(src_dir) + 1
            package = src_subdir[prefix_len:].replace(slash, '.') + '.'

        for file_name in files:
            src_path = os.path.join(src_subdir, file_name)
            dst_path = os.path.join(dst_subdir, file_name)
            if file_name.endswith('.py'):
                if trial_id == '':
                    annotated |= _expand_file_annotations(src_path, dst_path, nas_mode)
                else:
                    module = package + file_name[:-3]
                    annotated |= _generate_specific_file(src_path, dst_path, exp_id, trial_id, module)
            else:
                shutil.copyfile(src_path, dst_path)

        for dir_name in dirs:
            os.makedirs(os.path.join(dst_subdir, dir_name), exist_ok=True)

    return dst_dir if annotated else src_dir

def _expand_file_annotations(src_path, dst_path, nas_mode):
    with open(src_path) as src, open(dst_path, 'w') as dst:
        try:
            annotated_code = code_generator.parse(src.read(), nas_mode)
            if annotated_code is None:
                shutil.copyfile(src_path, dst_path)
                return False
            dst.write(annotated_code)
            return True

        except Exception as exc:  # pylint: disable=broad-except
            if exc.args:
                raise RuntimeError(src_path + ' ' + '\n'.join(str(arg) for arg in exc.args))
            else:
                raise RuntimeError('Failed to expand annotations for %s: %r' % (src_path, exc))

def _generate_specific_file(src_path, dst_path, exp_id, trial_id, module):
    with open(src_path) as src, open(dst_path, 'w') as dst:
        try:
            with open(os.path.expanduser('~/nni-experiments/%s/trials/%s/parameter.cfg'%(exp_id, trial_id))) as fd:
                para_cfg = json.load(fd)
            annotated_code = specific_code_generator.parse(src.read(), para_cfg["parameters"], module)
            if annotated_code is None:
                shutil.copyfile(src_path, dst_path)
                return False
            dst.write(annotated_code)
            return True

        except Exception as exc:  # pylint: disable=broad-except
            if exc.args:
                raise RuntimeError(src_path + ' ' + '\n'.join(str(arg) for arg in exc.args))
            else:
                raise RuntimeError('Failed to expand annotations for %s: %r' % (src_path, exc))
