# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.tools import annotation

import ast
import json
from pathlib import Path
import shutil
import tempfile

import pytest

cwd = Path(__file__).parent
shutil.rmtree(cwd / '_generated', ignore_errors=True)
shutil.copytree(cwd / 'testcase/annotated', cwd / '_generated/annotated')

def test_search_space_generator():
    search_space = annotation.generate_search_space(cwd / '_generated/annotated')
    expected = json.load((cwd / 'testcase/searchspace.json').open())
    assert search_space == expected

def test_code_generator():
    src_dir = cwd / 'testcase/usercode'
    dst_dir = cwd / '_generated/usercode'
    code_dir = annotation.expand_annotations(src_dir, dst_dir, nas_mode='classic_mode')
    assert Path(code_dir) == dst_dir
    expect_dir = cwd / 'testcase/annotated'
    _assert_source_equal(dst_dir, expect_dir, 'dir/simple.py')
    _assert_source_equal(dst_dir, expect_dir, 'mnist.py')
    _assert_source_equal(dst_dir, expect_dir, 'nas.py')
    assert (src_dir / 'nonpy.txt').read_text() == (dst_dir / 'nonpy.txt').read_text()

def test_annotation_detecting():
    src_dir = cwd / 'testcase/usercode/non_annotation'
    code_dir = annotation.expand_annotations(src_dir, tempfile.mkdtemp())
    assert Path(code_dir) == src_dir


def _assert_source_equal(dir1, dir2, file_name):
    ast1 = ast.parse((dir1 / file_name).read_text())
    ast2 = ast.parse((dir2 / file_name).read_text())
    _assert_ast_equal(ast1, ast2)

def _assert_ast_equal(ast1, ast2):
    assert type(ast1) is type(ast2)
    if isinstance(ast1, ast.AST):
        assert sorted(ast1._fields) == sorted(ast2._fields)
        for field_name in ast1._fields:
            field1 = getattr(ast1, field_name)
            field2 = getattr(ast2, field_name)
            _assert_ast_equal(field1, field2)
    elif isinstance(ast1, list):
        assert len(ast1) == len(ast2)
        for item1, item2 in zip(ast1, ast2):
            _assert_ast_equal(item1, item2)
    else:
        assert ast1 == ast2


if __name__ == '__main__':
    pytest.main()
