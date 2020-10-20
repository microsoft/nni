# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file

from .__init__ import *

import sys
import ast
import json
import os
import shutil
import tempfile
from unittest import TestCase, main, skipIf


class AnnotationTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir('nni_annotation')
        if os.path.isdir('_generated'):
            shutil.rmtree('_generated')

    def test_search_space_generator(self):
        shutil.copytree('testcase/annotated', '_generated/annotated')
        search_space = generate_search_space('_generated/annotated')
        with open('testcase/searchspace.json') as f:
            self.assertEqual(search_space, json.load(f))

    @skipIf(sys.version_info.major == 3 and sys.version_info.minor > 7, "skip for python3.8 temporarily")
    def test_code_generator(self):
        code_dir = expand_annotations('testcase/usercode', '_generated/usercode', nas_mode='classic_mode')
        self.assertEqual(code_dir, '_generated/usercode')
        self._assert_source_equal('testcase/annotated/nas.py', '_generated/usercode/nas.py')
        self._assert_source_equal('testcase/annotated/mnist.py', '_generated/usercode/mnist.py')
        self._assert_source_equal('testcase/annotated/dir/simple.py', '_generated/usercode/dir/simple.py')
        with open('testcase/usercode/nonpy.txt') as src, open('_generated/usercode/nonpy.txt') as dst:
            assert src.read() == dst.read()

    def test_annotation_detecting(self):
        dir_ = 'testcase/usercode/non_annotation'
        code_dir = expand_annotations(dir_, tempfile.mkdtemp())
        self.assertEqual(code_dir, dir_)

    def _assert_source_equal(self, src1, src2):
        with open(src1) as f1, open(src2) as f2:
            ast1 = ast.dump(ast.parse(f1.read()))
            ast2 = ast.dump(ast.parse(f2.read()))
        self.assertEqual(ast1, ast2)


if __name__ == '__main__':
    main()
