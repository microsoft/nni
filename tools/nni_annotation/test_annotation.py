# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


# pylint: skip-file

from .__init__ import *

import ast
import json
import os
import shutil
import tempfile
from unittest import TestCase, main


class AnnotationTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir('nni_annotation')
        if os.path.isdir('_generated'):
            shutil.rmtree('_generated')

    def test_search_space_generator(self):
        search_space = generate_search_space('testcase/annotated')
        with open('testcase/searchspace.json') as f:
            self.assertEqual(search_space, json.load(f))

    def test_code_generator(self):
        code_dir = expand_annotations('testcase/usercode', '_generated')
        self.assertEqual(code_dir, '_generated')
        self._assert_source_equal('testcase/annotated/mnist.py', '_generated/mnist.py')
        self._assert_source_equal('testcase/annotated/dir/simple.py', '_generated/dir/simple.py')
        with open('testcase/usercode/nonpy.txt') as src, open('_generated/nonpy.txt') as dst:
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
