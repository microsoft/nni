# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os

DATA_DIR = r'/mnt/chicm/data/salt'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_MASK_DIR =  os.path.join(TRAIN_DIR, 'masks')
TEST_IMG_DIR = os.path.join(TEST_DIR, 'images')

LABEL_FILE = os.path.join(DATA_DIR, 'train.csv')
DEPTHS_FILE = os.path.join(DATA_DIR, 'depths.csv')
META_FILE = os.path.join(DATA_DIR, 'meta.csv')

MODEL_DIR = os.path.join(DATA_DIR, 'models')

ID_COLUMN = 'id'
DEPTH_COLUMN = 'z'
X_COLUMN = 'file_path_image'
Y_COLUMN = 'file_path_mask'

H = W = 128
ORIG_H = ORIG_W = 101