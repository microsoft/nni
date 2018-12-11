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

from enum import Enum, unique

@unique
class OptimizeMode(Enum):
    """
    Oprimize Mode class
    """

    Minimize = "minimize"
    Maximize = "maximize"

class Constant:
    '''Constant for the Tuner.
    '''
    MAX_LAYERS = 100
    N_NEIGHBOURS = 8
    MAX_MODEL_SIZE = 1 << 24
    KERNEL_LAMBDA = 1.0
    BETA = 2.576
    MLP_MODEL_LEN = 3
    MLP_MODEL_WIDTH = 5
    MODEL_LEN = 3
    MODEL_WIDTH = 64
    POOLING_KERNEL_SIZE = 2
    DENSE_DROPOUT_RATE = 0.5
    CONV_DROPOUT_RATE = 0.25
    MLP_DROPOUT_RATE = 0.25
    CONV_BLOCK_DISTANCE = 2
    BATCH_SIZE = 128
    T_MIN = 0.0001
