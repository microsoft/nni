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


import numpy as np


class StorageLevel:
    DISK = 'disk'
    SPARSE = 'sparse'
    DENSE = 'dense'


class DataFormat:
    SVM = 'svm'
    NUMPY = 'numpy'
    ALL_FORMATS = [SVM, NUMPY]


class Preprocess:
    """
    center the data to mean 0 and create unit variance
    center the data to mean 0
    """
    ZSCORE = 'zscore'
    CENTER = 'center'


class Device:
    CUDA = 'cuda'
    CPU = 'cpu'


class Checkpoint:
    MODEL = 'model_state_dict'
    OPT = 'optimizer_state_dict'
    RNG = 'torch_rng_state'


class NanError(ValueError):
    pass


class Initialization:
    ZERO = 'zero'
    ON = 'on'
    OFF = 'off'
    ON_HIGH = 'onhigh'
    OFF_HIGH = 'offhigh'
    SKLEARN = 'sklearn'
    RANDOM = 'random'
    VALUE_DICT = {ZERO: 0,
                  ON: 1,
                  OFF: -1,
                  ON_HIGH: 5,
                  OFF_HIGH: -1,
                  SKLEARN: None,
                  RANDOM: None}


class Coefficients:
    """"
    coefficients for sublinear estimator were computed running the sublinear
    paper's authors' code
    """
    SLE = {1: np.array([0.60355337]),
           2: np.array([1.52705001, -0.34841729]),
           3: np.array([2.90254224, -1.87216745, 0.]),
           4: np.array([4.63445685, -5.19936195, 0., 1.50391676]),
           5: np.array([6.92948049, -14.12216211, 9.4475009, 0., -1.21093546]),
           6: np.array([9.54431082, -28.09414643, 31.84703652, -11.18763791, -1.14175281, 0.]),
           7: np.array([12.54505041, -49.64891525, 79.78828031, -46.72250909, 0., 0., 5.02973646]),
           8: np.array([16.03550163, -84.286182, 196.86078756, -215.36747071, 92.63961263, 0., 0., -4.86280869]),
           9: np.array([19.86409184, -130.76801006, 390.95349861, -570.09210416, 354.77764899, 0., -73.84234865, 0., 10.09148767]),
           10: np.array([2.41117752e+01, -1.94946061e+02, 7.34214614e+02, -1.42851995e+03, 1.41567410e+03, \
                         -5.81738134e+02, 0., 0., 3.11664751e+01, 1.05018365e+00]),
           11: np.array([28.75280839, -279.22576729, 1280.46325445, -3104.47148101, 3990.6092248, -2300.29413333, \
                         0., 427.35289033, 0., 0., -42.17587475]),
           12: np.array([33.85141912, -391.4229382, 2184.97827882, -6716.28280208, 11879.75233977, -11739.97267239, \
                         5384.94542245, 0., -674.23291712, 0., 0., 39.37456439])}


EPSILON = 1e-8
