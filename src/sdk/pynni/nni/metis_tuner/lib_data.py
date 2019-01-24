# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
import random


def match_val_type(vals, vals_bounds, vals_types):
    '''
    Update values in the array, to match their corresponding type
    '''
    vals_new = []

    for i, _ in enumerate(vals_types):
        if vals_types[i] == "discrete_int":
            # Find the closest integer in the array, vals_bounds
            vals_new.append(min(vals_bounds[i], key=lambda x: abs(x - vals[i])))
        elif vals_types[i] == "range_int":
            # Round down to the nearest integer
            vals_new.append(math.floor(vals[i]))
        elif vals_types[i] == "range_continuous":
            # Don't do any processing for continous numbers
            vals_new.append(vals[i])
        else:
            return None

    return vals_new


def rand(x_bounds, x_types):
    '''
    Random generate variable value within their bounds
    '''
    outputs = []

    for i, _ in enumerate(x_bounds):
        if x_types[i] == "discrete_int":
            temp = x_bounds[i][random.randint(0, len(x_bounds[i]) - 1)]
            outputs.append(temp)
        elif x_types[i] == "range_int":
            temp = random.randint(x_bounds[i][0], x_bounds[i][1])
            outputs.append(temp)
        elif x_types[i] == "range_continuous":
            temp = random.uniform(x_bounds[i][0], x_bounds[i][1])
            outputs.append(temp)
        else:
            return None

    return outputs
    