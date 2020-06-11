# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
            # pylint: disable=cell-var-from-loop
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
            temp = random.randint(x_bounds[i][0], x_bounds[i][1] - 1)
            outputs.append(temp)
        elif x_types[i] == "range_continuous":
            temp = random.uniform(x_bounds[i][0], x_bounds[i][1])
            outputs.append(temp)
        else:
            return None

    return outputs
