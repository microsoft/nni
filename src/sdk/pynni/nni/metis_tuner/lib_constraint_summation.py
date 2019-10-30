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

from operator import itemgetter


def check_feasibility(x_bounds, lowerbound, upperbound):
    '''
    This can have false positives.
    For examples, parameters can only be 0 or 5, and the summation constraint is between 6 and 7.
    '''
    # x_bounds should be sorted, so even for "discrete_int" type,
    # the smallest and the largest number should the first and the last element
    x_bounds_lowerbound = sum([x_bound[0] for x_bound in x_bounds])
    x_bounds_upperbound = sum([x_bound[-1] for x_bound in x_bounds])

    # return ((x_bounds_lowerbound <= lowerbound) and (x_bounds_upperbound >= lowerbound)) or \
    #        ((x_bounds_lowerbound <= upperbound) and (x_bounds_upperbound >= upperbound))
    return (x_bounds_lowerbound <= lowerbound <= x_bounds_upperbound) or \
           (x_bounds_lowerbound <= upperbound <= x_bounds_upperbound)

def rand(x_bounds, x_types, lowerbound, upperbound, max_retries=100):
    '''
    Key idea is that we try to move towards upperbound, by randomly choose one
    value for each parameter. However, for the last parameter,
    we need to make sure that its value can help us get above lowerbound
    '''
    outputs = None

    if check_feasibility(x_bounds, lowerbound, upperbound) is True:
        # Order parameters by their range size. We want the smallest range first,
        # because the corresponding parameter has less numbers to choose from
        x_idx_sorted = []
        for i, _ in enumerate(x_bounds):
            if x_types[i] == "discrete_int":
                x_idx_sorted.append([i, len(x_bounds[i])])
            elif (x_types[i] == "range_int") or (x_types[i] == "range_continuous"):
                x_idx_sorted.append([i, math.floor(x_bounds[i][1] - x_bounds[i][0])])
        x_idx_sorted = sorted(x_idx_sorted, key=itemgetter(1))

        for _ in range(max_retries):
            budget_allocated = 0
            outputs = [None] * len(x_bounds)

            for i, _ in enumerate(x_idx_sorted):
                x_idx = x_idx_sorted[i][0]
                # The amount of unallocated space that we have
                budget_max = upperbound - budget_allocated
                # NOT the Last x that we need to assign a random number
                if i < (len(x_idx_sorted) - 1):
                    if x_bounds[x_idx][0] <= budget_max:
                        if x_types[x_idx] == "discrete_int":
                            # Note the valid integer
                            temp = []
                            for j in x_bounds[x_idx]:
                                if j <= budget_max:
                                    temp.append(j)
                            # Randomly pick a number from the integer array
                            if temp:
                                outputs[x_idx] = temp[random.randint(0, len(temp) - 1)]

                        elif (x_types[x_idx] == "range_int") or \
                                    (x_types[x_idx] == "range_continuous"):
                            outputs[x_idx] = random.randint(x_bounds[x_idx][0],
                                                            min(x_bounds[x_idx][-1], budget_max))

                else:
                    # The last x that we need to assign a random number
                    randint_lowerbound = lowerbound - budget_allocated
                    randint_lowerbound = 0 if randint_lowerbound < 0 else randint_lowerbound

                    # This check:
                    # is our smallest possible value going to overflow the available budget space,
                    # and is our largest possible value going to underflow the lower bound
                    if (x_bounds[x_idx][0] <= budget_max) and \
                            (x_bounds[x_idx][-1] >= randint_lowerbound):
                        if x_types[x_idx] == "discrete_int":
                            temp = []
                            for j in x_bounds[x_idx]:
                                # if (j <= budget_max) and (j >= randint_lowerbound):
                                if randint_lowerbound <= j <= budget_max:
                                    temp.append(j)
                            if temp:
                                outputs[x_idx] = temp[random.randint(0, len(temp) - 1)]
                        elif (x_types[x_idx] == "range_int") or \
                                (x_types[x_idx] == "range_continuous"):
                            outputs[x_idx] = random.randint(randint_lowerbound,
                                                            min(x_bounds[x_idx][1], budget_max))
                if outputs[x_idx] is None:
                    break
                else:
                    budget_allocated += outputs[x_idx]
            if None not in outputs:
                break
    return outputs
    