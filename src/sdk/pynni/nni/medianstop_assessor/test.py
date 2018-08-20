# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging
import random

from .medianstop_assessor import MedianstopAssessor
from nni.assessor import AssessResult


logger = logging.getLogger('nni.contrib.medianstop_assessor')
logger.debug('START')


def test():
    '''
    tests.
    '''
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    parser.add_argument('--start_from', type=int, default=10, dest='start_step',
                        help='Assessing each trial from the step start_step.')
    parser.add_argument('--optimize_mode', type=str, default='maximize',
                        help='Select optimize mode for Tuner: minimize or maximize.')
    FLAGS, _ = parser.parse_known_args()

    lcs = [[1,1,1,1,1,1,1,1,1,1],
           [2,2,2,2,2,2,2,2,2,2],
           [3,3,3,3,3,3,3,3,3,3],
           [4,4,4,4,4,4,4,4,4,4]]
    #lcs = [[1,1,1,1,1,1,1,1,1,1],
    #       [1,1,1,1,1,1,1,1,1,1],
    #       [1,1,1,1,1,1,1,1,1,1]]

    assessor = MedianstopAssessor(FLAGS.start_step, FLAGS.optimize_mode)
    for i in range(4):
        #lc = []
        to_complete = True
        for k in range(10):
            #d = random.randint(i*100+0, i*100+100)
            #lc.append(d)
            ret = assessor.assess_trial(i, lcs[i][:k+1])
            print('result: %d', ret)
            if ret == AssessResult.Bad:
                assessor.trial_end(i, False)
                to_complete = False
                break
        if to_complete:
            assessor.trial_end(i, True)

try:
    test()
except Exception as exception:
    logger.exception(exception)
    raise