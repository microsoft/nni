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
from enum import Enum

from .darkopt_assessor import DarkoptAssessor
from nni.assessor import AssessResult


logger = logging.getLogger('nni.contrib.darkopt_assessor')
logger.debug('START')

class OptimizeMode(Enum):
    Maximize = 'maximize'
    Minimize = 'minimize'

def test():
    '''
    tests.
    '''
    # run accessor for mnist:
    # python -m nni.contribution.darkopt_assessor --best_score=0.90 --period=200 --threshold=0.9 --optimize_mode=maximize
    parser = argparse.ArgumentParser(
        description='parse command line parameters.')
    parser.add_argument('--best_score', type=float,
                        help='Expected best score for Assessor.')
    parser.add_argument('--period', type=int,
                        help='Expected period for Assessor.')
    parser.add_argument('--threshold', type=float,
                        help='Threshold for Assessor.')
    parser.add_argument('--optimize_mode', type=str, default='maximize',
            help='Select optimize mode for Assessor: minimize or maximize.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.optimize_mode not in [ mode.value for mode in OptimizeMode ]:
        raise AttributeError('Unsupported optimzie mode "%s"' % FLAGS.optimize_mode)

    lcs = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
           [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],
           [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3],
           [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]]
    #lcs = [[1,1,1,1,1,1,1,1,1,1],
    #       [1,1,1,1,1,1,1,1,1,1],
    #       [1,1,1,1,1,1,1,1,1,1]]

    assessor = DarkoptAssessor(FLAGS.best_score, FLAGS.period, FLAGS.threshold, FLAGS.optimize_mode)
    for i in range(4):
        #lc = []
        for k in range(10):
            #d = random.randint(i*100+0, i*100+100)
            #lc.append(d)
            ret = assessor.assess_trial(i, lcs[i][:k+1])
            print('result: %d', ret)

try:
    test()
except Exception as exception:
    logger.exception(exception)
    raise