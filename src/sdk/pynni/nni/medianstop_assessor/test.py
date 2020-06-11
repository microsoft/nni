# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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

    assessor = MedianstopAssessor(FLAGS.optimize_mode, FLAGS.start_step)
    for i in range(len(lcs)):
        #lc = []
        to_complete = True
        for k in range(len(lcs[0])):
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
