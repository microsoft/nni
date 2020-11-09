# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A test for hyperband, using nasbench201. So it need install the dependencies for nasbench201 at first.
"""
import argparse
import logging
import random
import time

import nni
from nni.utils import merge_parameter
from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats


logger = logging.getLogger('test_hyperband')


def main(args):
    r = args.pop('TRIAL_BUDGET')
    dataset = [t for t in query_nb201_trial_stats(args, 200, 'cifar100', include_intermediates=True)]
    test_acc = random.choice(dataset)['intermediates'][r - 1]['ori_test_acc'] / 100
    time.sleep(random.randint(0, 10))
    nni.report_final_result(test_acc)
    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')

def get_params():
    parser = argparse.ArgumentParser(description='Hyperband Test')
    parser.add_argument("--0_1", type=str, default='none')
    parser.add_argument("--0_2", type=str, default='none')
    parser.add_argument("--0_3", type=str, default='none')
    parser.add_argument("--1_2", type=str, default='none')
    parser.add_argument("--1_3", type=str, default='none')
    parser.add_argument("--2_3", type=str, default='none')
    parser.add_argument("--TRIAL_BUDGET", type=int, default=200)

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
