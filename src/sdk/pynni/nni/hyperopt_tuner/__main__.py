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
'''
__main__.py
'''

import argparse
import logging

from .hyperopt_tuner import HyperoptTuner, OptimizeMode


logger = logging.getLogger('nni.examples.hyperopt_tuner')
logger.debug('START')


def main():
    '''
    main function.
    '''
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    parser.add_argument('--optimize_mode', type=str, default='maximize',
                        help='Select optimize mode for Tuner: minimize or maximize.')
    parser.add_argument('--algorithm_name', type=str, default='tpe',
                        help='Select algorithm for Tuner: tpe, random_search or anneal.')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.optimize_mode not in [mode.value for mode in OptimizeMode]:
        raise AttributeError('Unsupported optimize mode "%s"' % FLAGS.optimize_mode)

    tuner = HyperoptTuner(FLAGS.algorithm_name, FLAGS.optimize_mode)
    tuner.run()


try:
    main()
except Exception as exception:
    logger.exception(exception)
    raise
