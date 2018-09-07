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
import os
import sys
import argparse
import logging
import json
import importlib

from nni.msg_dispatcher import MsgDispatcher
from nni.hyperopt_tuner.hyperopt_tuner import HyperoptTuner
from nni.evolution_tuner.evolution_tuner import EvolutionTuner
from nni.medianstop_assessor.medianstop_assessor import MedianstopAssessor

logger = logging.getLogger('nni.main')
logger.debug('START')

BUILT_IN_CLASS_NAMES = ['HyperoptTuner', 'EvolutionTuner', 'MedianstopAssessor']

def create_builtin_class_instance(classname, jsonstr_args):
    if jsonstr_args:
        class_args = json.loads(jsonstr_args)
        instance = eval(classname)(**class_args)
    else:
        instance = eval(classname)()
    return instance

def create_customized_class_instance(class_dir, class_filename, classname, jsonstr_args):
    if not os.path.isfile(os.path.join(class_dir, class_filename)):
        raise ValueError('Class file not found: {}'.format(os.path.join(class_dir, class_filename)))
    sys.path.append(class_dir)
    module_name = class_filename.split('.')[0]
    class_module = importlib.import_module(module_name)
    class_constructor = getattr(class_module, classname)
    if jsonstr_args:
        class_args = json.loads(jsonstr_args)
        instance = class_constructor(**class_args)
    else:
        instance = class_constructor()
    return instance

def parse_args():
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    parser.add_argument('--tuner_class_name', type=str, required=True,
                        help='Tuner class name, the class must be a subclass of nni.Tuner')
    parser.add_argument('--tuner_args', type=str, required=False,
                        help='Parameters pass to tuner __init__ constructor')
    parser.add_argument('--tuner_directory', type=str, required=False,
                        help='Tuner directory')
    parser.add_argument('--tuner_class_filename', type=str, required=False,
                        help='Tuner class file path')

    parser.add_argument('--assessor_class_name', type=str, required=False,
                        help='Assessor class name, the class must be a subclass of nni.Assessor')
    parser.add_argument('--assessor_args', type=str, required=False,
                        help='Parameters pass to assessor __init__ constructor')
    parser.add_argument('--assessor_directory', type=str, required=False,
                        help='Assessor directory')
    parser.add_argument('--assessor_class_filename', type=str, required=False,
                        help='Assessor class file path')

    flags, _ = parser.parse_known_args()
    return flags

def main():
    '''
    main function.
    '''

    args = parse_args()

    tuner = None
    assessor = None

    if args.tuner_class_name is None:
        raise ValueError('Tuner must be specified')
    if args.tuner_class_name in BUILT_IN_CLASS_NAMES:
        tuner = create_builtin_class_instance(args.tuner_class_name, args.tuner_args)
    else:
        tuner = create_customized_class_instance(args.tuner_directory, args.tuner_class_filename, args.tuner_class_name, args.tuner_args)

    if args.assessor_class_name:
        if args.assessor_class_name in BUILT_IN_CLASS_NAMES:
            assessor = create_builtin_class_instance(args.assessor_class_name, args.assessor_args)
        else:
            assessor = create_customized_class_instance(args.assessor_directory, \
                args.assessor_class_filename, args.assessor_class_name, args.assessor_args)

    if tuner is None:
        raise AssertionError('Failed to create Tuner instance')

    dispatcher = MsgDispatcher(tuner, assessor)

    try:
        dispatcher.run()
        tuner._on_exit()
        if assessor is not None:
            assessor._on_exit()
    except Exception as exception:
        logger.exception(exception)
        tuner._on_error()
        if assessor is not None:
            assessor._on_error()
        raise

if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
