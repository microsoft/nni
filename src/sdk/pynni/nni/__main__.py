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

from .constants import ModuleName, ClassName, ClassArgs, AdvisorModuleName, AdvisorClassName
from nni.common import enable_multi_thread
from nni.msg_dispatcher import MsgDispatcher
from nni.multi_phase.multi_phase_dispatcher import MultiPhaseMsgDispatcher
logger = logging.getLogger('nni.main')
logger.debug('START')

if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()

def augment_classargs(input_class_args, classname):
    if classname in ClassArgs:
        for key, value in ClassArgs[classname].items():
            if key not in input_class_args:
                input_class_args[key] = value
    return input_class_args

def create_builtin_class_instance(classname, jsonstr_args, is_advisor = False):
    if is_advisor:
        if classname not in AdvisorModuleName or \
            importlib.util.find_spec(AdvisorModuleName[classname]) is None:
            raise RuntimeError('Advisor module is not found: {}'.format(classname))
        class_module = importlib.import_module(AdvisorModuleName[classname])
        class_constructor = getattr(class_module, AdvisorClassName[classname])
    else:
        if classname not in ModuleName or \
            importlib.util.find_spec(ModuleName[classname]) is None:
            raise RuntimeError('Tuner module is not found: {}'.format(classname))
        class_module = importlib.import_module(ModuleName[classname])
        class_constructor = getattr(class_module, ClassName[classname])
    if jsonstr_args:
        class_args = augment_classargs(json.loads(jsonstr_args), classname)
    else:
        class_args = augment_classargs({}, classname)
    if class_args:
        instance = class_constructor(**class_args)
    else:
        instance = class_constructor()
    return instance

def create_customized_class_instance(class_dir, class_filename, classname, jsonstr_args):
    if not os.path.isfile(os.path.join(class_dir, class_filename)):
        raise ValueError('Class file not found: {}'.format(
            os.path.join(class_dir, class_filename)))
    sys.path.append(class_dir)
    module_name = os.path.splitext(class_filename)[0]
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
    parser.add_argument('--advisor_class_name', type=str, required=False,
                        help='Advisor class name, the class must be a subclass of nni.MsgDispatcherBase')
    parser.add_argument('--advisor_class_filename', type=str, required=False,
                        help='Advisor class file path')
    parser.add_argument('--advisor_args', type=str, required=False,
                        help='Parameters pass to advisor __init__ constructor')
    parser.add_argument('--advisor_directory', type=str, required=False,
                        help='Advisor directory')

    parser.add_argument('--tuner_class_name', type=str, required=False,
                        help='Tuner class name, the class must be a subclass of nni.Tuner')
    parser.add_argument('--tuner_class_filename', type=str, required=False,
                        help='Tuner class file path')
    parser.add_argument('--tuner_args', type=str, required=False,
                        help='Parameters pass to tuner __init__ constructor')
    parser.add_argument('--tuner_directory', type=str, required=False,
                        help='Tuner directory')

    parser.add_argument('--assessor_class_name', type=str, required=False,
                        help='Assessor class name, the class must be a subclass of nni.Assessor')
    parser.add_argument('--assessor_args', type=str, required=False,
                        help='Parameters pass to assessor __init__ constructor')
    parser.add_argument('--assessor_directory', type=str, required=False,
                        help='Assessor directory')
    parser.add_argument('--assessor_class_filename', type=str, required=False,
                        help='Assessor class file path')

    parser.add_argument('--multi_phase', action='store_true')
    parser.add_argument('--multi_thread', action='store_true')

    flags, _ = parser.parse_known_args()
    return flags

def main():
    '''
    main function.
    '''

    args = parse_args()
    if args.multi_thread:
        enable_multi_thread()

    if args.advisor_class_name:
        # advisor is enabled and starts to run
        if args.multi_phase:
            raise AssertionError('multi_phase has not been supported in advisor')
        if args.advisor_class_name in AdvisorModuleName:
            dispatcher = create_builtin_class_instance(
                args.advisor_class_name,
                args.advisor_args, True)
        else:
            dispatcher = create_customized_class_instance(
                args.advisor_directory,
                args.advisor_class_filename,
                args.advisor_class_name,
                args.advisor_args)
        if dispatcher is None:
            raise AssertionError('Failed to create Advisor instance')
        try:
            dispatcher.run()
        except Exception as exception:
            logger.exception(exception)
            raise
    else:
        # tuner (and assessor) is enabled and starts to run
        tuner = None
        assessor = None
        if args.tuner_class_name in ModuleName:
            tuner = create_builtin_class_instance(
                args.tuner_class_name, 
                args.tuner_args)
        else:
            tuner = create_customized_class_instance(
                args.tuner_directory,
                args.tuner_class_filename,
                args.tuner_class_name,
                args.tuner_args)

        if tuner is None:
            raise AssertionError('Failed to create Tuner instance')

        if args.assessor_class_name:
            if args.assessor_class_name in ModuleName:
                assessor = create_builtin_class_instance(
                    args.assessor_class_name,
                    args.assessor_args)
            else:
                assessor = create_customized_class_instance(
                    args.assessor_directory,
                    args.assessor_class_filename,
                    args.assessor_class_name,
                    args.assessor_args)
            if assessor is None:
                raise AssertionError('Failed to create Assessor instance')

        if args.multi_phase:
            dispatcher = MultiPhaseMsgDispatcher(tuner, assessor)
        else:
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
