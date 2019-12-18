# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
__main__.py
'''
import os
import sys
import argparse
import logging
import json
import importlib

from .common import enable_multi_thread, enable_multi_phase
from .constants import ModuleName, ClassName, ClassArgs, AdvisorModuleName, AdvisorClassName
from .msg_dispatcher import MsgDispatcher

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


def create_builtin_class_instance(class_name, class_args, is_advisor=False):
    if is_advisor:
        if class_name not in AdvisorModuleName or \
            importlib.util.find_spec(AdvisorModuleName[class_name]) is None:
            raise RuntimeError('Advisor module is not found: {}'.format(class_name))
        class_module = importlib.import_module(AdvisorModuleName[class_name])
        class_constructor = getattr(class_module, AdvisorClassName[class_name])
    else:
        if class_name not in ModuleName or \
            importlib.util.find_spec(ModuleName[class_name]) is None:
            raise RuntimeError('Tuner module is not found: {}'.format(class_name))
        class_module = importlib.import_module(ModuleName[class_name])
        class_constructor = getattr(class_module, ClassName[class_name])
    if class_args:
        class_args = augment_classargs(class_args, class_name)
    else:
        class_args = augment_classargs({}, class_name)
    if class_args:
        instance = class_constructor(**class_args)
    else:
        instance = class_constructor()

    return instance


def create_customized_class_instance(class_params):
    code_dir = class_params.get('codeDir')
    class_filename = class_params.get('classFileName')
    class_name = class_params.get('className')
    class_args = class_params.get('classArgs')

    if not os.path.isfile(os.path.join(code_dir, class_filename)):
        raise ValueError('Class file not found: {}'.format(
            os.path.join(code_dir, class_filename)))
    sys.path.append(code_dir)
    module_name = os.path.splitext(class_filename)[0]
    class_module = importlib.import_module(module_name)
    class_constructor = getattr(class_module, class_name)
    if class_args:
        instance = class_constructor(**class_args)
    else:
        instance = class_constructor()
    return instance


def parse_args():
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    parser.add_argument('--exp_params', type=str, required=True)
    flags, _ = parser.parse_known_args()
    return flags

def main():
    '''
    main function.
    '''

    args = parse_args()
    logger.debug('exp_params: [%s]', args.exp_params)
    exp_params = json.loads(args.exp_params)
    logger.debug('exp_params json obj: [%s]', json.dumps(exp_params, indent=4))

    if exp_params.get('multiThread'):
        enable_multi_thread()
    if exp_params.get('multiPhase'):
        enable_multi_phase()

    if exp_params.get('advisor') is not None:
        # advisor is enabled and starts to run
        _run_advisor(exp_params)
    else:
        # tuner (and assessor) is enabled and starts to run
        assert exp_params.get('tuner') is not None
        tuner = _create_tuner(exp_params)
        if exp_params.get('assessor') is not None:
            assessor = _create_assessor(exp_params)
        else:
            assessor = None
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


def _run_advisor(exp_params):
    if exp_params.get('advisor').get('builtinAdvisorName') in AdvisorModuleName:
        dispatcher = create_builtin_class_instance(
            exp_params.get('advisor').get('builtinAdvisorName'),
            exp_params.get('advisor').get('classArgs'), True)
    else:
        dispatcher = create_customized_class_instance(exp_params.get('advisor'))
    if dispatcher is None:
        raise AssertionError('Failed to create Advisor instance')
    try:
        dispatcher.run()
    except Exception as exception:
        logger.exception(exception)
        raise


def _create_tuner(exp_params):
    if exp_params.get('tuner').get('builtinTunerName') in ModuleName:
        tuner = create_builtin_class_instance(
            exp_params.get('tuner').get('builtinTunerName'),
            exp_params.get('tuner').get('classArgs'))
    else:
        tuner = create_customized_class_instance(exp_params.get('tuner'))
    if tuner is None:
        raise AssertionError('Failed to create Tuner instance')
    return tuner


def _create_assessor(exp_params):
    if exp_params.get('assessor').get('builtinAssessorName') in ModuleName:
        assessor = create_builtin_class_instance(
            exp_params.get('assessor').get('builtinAssessorName'),
            exp_params.get('assessor').get('classArgs'))
    else:
        assessor = create_customized_class_instance(exp_params.get('assessor'))
    if assessor is None:
        raise AssertionError('Failed to create Assessor instance')
    return assessor


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
