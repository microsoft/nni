# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import argparse
import logging
import json
import importlib
import base64

from .common import enable_multi_thread, enable_multi_phase
from .msg_dispatcher import MsgDispatcher
from .package_utils import get_all_builtin_names, get_builtin_algo_meta, ALGO_TYPES

logger = logging.getLogger('nni.main')
logger.debug('START')

if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()

def create_builtin_class_instance(builtin_name, input_class_args, algo_type):
    assert algo_type in ALGO_TYPES
    if builtin_name not in get_all_builtin_names(algo_type):
        raise RuntimeError('Builtin name is not found: {}'.format(builtin_name))

    def parse_algo_meta(algo_meta, input_class_args):
        """
        1. parse class_name field in meta data into module name and class name,
        for example:
            parse class_name 'nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner' in meta data into:
            module name: nni.hyperopt_tuner.hyperopt_tuner
            class name: HyperoptTuner
        2. merge user specified class args together with builtin class args.
        """
        assert algo_meta
        full_class_name = algo_meta['class_name']
        parts = full_class_name.split('.')
        module_name, class_name = '.'.join(parts[:-1]), parts[-1]

        class_args = {}
        if 'class_args' in algo_meta:
            class_args = algo_meta['class_args']
        if input_class_args is not None:
            class_args.update(input_class_args)

        return module_name, class_name, class_args

    algo_meta = get_builtin_algo_meta(algo_type, builtin_name)
    module_name, class_name, class_args = parse_algo_meta(algo_meta, input_class_args)

    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError('Builtin module can not be loaded: {}'.format(module_name))

    class_module = importlib.import_module(module_name)
    class_constructor = getattr(class_module, class_name)

    instance = class_constructor(**class_args)

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

    if class_args is None:
        class_args = {}
    instance = class_constructor(**class_args)

    return instance


def main():
    parser = argparse.ArgumentParser(description='Dispatcher command line parser')
    parser.add_argument('--exp_params', type=str, required=True)
    args, _ = parser.parse_known_args()

    exp_params_decode = base64.b64decode(args.exp_params).decode('utf-8')
    logger.debug('decoded exp_params: [%s]', exp_params_decode)
    exp_params = json.loads(exp_params_decode)
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
    if exp_params.get('advisor').get('builtinAdvisorName'):
        dispatcher = create_builtin_class_instance(
            exp_params.get('advisor').get('builtinAdvisorName'),
            exp_params.get('advisor').get('classArgs'),
            'advisors')
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
    if exp_params.get('tuner').get('builtinTunerName'):
        tuner = create_builtin_class_instance(
            exp_params.get('tuner').get('builtinTunerName'),
            exp_params.get('tuner').get('classArgs'),
            'tuners')
    else:
        tuner = create_customized_class_instance(exp_params.get('tuner'))
    if tuner is None:
        raise AssertionError('Failed to create Tuner instance')
    return tuner


def _create_assessor(exp_params):
    if exp_params.get('assessor').get('builtinAssessorName'):
        assessor = create_builtin_class_instance(
            exp_params.get('assessor').get('builtinAssessorName'),
            exp_params.get('assessor').get('classArgs'),
            'assessors')
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
