# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import logging
import json
import base64

from .runtime.common import enable_multi_thread
from .runtime.msg_dispatcher import MsgDispatcher
from .tools.package_utils import create_builtin_class_instance, create_customized_class_instance

logger = logging.getLogger('nni.main')
logger.debug('START')

if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()


def main():
    parser = argparse.ArgumentParser(description='Dispatcher command line parser')
    parser.add_argument('--exp_params', type=str, required=True)
    args, _ = parser.parse_known_args()

    exp_params_decode = base64.b64decode(args.exp_params).decode('utf-8')
    logger.debug('decoded exp_params: [%s]', exp_params_decode)
    exp_params = json.loads(exp_params_decode)
    logger.debug('exp_params json obj: [%s]', json.dumps(exp_params, indent=4))

    if exp_params.get('deprecated', {}).get('multiThread'):
        enable_multi_thread()

    if 'trainingServicePlatform' in exp_params:  # config schema is v1
        from types import SimpleNamespace
        from .experiment.config.convert import convert_algo
        for algo_type in ['tuner', 'assessor', 'advisor']:
            if algo_type in exp_params:
                exp_params[algo_type] = convert_algo(algo_type, exp_params, SimpleNamespace()).json()

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
    if exp_params.get('advisor').get('name'):
        dispatcher = create_builtin_class_instance(
            exp_params['advisor']['name'],
            exp_params['advisor'].get('classArgs'),
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
    if exp_params['tuner'].get('name'):
        tuner = create_builtin_class_instance(
            exp_params['tuner']['name'],
            exp_params['tuner'].get('classArgs'),
            'tuners')
    else:
        tuner = create_customized_class_instance(exp_params['tuner'])
    if tuner is None:
        raise AssertionError('Failed to create Tuner instance')
    return tuner


def _create_assessor(exp_params):
    if exp_params['assessor'].get('name'):
        assessor = create_builtin_class_instance(
            exp_params['assessor']['name'],
            exp_params['assessor'].get('classArgs'),
            'assessors')
    else:
        assessor = create_customized_class_instance(exp_params['assessor'])
    if assessor is None:
        raise AssertionError('Failed to create Assessor instance')
    return assessor


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
