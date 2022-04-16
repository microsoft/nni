# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import logging
import json
import base64

from .runtime.msg_dispatcher import MsgDispatcher
from .runtime.msg_dispatcher_base import MsgDispatcherBase
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

    if 'trainingServicePlatform' in exp_params:  # config schema is v1
        from .experiment.config.convert import convert_algo
        for algo_type in ['tuner', 'assessor']:
            if algo_type in exp_params:
                exp_params[algo_type] = convert_algo(algo_type, exp_params[algo_type])
        if 'advisor' in exp_params:
            exp_params['tuner'] = convert_algo('advisor', exp_params['advisor'])

    assert exp_params.get('tuner') is not None
    tuner = _create_algo(exp_params['tuner'], 'tuner')

    if isinstance(tuner, MsgDispatcherBase):  # is advisor
        logger.debug(f'Tuner {type(tuner).__name__} is advisor.')
        if exp_params.get('assessor') is not None:
            logger.error('Tuner {type(tuner).__name__} does not support early stopping. Assessor is ignored.')
        tuner.run()
        return

    if exp_params.get('assessor') is not None:
        assessor = _create_algo(exp_params['assessor'], 'assessor')
    else:
        assessor = None
    dispatcher = MsgDispatcher(tuner, assessor)

    try:
        dispatcher.run()
        tuner._on_exit()
        if assessor is not None:
            assessor._on_exit()
    except Exception as exception:
        tuner._on_error()
        if assessor is not None:
            assessor._on_error()
        raise


def _create_algo(algo_config, algo_type):
    if algo_config.get('name'):
        algo = create_builtin_class_instance(algo_config['name'], algo_config.get('classArgs'), algo_type + 's')
    else:
        algo = create_customized_class_instance(algo_config)
    if algo is None:
        raise AssertionError(f'Failed to create {algo_type} instance')
    return algo


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
