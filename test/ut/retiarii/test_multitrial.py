import multiprocessing
import os
import sys
import time

import pytest
import pytorch_lightning as pl
from nni.retiarii import strategy
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from .test_oneshot import _mnist_net

pytestmark = pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')


def _trial_params(rootpath):
    params = {}
    if sys.platform == 'windows':
        params['envs'] = f'set PYTHONPATH={rootpath} && '
    else:
        params['envs'] = f'PYTHONPATH={rootpath}:$PYTHONPATH'
    return params


def ensure_success(exp: RetiariiExperiment):
    # check experiment directory exists
    exp_dir = os.path.join(
        exp.config.canonical_copy().experiment_working_directory,
        exp.id
    )
    assert os.path.exists(exp_dir) and os.path.exists(os.path.join(exp_dir, 'trials'))

    # check job status
    job_stats = exp.get_job_statistics()
    if not (len(job_stats) == 1 and job_stats[0]['trialJobStatus'] == 'SUCCEEDED'):
        print('Experiment jobs did not all succeed. Status is:', job_stats, file=sys.stderr)
        print('Trying to fetch trial logs.', file=sys.stderr)

        for root, _, files in os.walk(os.path.join(exp_dir, 'trials')):
            for file in files:
                fpath = os.path.join(root, file)
                print('=' * 10 + ' ' + fpath + ' ' + '=' * 10, file=sys.stderr)
                print(open(fpath).read(), file=sys.stderr)

        raise RuntimeError('Experiment jobs did not all succeed.')


@pytest.mark.parametrize('model', [
    'simple', 'simple_value_choice', 'value_choice', 'repeat', 'custom_op'
])
def test_multi_trial(model, pytestconfig):
    evaluator_kwargs = {
        'max_epochs': 1
    }

    base_model, evaluator = _mnist_net(model, evaluator_kwargs)

    search_strategy = strategy.Random()
    exp = RetiariiExperiment(base_model, evaluator, strategy=search_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'mnist_unittest'
    exp_config.trial_concurrency = 1
    exp_config.max_trial_number = 1
    exp_config.trial_command_params = _trial_params(pytestconfig.rootpath)
    exp.run(exp_config)
    ensure_success(exp)
    assert isinstance(exp.export_top_models()[0], dict)
    exp.stop()


def _test_experiment_in_separate_process(rootpath):
    try:
        base_model, evaluator = _mnist_net('simple', {'max_epochs': 1})
        search_strategy = strategy.Random()
        exp = RetiariiExperiment(base_model, evaluator, strategy=search_strategy)
        exp_config = RetiariiExeConfig('local')
        exp_config.experiment_name = 'mnist_unittest'
        exp_config.trial_concurrency = 1
        exp_config.max_trial_number = 1
        exp_config.trial_command_params = _trial_params(rootpath)
        print('Prepared to run experiment.')
        exp.run(exp_config)
        ensure_success(exp)
        assert isinstance(exp.export_top_models()[0], dict)
    finally:
        # https://stackoverflow.com/questions/34506638/how-to-register-atexit-function-in-pythons-multiprocessing-subprocess
        import atexit
        atexit._run_exitfuncs()


def test_exp_exit_without_stop(pytestconfig):
    process = multiprocessing.Process(
        target=_test_experiment_in_separate_process,
        kwargs=dict(rootpath=pytestconfig.rootpath)
    )
    process.start()
    print('Waiting for experiment in sub-process.')
    for _ in range(600):
        if process.is_alive():
            time.sleep(1)
        else:
            assert process.exitcode == 0
            return
    process.kill()
    raise RuntimeError('Experiment fails to stop in 600 seconds.')
