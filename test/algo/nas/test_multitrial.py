import multiprocessing
import os
import sys
import time

import pytest
import pytorch_lightning as pl
from nni.retiarii import strategy
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from ut.nas.test_experiment import nas_experiment_trial_params, ensure_success
from .test_oneshot import _mnist_net

pytestmark = pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')


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
    exp_config._trial_command_params = nas_experiment_trial_params(pytestconfig.rootpath)
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
        exp_config._trial_command_params = nas_experiment_trial_params(rootpath)
        exp.run(exp_config)
        ensure_success(exp)
        assert isinstance(exp.export_top_models()[0], dict)
    finally:
        # https://stackoverflow.com/questions/34506638/how-to-register-atexit-function-in-pythons-multiprocessing-subprocess
        import atexit
        atexit._run_exitfuncs()


def test_exp_exit_without_stop(pytestconfig):
    # NOTE: Multiprocessing has compatibility issue with OpenMP.
    # It makes the MNIST dataset fails to load on pipeline.
    # https://github.com/pytorch/pytorch/issues/50669
    # Need to use spawn as a workaround of this issue.
    ctx = multiprocessing.get_context('spawn')
    process = ctx.Process(
        target=_test_experiment_in_separate_process,
        kwargs=dict(rootpath=pytestconfig.rootpath)
    )
    process.start()
    print('Waiting for experiment in sub-process.')
    timeout = 180
    for _ in range(timeout):
        if process.is_alive():
            time.sleep(1)
        else:
            assert process.exitcode == 0
            return
    process.kill()
    raise RuntimeError(f'Experiment fails to stop in {timeout} seconds.')
