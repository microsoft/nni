import multiprocessing
import os
import subprocess
import time

import pytest
import pytorch_lightning as pl
from nni.retiarii import strategy
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from ut.nas.test_experiment import nas_experiment_trial_params, ensure_success
from .test_oneshot import _mnist_net

# pytestmark = pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')
pytestmark = pytest.mark.skip(reason='Will be rewritten.')


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


def test_multitrial_experiment_resume_view(pytestconfig):
    # start a normal nas experiment
    base_model, evaluator = _mnist_net('simple', {'max_epochs': 1})
    search_strategy = strategy.Random()
    exp = RetiariiExperiment(base_model, evaluator, strategy=search_strategy)
    exp_id = exp.id
    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 1
    exp_config.max_trial_number = 1
    exp_config._trial_command_params = nas_experiment_trial_params(pytestconfig.rootpath)
    exp.run(exp_config)
    ensure_success(exp)
    assert isinstance(exp.export_top_models()[0], dict)
    exp.stop()

    # resume the above nas experiment. only tested the resume logic in the python side,
    # as no more trial is executed after resume, the above experiment is already finished
    print('python api resume...')
    exp = RetiariiExperiment.resume(exp_id)
    ensure_success(exp)
    # sleep here because there would be several seconds for the experiment status to change
    # to ERROR from INITIALIZED/RUNNING if the resume gets error.
    time.sleep(6)
    assert exp.get_status() == 'DONE', f'The experiment status should not be {exp.get_status()}'
    # TODO: currently `export_top_models` does not work as strategy's states are not resumed
    # assert isinstance(exp.export_top_models()[0], dict)
    exp.stop()
    # view the above experiment in non blocking mode then stop it
    print('python api view...')
    exp = RetiariiExperiment.view(exp_id, non_blocking=True)
    assert exp.get_status() == 'VIEWED', f'The experiment status should not be {exp.get_status()}'
    exp.stop()

    # the following is nnictl resume and view
    print('nnictl resume...')
    new_env = os.environ.copy()
    new_env['PYTHONPATH'] = str(pytestconfig.rootpath)
    # NOTE: experiment status (e.g., ERROR) is not checked, because it runs in blocking mode and
    # the rest server exits right after the command is done
    proc = subprocess.run(f'nnictl resume {exp_id}', shell=True, env=new_env)
    assert proc.returncode == 0, 'resume nas experiment failed with code %d' % proc.returncode
    print('nnictl view...')
    proc = subprocess.run(f'nnictl view {exp_id}', shell=True)
    assert proc.returncode == 0, 'view nas experiment failed with code %d' % proc.returncode
    proc = subprocess.run(f'nnictl stop {exp_id}', shell=True)
    assert proc.returncode == 0, 'stop viewed nas experiment failed with code %d' % proc.returncode