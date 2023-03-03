import time
import sys

from pathlib import Path

from nni.experiment import Experiment

search_space = {
    'C': {'_type': 'uniform', '_value': [0.1, 1]},
    'kernel': {'_type': 'choice', '_value': ['linear', 'rbf', 'poly', 'sigmoid']},
    'degree': {'_type': 'choice', '_value': [1, 2, 3, 4]},
    'gamma': {'_type': 'uniform', '_value': [0.01, 0.1]},
    'coef0': {'_type': 'uniform', '_value': [0.01, 0.1]}
}

_wait_seconds = 30.  # Wait for experiment to start. But it shouldn't take that long?
_refresh_seconds = 5.  # Wait for status to change.

def _create_experiment(max_trial_number):
    experiment = Experiment('local')
    experiment.config.trial_command = sys.executable + ' assets/trial_sklearn.py'
    experiment.config.trial_code_directory = Path(__file__).parent
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'Random'
    experiment.config.max_trial_number = max_trial_number
    experiment.config.trial_concurrency = 2
    return experiment

def test_experiment_run():
    experiment = _create_experiment(5)
    experiment.run()
    experiment.stop()

def test_experiment_resume_view():
    experiment = _create_experiment(100)
    experiment.run(wait_completion=False)

    time.sleep(_wait_seconds)

    assert experiment.get_status() == 'RUNNING'
    stats = {s['trialJobStatus']: s['trialJobNumber'] for s in experiment.get_job_statistics()}
    assert sum(stats.values()) > 0
    assert all(s in ['WAITING', 'RUNNING', 'SUCCEEDED'] for s in stats)

    experiment.stop()

    experiment.resume(wait_completion=False)

    time.sleep(_wait_seconds)

    assert experiment.get_status() == 'RUNNING'
    stats2 = {s['trialJobStatus']: s['trialJobNumber'] for s in experiment.get_job_statistics()}
    assert sum(stats2.values()) > sum(stats.values()), f'{stats2} vs. {stats}'
    assert all(s in ['WAITING', 'RUNNING', 'SUCCEEDED'] for s in stats)

    experiment.stop()

    try:
        experiment.view(non_blocking=True)

        time.sleep(_refresh_seconds)  # Wait for starting to run.

        stats3 = {s['trialJobStatus']: s['trialJobNumber'] for s in experiment.get_job_statistics()}
        assert sum(stats3.values()) >= sum(stats2.values()), f'{stats3} vs. {stats2}'
        assert len(experiment.list_trial_jobs()) == sum(stats3.values())
        assert experiment.get_status() == 'VIEWED'
    finally:
        experiment.stop()

def test_experiment_resume_with_id():
    experiment = _create_experiment(100)
    experiment.run(wait_completion=False)

    time.sleep(_wait_seconds)

    num_trials = len(experiment.list_trial_jobs())
    experiment.stop()

    time.sleep(_wait_seconds)  # Otherwise manifest is not updated. Experiment is still "running".

    experiment2 = Experiment(None, id=experiment.id)
    experiment2.run_or_resume(wait_completion=False)

    time.sleep(_wait_seconds)

    assert len(experiment2.list_trial_jobs()) > num_trials
    assert experiment2.get_status() == 'RUNNING'

    experiment2.stop()

    time.sleep(_wait_seconds)

    experiment3 = Experiment(None, id=experiment.id)
    try:
        experiment3.view(non_blocking=True)

        time.sleep(_refresh_seconds)
        assert experiment3.get_status() == 'VIEWED'
    finally:
        experiment3.stop()
