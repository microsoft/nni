import argparse
import os
import sys
import pytorch_lightning as pl
import pytest
from subprocess import Popen

from nni.retiarii import strategy
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from .test_oneshot import _mnist_net

pytestmark = pytest.mark.skipif(pl.__version__ < '1.0', reason='Incompatible APIs')

def test_multi_trial():
    evaluator_kwargs = {
        'max_epochs': 1
    }

    to_test = [
        # (model, evaluator)
        _mnist_net('simple', evaluator_kwargs),
        _mnist_net('simple_value_choice', evaluator_kwargs),
        _mnist_net('value_choice', evaluator_kwargs),
        _mnist_net('repeat', evaluator_kwargs),
        _mnist_net('custom_op', evaluator_kwargs),
    ]

    for base_model, evaluator in to_test:
        search_strategy = strategy.Random()
        exp = RetiariiExperiment(base_model, evaluator, strategy=search_strategy)
        exp_config = RetiariiExeConfig('local')
        exp_config.experiment_name = 'mnist_unittest'
        exp_config.trial_concurrency = 1
        exp_config.max_trial_number = 1
        exp_config.training_service.use_active_gpu = False
        exp.run(exp_config, 8080)
        assert isinstance(exp.export_top_models()[0], dict)
        exp.stop()

python_script = """
from nni.retiarii import strategy
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from test_oneshot import _mnist_net

base_model, evaluator = _mnist_net('simple', {'max_epochs': 1})
search_strategy = strategy.Random()
exp = RetiariiExperiment(base_model, evaluator, strategy=search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'mnist_unittest'
exp_config.trial_concurrency = 1
exp_config.max_trial_number = 1
exp_config.training_service.use_active_gpu = False
exp.run(exp_config, 8080)
assert isinstance(exp.export_top_models()[0], dict)
"""

@pytest.mark.timeout(600)
def test_exp_exit_without_stop():
    script_name = 'tmp_multi_trial.py'
    with open(script_name, 'w') as f:
        f.write(python_script)
    proc = Popen([sys.executable, script_name])
    proc.wait()
    os.remove(script_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all', metavar='E',
                        help='experiment to run, default = all')
    args = parser.parse_args()

    if args.exp == 'all':
        test_multi_trial()
        test_exp_exit_without_stop()
    else:
        globals()[f'test_{args.exp}']()
