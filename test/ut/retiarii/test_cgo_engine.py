import json
import os
import sys
import threading
import unittest
import logging
import time
import torch

from pathlib import Path

from nni.retiarii.execution.cgo_engine import CGOExecutionEngine
from nni.retiarii.execution.logical_optimizer.logical_plan import LogicalPlan
from nni.retiarii.execution.logical_optimizer.opt_dedup_input import DedupInputOptimizer
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii import Model, Node

from nni.retiarii import Model, submit_models
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.integration import RetiariiAdvisor
from nni.retiarii.trainer import PyTorchImageClassificationTrainer, PyTorchMultiModelTrainer
from nni.retiarii.utils import import_


def _load_mnist(n_models: int = 1):
    path = Path(__file__).parent / 'converted_mnist_pytorch.json'
    with open(path) as f:
        mnist_model = Model._load(json.load(f))
    if n_models == 1:
        return mnist_model
    else:
        models = [mnist_model]
        for i in range(n_models-1):
            models.append(mnist_model.fork())
        return models


@unittest.skip('Skipped in this version')
class CGOEngineTest(unittest.TestCase):

    def test_submit_models(self):
        os.environ['CGO'] = 'true'
        os.makedirs('generated', exist_ok=True)
        from nni.runtime import protocol, platform
        import nni.runtime.platform.test as tt
        protocol._out_file = open('generated/debug_protocol_out_file.py', 'wb')
        protocol._in_file = open('generated/debug_protocol_out_file.py', 'rb')

        models = _load_mnist(2)
        advisor = RetiariiAdvisor()
        submit_models(*models)

        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            cmd, data = protocol.receive()
            params = json.loads(data)
            params['parameters']['training_kwargs']['max_steps'] = 100

            tt.init_params(params)

            trial_thread = threading.Thread(target=CGOExecutionEngine.trial_execute_graph())
            trial_thread.start()
            last_metric = None
            while True:
                time.sleep(1)
                if tt._last_metric:
                    metric = tt.get_last_metric()
                    if metric == last_metric:
                        continue
                    advisor.handle_report_metric_data(metric)
                    last_metric = metric
                if not trial_thread.is_alive():
                    break

            trial_thread.join()
        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()


if __name__ == '__main__':
    unittest.main()
