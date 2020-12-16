import json
import os
import sys
import threading
import unittest
import logging
import time

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
class DedupInputTest(unittest.TestCase):
    def _build_logical_with_mnist(self, n_models: int):
        lp = LogicalPlan()
        models = _load_mnist(n_models=n_models)
        for m in models:
            lp.add_model(m)
        return lp, models

    def _test_add_model(self):
        lp, models = self._build_logical_with_mnist(3)
        for node in lp.logical_graph.hidden_nodes:
            old_nodes = [m.root_graph.get_node_by_id(node.id) for m in models]

            self.assertTrue(any([old_nodes[0].__repr__() == Node.__repr__(x) for x in old_nodes]))

    def test_dedup_input(self):
        os.environ['CGO'] = 'true'
        lp, models = self._build_logical_with_mnist(3)
        opt = DedupInputOptimizer()
        opt.convert(lp)
        with open('dedup_logical_graph.json', 'r') as fp:
            correct_dump = fp.readlines()
        lp_dump = lp.logical_graph._dump()

        self.assertTrue(correct_dump[0] == json.dumps(lp_dump))

        advisor = RetiariiAdvisor()
        cgo = CGOExecutionEngine()

        phy_models = cgo._assemble(lp)
        self.assertTrue(len(phy_models) == 1)
        # logging.info(phy_models[0][0]._dump())
        # script=model_to_pytorch_script(phy_models[0][0], placement = phy_models[0][1])
        # logging.info(script)
        # with open('generated/debug_dedup_input.py', 'w') as fp:
        #     fp.write(script)
        # sys.path.insert(0, 'generated')
        # multi_model = import_('debug_dedup_input.logical_0')
        # trainer = PyTorchMultiModelTrainer(
        #     multi_model(), phy_models[0][0].training_config.kwargs
        # )
        # trainer.fit()

        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()


if __name__ == '__main__':
    unittest.main()
