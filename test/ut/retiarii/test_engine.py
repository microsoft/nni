import json
import os
import unittest
from pathlib import Path

import nni.retiarii
from nni.retiarii import Model, submit_models
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.execution import set_execution_engine
from nni.retiarii.execution.base import BaseExecutionEngine
from nni.retiarii.execution.python import PurePythonExecutionEngine
from nni.retiarii.integration import RetiariiAdvisor


class EngineTest(unittest.TestCase):
    def test_codegen(self):
        with open(self.enclosing_dir / 'mnist_pytorch.json') as f:
            model = Model._load(json.load(f))
            script = model_to_pytorch_script(model)
        with open(self.enclosing_dir / 'debug_mnist_pytorch.py') as f:
            reference_script = f.read()
        self.assertEqual(script.strip(), reference_script.strip())

    def test_base_execution_engine(self):
        advisor = RetiariiAdvisor()
        set_execution_engine(BaseExecutionEngine())
        with open(self.enclosing_dir / 'mnist_pytorch.json') as f:
            model = Model._load(json.load(f))
        submit_models(model, model)

        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()

    def test_py_execution_engine(self):
        
        advisor = RetiariiAdvisor()
        set_execution_engine(PurePythonExecutionEngine())
        model = Model._load({
            '_model': {
                'inputs': None,
                'outputs': None,
                'nodes': {
                    'layerchoice_1': {
                        'operation': {'type': 'LayerChoice', 'parameters': {'candidates': ['0', '1']}}
                    }
                },
                'edges': []
            }
        })
        model.python_class = object
        submit_models(model, model)

        advisor.stopping = True
        advisor.default_worker.join()
        advisor.assessor_worker.join()

    def setUp(self) -> None:
        self.enclosing_dir = Path(__file__).parent
        os.makedirs(self.enclosing_dir / 'generated', exist_ok=True)
        from nni.runtime import protocol
        protocol._out_file = open(self.enclosing_dir / 'generated/debug_protocol_out_file.py', 'wb')

    def tearDown(self) -> None:
        from nni.runtime import protocol
        protocol._out_file.close()
        nni.retiarii.execution.api._execution_engine = None
        nni.retiarii.integration_api._advisor = None
