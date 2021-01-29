import unittest

import nni.retiarii.nn.pytorch as nn
import torch
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.codegen import model_to_pytorch_script


class TestHighLevelAPI(unittest.TestCase):
    def test_layer_choice(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.LayerChoice([
                    nn.Conv2d(3, 3, kernel_size=1),
                    nn.Conv2d(3, 5, kernel_size=1)
                ])

        model = Net()
        script_module = torch.jit.script(model)
        model_ir = convert_to_graph(script_module, model)
        model_code = model_to_pytorch_script(model_ir)
        print(model_code)
