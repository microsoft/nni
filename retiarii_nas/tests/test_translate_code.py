import unittest

import torch
import torch.nn as nn
from sdk.translate_code import gen_pytorch_graph


class TranslateCodeTest(unittest.TestCase):
    def test_split(self):
        class Net(nn.Module):
            def forward(self, x):
                x1, x2 = x.split(2)
                return torch.cat((x1, x2))
        gen_pytorch_graph(Net(), torch.randn(4, 6))


if __name__ == '__main__':
    unittest.main()
