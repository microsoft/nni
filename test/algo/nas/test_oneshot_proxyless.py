import torch
import torch.nn as nn

from nni.nas.hub.pytorch.nasbench201 import OPS_WITH_STRIDE
from nni.nas.oneshot.pytorch.supermodule.proxyless import ProxylessMixedLayer, ProxylessMixedInput, _iter_tensors


def test_proxyless_bp():
    op = ProxylessMixedLayer(
        [(name, value(3, 3, 1)) for name, value in OPS_WITH_STRIDE.items()],
        nn.Parameter(torch.randn(len(OPS_WITH_STRIDE))),
        nn.Softmax(-1), 'proxyless'
    )

    optimizer = torch.optim.SGD(op.parameters(arch=True), 0.1)

    for _ in range(10):
        x = torch.randn(1, 3, 9, 9).requires_grad_()
        op.resample({})
        y = op(x).sum()
        optimizer.zero_grad()
        y.backward()
        assert op._arch_alpha.grad.abs().sum().item() != 0


def test_proxyless_input():
    inp = ProxylessMixedInput(6, 2, nn.Parameter(torch.zeros(6)), nn.Softmax(-1), 'proxyless')

    optimizer = torch.optim.SGD(inp.parameters(arch=True), 0.1)
    for _ in range(10):
        x = [torch.randn(1, 3, 9, 9).requires_grad_() for _ in range(6)]
        inp.resample({})
        y = inp(x).sum()
        optimizer.zero_grad()
        y.backward()


def test_iter_tensors():
    a = (torch.zeros(3, 1), {'a': torch.zeros(5, 1), 'b': torch.zeros(6, 1)}, [torch.zeros(7, 1)])
    ret = []
    for x in _iter_tensors(a):
        ret.append(x.shape[0])
    assert ret == [3, 5, 6, 7]


class MultiInputLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, q, k, v=None, mask=None):
        return q + self.d, 2 * k - 2 * self.d, v, mask


def test_proxyless_multi_input():
    op = ProxylessMixedLayer(
        [
            ('a', MultiInputLayer(1)),
            ('b', MultiInputLayer(3))
        ],
        nn.Parameter(torch.randn(2)),
        nn.Softmax(-1), 'proxyless'
    )

    optimizer = torch.optim.SGD(op.parameters(arch=True), 0.1)

    for retry in range(10):
        q = torch.randn(1, 3, 9, 9).requires_grad_()
        k = torch.randn(1, 3, 9, 8).requires_grad_()
        v = None if retry < 5 else torch.randn(1, 3, 9, 7).requires_grad_()
        mask = None if retry % 5 < 2 else torch.randn(1, 3, 9, 6).requires_grad_()
        op.resample({})
        y = op(q, k, v, mask=mask)
        y = y[0].sum() + y[1].sum()
        optimizer.zero_grad()
        y.backward()
        assert op._arch_alpha.grad.abs().sum().item() != 0, op._arch_alpha.grad
