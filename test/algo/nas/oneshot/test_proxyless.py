import random

import pytest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import nni
from nni.mutable import frozen
from nni.nas.hub.pytorch.nasbench201 import OPS_WITH_STRIDE
from nni.nas.nn.pytorch import Repeat
from nni.nas.oneshot.pytorch.supermodule.base import BaseSuperNetModule
from nni.nas.oneshot.pytorch.supermodule.proxyless import ProxylessMixedLayer, ProxylessMixedInput, ProxylessMixedRepeat, _iter_tensors
from pytorch_lightning import LightningModule, Trainer

@pytest.fixture(autouse=True)
def context():
    frozen._ENSURE_FROZEN_STRICT = False
    yield
    frozen._ENSURE_FROZEN_STRICT = True


def test_proxyless_bp():
    op = ProxylessMixedLayer(
        {name: value(3, 3, 1) for name, value in OPS_WITH_STRIDE.items()},
        nn.Parameter(torch.randn(len(OPS_WITH_STRIDE))),
        nn.Softmax(-1), 'proxyless'
    )

    optimizer = torch.optim.SGD(op.arch_parameters(), 0.1)

    for _ in range(10):
        x = torch.randn(1, 3, 9, 9).requires_grad_()
        assert op.resample({})['proxyless'] in OPS_WITH_STRIDE
        y = op(x).sum()
        optimizer.zero_grad()
        y.backward()
        assert op._arch_alpha.grad.abs().sum().item() != 0

    assert op.export({})['proxyless'] in OPS_WITH_STRIDE


def test_proxyless_bp_hook_once():

    class DistributedModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.op = ProxylessMixedLayer({
                'linear0': nn.Linear(3, 3, bias=False),
                'linear1': ProxylessMixedLayer({
                    'linear10': nn.Linear(3, 3, bias=False),
                    'linear11': nn.Linear(3, 3, bias=False),
                }, nn.Parameter(torch.randn(2)), nn.Softmax(-1), 'linear1'),
            }, nn.Parameter(torch.randn(2)), nn.Softmax(-1), 'linear')

        def configure_optimizers(self):
            return torch.optim.SGD(self.op.arch_parameters(), 1e-3)

        def training_step(self, batch):
            x, = batch
            self.op.resample({})
            self.op['linear1'].resample({})
            y = self.op(x).sum()
            return y

    dataset = TensorDataset(torch.randn(20, 3))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    trainer = Trainer(
        max_epochs=1,
        accelerator='cpu', devices=1, num_nodes=1, strategy='ddp_find_unused_parameters_true',
        use_distributed_sampler=False,
    )

    trainer.fit(DistributedModule(), dataloader)


def test_proxyless_input():
    inp = ProxylessMixedInput(6, 2, nn.Parameter(torch.zeros(6)), nn.Softmax(-1), 'proxyless')

    optimizer = torch.optim.SGD(inp.arch_parameters(), 0.1)
    for _ in range(10):
        x = [torch.randn(1, 3, 9, 9).requires_grad_() for _ in range(6)]
        assert len(inp.resample({})['proxyless']) == 2
        y = inp(x).sum()
        optimizer.zero_grad()
        y.backward()

    assert all(0 <= x < 6 for x in inp.export({})['proxyless'])


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
        a, b = q + self.d, 2 * k - 2 * self.d
        if v is not None and mask is not None:
            return a, b, v, mask
        elif v is not None:
            return a, b, v
        elif mask is not None:
            return a, b, mask
        return a, b


def test_proxyless_multi_input():
    op = ProxylessMixedLayer(
        {
            'a': MultiInputLayer(1),
            'b': MultiInputLayer(3)
        },
        nn.Parameter(torch.randn(2)),
        nn.Softmax(-1), 'proxyless'
    )

    optimizer = torch.optim.SGD(op.arch_parameters(), 0.1)

    for retry in range(10):
        q = torch.randn(1, 3, 9, 9).requires_grad_()
        k = torch.randn(1, 3, 9, 8).requires_grad_()
        v = None if retry < 3 else torch.randn(1, 3, 9, 7).requires_grad_()
        mask = None if retry < 5 else torch.randn(1, 3, 9, 6).requires_grad_()
        op.resample({})
        if mask is None:
            if v is None:
                y = op(q, k)
            else:
                y = op(q, k, v)
        else:
            y = op(q, k, v, mask)
        y = y[0].sum() + y[1].sum()
        optimizer.zero_grad()
        y.backward()
        assert op._arch_alpha.grad.abs().sum().item() != 0, op._arch_alpha.grad


def test_proxyless_repeat():
    repeat = Repeat(nn.Linear(3, 3), (2, 4), label='rep')
    repeat = ProxylessMixedRepeat.mutate(repeat, '', {}, {})
    assert isinstance(repeat.blocks[1], nn.Linear)
    assert isinstance(repeat.blocks[2], ProxylessMixedLayer)
    memo = {}
    for module in repeat.modules():
        if isinstance(module, ProxylessMixedLayer):
            memo.update(module.resample(memo))
    assert len(memo) == 2
    assert memo['rep/in_repeat_2'] in ['0', '1']
    assert memo['rep/in_repeat_3'] in ['0', '1']
    assert repeat(torch.randn(2, 3)).size(1) == 3
    assert len(repeat.export_probs({})['rep']) == 3
    assert repeat.export({})['rep'] in [2, 3, 4]

    assert repeat.contains({'rep': 3})
    assert not repeat.contains({'rep': 1})
    repeat = repeat.freeze({'rep': 3})
    assert isinstance(repeat, nn.Sequential) and len(repeat) == 3
    assert isinstance(repeat[1], nn.Linear)
    assert isinstance(repeat[2], nn.Linear)
    assert repeat(torch.randn(3, 3)).size(1) == 3


def test_proxyless_repeat_nested():
    repeat = Repeat(
        lambda index: ProxylessMixedLayer(
            {name: value(3, 3, 1) for name, value in OPS_WITH_STRIDE.items()},
            nn.Parameter(torch.randn(len(OPS_WITH_STRIDE))),
            nn.Softmax(-1),
            f'layer{index}'
        ),
        nni.choice('rep', [2, 4, 5])
    )
    repeat = ProxylessMixedRepeat.mutate(repeat, '', {}, {})

    ctrl_params = []
    for m in repeat.modules():
        if isinstance(m, BaseSuperNetModule) and hasattr(m, 'arch_parameters'):
            ctrl_params += list(m.arch_parameters())
    assert len(ctrl_params) > 0
    optimizer = torch.optim.SGD(set(ctrl_params), 1e-3)

    for _ in range(10):
        x = torch.randn(1, 3, 9, 9).requires_grad_()
        memo = {}
        for module in repeat.modules():
            if isinstance(module, ProxylessMixedLayer):
                memo.update(module.resample(memo))

        y = repeat(x).sum()
        optimizer.zero_grad(set_to_none=False)
        y.backward()
        optimizer.step()

    for param in ctrl_params:
        assert param.ndim == 1
        assert param.grad is not None  # Have gradient once.

    assert repeat.export({})['rep'] in [2, 4, 5]

    ops = list(OPS_WITH_STRIDE.keys())
    repeat = repeat.freeze({
        'rep': 4,
        'layer0': random.choice(ops),
        'layer1': random.choice(ops),
        'layer2': random.choice(ops),
        'layer3': random.choice(ops),
    })
    assert isinstance(repeat, nn.Sequential)
    assert not isinstance(repeat[3], ProxylessMixedLayer) and isinstance(repeat[3], nn.Module)
