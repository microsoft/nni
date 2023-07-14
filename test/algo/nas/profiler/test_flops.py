import torch

from nni.compression.utils.counter import count_flops_params
from nni.nas.hub.pytorch import ProxylessNAS, MobileNetV3Space, AutoFormer, ShuffleNetSpace
from nni.nas.profiler.pytorch.flops import FlopsParamsProfiler

def test_proxylessnas():
    net = ProxylessNAS()

    proxy_mobile_arch = {
        's2_depth': 2,
        's2_i0': 'k5e3',
        's2_i1': 'k3e3',
        's3_depth': 4,
        's3_i0': 'k7e3',
        's3_i1': 'k3e3',
        's3_i2': 'k5e3',
        's3_i3': 'k5e3',
        's4_depth': 4,
        's4_i0': 'k7e6',
        's4_i1': 'k5e3',
        's4_i2': 'k5e3',
        's4_i3': 'k5e3',
        's5_depth': 4,
        's5_i0': 'k5e6',
        's5_i1': 'k5e3',
        's5_i2': 'k5e3',
        's5_i3': 'k5e3',
        's6_depth': 4,
        's6_i0': 'k7e6',
        's6_i1': 'k7e6',
        's6_i2': 'k7e3',
        's6_i3': 'k7e3',
        's7_depth': 1,
        's7_i0': 'k7e6'
    }
    model = net.freeze(proxy_mobile_arch)
    from nni.compression.utils.counter import count_flops_params
    flops, params, _ = count_flops_params(model, torch.randn(1, 3, 224, 224))

    # Hack because expression switch doesn't support lazy evaluation.
    sample = {}
    net.default(sample)
    sample.update(proxy_mobile_arch)

    profiler = FlopsParamsProfiler(net, torch.randn(1, 3, 224, 224), count_bias=False, count_normalization=False, count_activation=False)
    result = profiler.profile(sample)
    assert result.flops == flops
    assert result.params == params


def test_mobilenet_v3():
    net = MobileNetV3Space()

    sample = {}
    model = net.random(sample)
    flops, params, _ = count_flops_params(model, torch.randn(1, 3, 224, 224))

    profiler = FlopsParamsProfiler(net, torch.randn(1, 3, 224, 224), count_bias=False, count_normalization=False, count_activation=False)
    result = profiler.profile(sample)
    assert result.flops == flops
    assert result.params == params


def test_autoformer():
    net = AutoFormer()

    sample = {}
    model = net.random(sample)
    flops, params, _ = count_flops_params(model, torch.randn(1, 3, 224, 224))

    profiler = FlopsParamsProfiler(net, torch.randn(1, 3, 224, 224), count_bias=False, count_normalization=False, count_activation=False)
    result = profiler.profile(sample)

    assert 0.9 < result.params / params < 1.1
    # sanity check so far

    model = AutoFormer.load_searched_model('autoformer-tiny')
    profiler = FlopsParamsProfiler(model, torch.randn(1, 3, 224, 224), count_bias=False, count_normalization=False, count_activation=False)
    result = profiler.profile({})
    assert 5.7e6 < result.params < 5.8e6
    assert 1.4e9 < result.flops < 1.5e9


def test_shufflenet():
    net = ShuffleNetSpace()
    profiler = FlopsParamsProfiler(net, torch.randn(1, 3, 224, 224), count_bias=False, count_normalization=False, count_activation=False)

    arch = {
        'layer_1': 'k7',
        'layer_2': 'k5',
        'layer_3': 'k3',
        'layer_4': 'k5',
        'layer_5': 'k7',
        'layer_6': 'k3',
        'layer_7': 'k7',
        'layer_8': 'k3',
        'layer_9': 'k7',
        'layer_10': 'k3',
        'layer_11': 'k7',
        'layer_12': 'xcep',
        'layer_13': 'k3',
        'layer_14': 'k3',
        'layer_15': 'k3',
        'layer_16': 'k3',
        'layer_17': 'xcep',
        'layer_18': 'k7',
        'layer_19': 'xcep',
        'layer_20': 'xcep'
    }

    assert 310e6 < profiler.profile(arch).flops < 330e6
    assert 3.4e6 < profiler.profile(arch).params < 3.6e6
    result = profiler.profile(arch)

    flops, params, _ = count_flops_params(net.freeze(arch), torch.randn(1, 3, 224, 224))
    assert result.flops == flops
    assert result.params == params
