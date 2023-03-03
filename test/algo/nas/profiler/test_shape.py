import re
from functools import partial

import torch

from nni.nas.hub.pytorch import *
from nni.nas.profiler.pytorch.utils import *


def _real_shape_inference(model, *args, **kwargs):
    shapes = {}

    def _save_hook(module, input, output, name):
        try:
            input_shapes = tuple(i.shape for i in input) if isinstance(input, tuple) else input.shape
            output_shapes = tuple(o.shape for o in output) if isinstance(output, tuple) else output.shape

            shapes[name] = (input_shapes, output_shapes)
        except:
            pass

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(partial(_save_hook, name=name)))

    model(*args, **kwargs)

    for hook in hooks:
        hook.remove()

    return shapes


def test_proxylessnas():
    net = ProxylessNAS()
    input = ShapeTensor(torch.randn(2, 3, 224, 224), True)
    output = shape_inference(net, input)
    assert output.real_shape == MutableShape(2, 1000)

    sample = {}
    real_model = net.random(sample)
    shapes = submodule_input_output_shapes(net, input)
    input_shapes = {k: v[0][0].freeze(sample) for k, v in shapes.items()}  # unwrap tuple
    output_shapes = {k: v[1].freeze(sample) for k, v in shapes.items()}

    expected_shapes = _real_shape_inference(real_model, torch.randn(2, 3, 224, 224))
    expected_input_shapes = {k: v[0][0] for k, v in expected_shapes.items()}
    expected_output_shapes = {k: v[1] for k, v in expected_shapes.items()}

    left_keys = set(expected_input_shapes)
    for k in input_shapes:
        if k in expected_input_shapes and k in left_keys:
            assert input_shapes[k] == expected_input_shapes[k]
            assert output_shapes[k] == expected_output_shapes[k]
            left_keys.remove(k)
        elif k.startswith('blocks.'):
            k_ = 'blocks.' + '.'.join([d for d in k.split('.') if d.isdigit()])
            if k_ in left_keys and output_shapes[k] == expected_output_shapes[k_] and input_shapes[k] == expected_input_shapes[k_]:
                left_keys.remove(k_)
    assert not left_keys


def test_mobilenet_v3():
    net = MobileNetV3Space()
    input = ShapeTensor(torch.randn(2, 3, 224, 224), True)
    output = shape_inference(net, input)
    assert output.real_shape == MutableShape(2, 1000)

    sample = {}
    real_model = net.random(sample)
    shapes = submodule_input_output_shapes(net, input)
    input_shapes = {k: v[0][0].freeze(sample) for k, v in shapes.items()}  # unwrap tuple
    output_shapes = {k: v[1].freeze(sample) for k, v in shapes.items()}

    expected_shapes = _real_shape_inference(real_model, torch.randn(2, 3, 224, 224))
    expected_input_shapes = {k: v[0][0] for k, v in expected_shapes.items()}
    expected_output_shapes = {k: v[1] for k, v in expected_shapes.items()}

    for k in input_shapes:
        if k in expected_input_shapes:
            assert input_shapes[k] == expected_input_shapes[k]
            assert output_shapes[k] == expected_output_shapes[k]
        # could be blocks.x.blocks.y. Rewrite it as blocks.x.y
        elif re.match(r'blocks\.\d+\.blocks\.\d+', k):
            k_ = re.sub(r'blocks\.(\d+)\.blocks\.(\d+)', r'blocks.\1.\2', k)
            if k_ in expected_input_shapes:
                assert input_shapes[k] == expected_input_shapes[k_]
                assert output_shapes[k] == expected_output_shapes[k_]
        else:
            assert False, f'Unexpected key {k}'


def test_autoformer():
    net = AutoFormer()
    input = ShapeTensor(torch.randn(1, 3, 224, 224), True)
    output = shape_inference(net, input)
    assert output.real_shape == MutableShape(1, 1000)

    sample = {}
    real_model = net.random(sample)
    shapes = submodule_input_output_shapes(net, input)
    input_shapes = {k: v[0][0].freeze(sample) for k, v in shapes.items()}
    output_shapes = {k: v[1].freeze(sample) for k, v in shapes.items()}

    expected_shapes = _real_shape_inference(real_model, torch.randn(1, 3, 224, 224))
    expected_input_shapes = {k: v[0][0] for k, v in expected_shapes.items()}
    expected_output_shapes = {k: v[1] for k, v in expected_shapes.items()}

    for k, v in input_shapes.items():
        k_ = k.replace('blocks.blocks.', 'blocks.')
        if k_ in expected_input_shapes:
            assert v == expected_input_shapes[k_]
            assert output_shapes[k] == expected_output_shapes[k_]


def test_shufflenet():
    net = ShuffleNetSpace(channel_search=True)
    input = ShapeTensor(torch.randn(1, 3, 224, 224), True)
    output = shape_inference(net, input)
    shapes = submodule_input_output_shapes(net, input)
    assert output.real_shape == MutableShape(1, 1000)

    sample = {}
    real_model = net.random(sample)
    shapes = submodule_input_output_shapes(net, input)
    input_shapes = {k: v[0][0].freeze(sample) for k, v in shapes.items()}
    output_shapes = {k: v[1].freeze(sample) for k, v in shapes.items()}

    expected_shapes = _real_shape_inference(real_model, torch.randn(1, 3, 224, 224))
    expected_input_shapes = {k: v[0][0] for k, v in expected_shapes.items()}
    expected_output_shapes = {k: v[1] for k, v in expected_shapes.items()}

    for k in expected_input_shapes:
        if k in input_shapes:
            assert input_shapes[k] == expected_input_shapes[k]
            assert output_shapes[k] == expected_output_shapes[k]
        else:
            kpre, kpre2, kpost = k.split('.', 2)  # features 0 branch_main.1
            kpre = kpre + '.' + kpre2
            for k_ in input_shapes:
                if k_.startswith(kpre) and k_.endswith(kpost) and \
                        input_shapes[k_] == expected_input_shapes[k] and \
                        output_shapes[k_] == expected_output_shapes[k]:
                    break
            else:
                assert False, f'{k} not found'
