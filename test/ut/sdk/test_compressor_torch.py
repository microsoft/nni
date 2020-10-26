# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest import TestCase, main
import numpy as np
import torch
import torch.nn.functional as F
import schema
import nni.algorithms.compression.pytorch.pruning as torch_pruner
import nni.algorithms.compression.pytorch.quantization as torch_quantizer
import math


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.conv2 = torch.nn.Conv2d(5, 10, 5, 1)
        self.bn2 = torch.nn.BatchNorm2d(10)
        self.fc1 = torch.nn.Linear(4 * 4 * 10, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CompressorTestCase(TestCase):
    def test_torch_quantizer_modules_detection(self):
        # test if modules can be detected
        model = TorchModel()
        config_list = [{
            'quant_types': ['weight'],
            'quant_bits': 8,
            'op_types': ['Conv2d', 'Linear']
        }, {
            'quant_types': ['output'],
            'quant_bits': 8,
            'quant_start_step': 0,
            'op_types': ['ReLU']
        }]

        model.relu = torch.nn.ReLU()
        quantizer = torch_quantizer.QAT_Quantizer(model, config_list)
        quantizer.compress()
        modules_to_compress = quantizer.get_modules_to_compress()
        modules_to_compress_name = [t[0].name for t in modules_to_compress]
        assert "conv1" in modules_to_compress_name
        assert "conv2" in modules_to_compress_name
        assert "fc1" in modules_to_compress_name
        assert "fc2" in modules_to_compress_name
        assert "relu" in modules_to_compress_name
        assert len(modules_to_compress_name) == 5

    def test_torch_level_pruner(self):
        model = TorchModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        configure_list = [{'sparsity': 0.8, 'op_types': ['default']}]
        torch_pruner.LevelPruner(model, configure_list, optimizer).compress()

    def test_torch_naive_quantizer(self):
        model = TorchModel()
        configure_list = [{
            'quant_types': ['weight'],
            'quant_bits': {
                'weight': 8,
            },
            'op_types': ['Conv2d', 'Linear']
        }]
        torch_quantizer.NaiveQuantizer(model, configure_list).compress()

    def test_torch_fpgm_pruner(self):
        """
        With filters(kernels) weights defined as above (w), it is obvious that w[4] and w[5] is the Geometric Median
        which minimize the total geometric distance by defination of Geometric Median in this paper:
        Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration,
        https://arxiv.org/pdf/1811.00250.pdf

        So if sparsity is 0.2, the expected masks should mask out w[4] and w[5], this can be verified through:
        `all(torch.sum(masks, (1, 2, 3)).numpy() == np.array([125., 125., 125., 125., 0., 0., 125., 125., 125., 125.]))`

        If sparsity is 0.6, the expected masks should mask out w[2] - w[7], this can be verified through:
        `all(torch.sum(masks, (1, 2, 3)).numpy() == np.array([125., 125., 0., 0., 0., 0., 0., 0., 125., 125.]))`
        """
        w = np.array([np.ones((5, 5, 5)) * (i+1) for i in range(10)]).astype(np.float32)

        model = TorchModel()
        config_list = [{'sparsity': 0.6, 'op_types': ['Conv2d']}, {'sparsity': 0.2, 'op_types': ['Conv2d']}]
        pruner = torch_pruner.FPGMPruner(model, config_list, torch.optim.SGD(model.parameters(), lr=0.01))

        model.conv2.module.weight.data = torch.tensor(w).float()
        masks = pruner.calc_mask(model.conv2)
        assert all(torch.sum(masks['weight_mask'], (1, 2, 3)).numpy() == np.array([125., 125., 125., 125., 0., 0., 125., 125., 125., 125.]))

        model.conv2.module.weight.data = torch.tensor(w).float()
        model.conv2.if_calculated = False
        model.conv2.config = config_list[0]
        masks = pruner.calc_mask(model.conv2)
        assert all(torch.sum(masks['weight_mask'], (1, 2, 3)).numpy() == np.array([125., 125., 0., 0., 0., 0., 0., 0., 125., 125.]))

       
    def test_torch_l1filter_pruner(self):
        """
        Filters with the minimum sum of the weights' L1 norm are pruned in this paper:
        PRUNING FILTERS FOR EFFICIENT CONVNETS,
        https://arxiv.org/abs/1608.08710

        So if sparsity is 0.2 for conv1, the expected masks should mask out filter 0, this can be verified through:
        `all(torch.sum(mask1, (1, 2, 3)).numpy() == np.array([0., 25., 25., 25., 25.]))`

        If sparsity is 0.6 for conv2, the expected masks should mask out filter 0,1,2, this can be verified through:
        `all(torch.sum(mask2, (1, 2, 3)).numpy() == np.array([0., 0., 0., 0., 0., 0., 125., 125., 125., 125.]))`
        """
        w1 = np.array([np.ones((1, 5, 5))*i for i in range(5)]).astype(np.float32)
        w2 = np.array([np.ones((5, 5, 5))*i for i in range(10)]).astype(np.float32)

        model = TorchModel()
        config_list = [{'sparsity': 0.2, 'op_types': ['Conv2d'], 'op_names': ['conv1']},
                       {'sparsity': 0.6, 'op_types': ['Conv2d'], 'op_names': ['conv2']}]
        pruner = torch_pruner.L1FilterPruner(model, config_list)

        model.conv1.module.weight.data = torch.tensor(w1).float()
        model.conv2.module.weight.data = torch.tensor(w2).float()
        mask1 = pruner.calc_mask(model.conv1)
        mask2 = pruner.calc_mask(model.conv2)
        assert all(torch.sum(mask1['weight_mask'], (1, 2, 3)).numpy() == np.array([0., 25., 25., 25., 25.]))
        assert all(torch.sum(mask2['weight_mask'], (1, 2, 3)).numpy() == np.array([0., 0., 0., 0., 0., 0., 125., 125., 125., 125.]))

    def test_torch_slim_pruner(self):
        """
        Scale factors with minimum l1 norm in the BN layers are pruned in this paper:
        Learning Efficient Convolutional Networks through Network Slimming,
        https://arxiv.org/pdf/1708.06519.pdf

        So if sparsity is 0.2, the expected masks should mask out channel 0, this can be verified through:
        `all(mask1.numpy() == np.array([0., 1., 1., 1., 1.]))`
        `all(mask2.numpy() == np.array([0., 1., 1., 1., 1.]))`

        If sparsity is 0.6, the expected masks should mask out channel 0,1,2, this can be verified through:
        `all(mask1.numpy() == np.array([0., 0., 0., 1., 1.]))`
        `all(mask2.numpy() == np.array([0., 0., 0., 1., 1.]))`
        """
        w = np.array([0, 1, 2, 3, 4])
        model = TorchModel()
        config_list = [{'sparsity': 0.2, 'op_types': ['BatchNorm2d']}]
        model.bn1.weight.data = torch.tensor(w).float()
        model.bn2.weight.data = torch.tensor(-w).float()
        pruner = torch_pruner.SlimPruner(model, config_list)

        mask1 = pruner.calc_mask(model.bn1)
        mask2 = pruner.calc_mask(model.bn2)
        assert all(mask1['weight_mask'].numpy() == np.array([0., 1., 1., 1., 1.]))
        assert all(mask2['weight_mask'].numpy() == np.array([0., 1., 1., 1., 1.]))
        assert all(mask1['bias_mask'].numpy() == np.array([0., 1., 1., 1., 1.]))
        assert all(mask2['bias_mask'].numpy() == np.array([0., 1., 1., 1., 1.]))

        model = TorchModel()
        config_list = [{'sparsity': 0.6, 'op_types': ['BatchNorm2d']}]
        model.bn1.weight.data = torch.tensor(w).float()
        model.bn2.weight.data = torch.tensor(w).float()
        pruner = torch_pruner.SlimPruner(model, config_list)

        mask1 = pruner.calc_mask(model.bn1)
        mask2 = pruner.calc_mask(model.bn2)
        assert all(mask1['weight_mask'].numpy() == np.array([0., 0., 0., 1., 1.]))
        assert all(mask2['weight_mask'].numpy() == np.array([0., 0., 0., 1., 1.]))
        assert all(mask1['bias_mask'].numpy() == np.array([0., 0., 0., 1., 1.]))
        assert all(mask2['bias_mask'].numpy() == np.array([0., 0., 0., 1., 1.]))

    def test_torch_taylorFOweight_pruner(self):
        """
        Filters with the minimum importance approxiamtion based on the first order 
        taylor expansion on the weights (w*grad)**2 are pruned in this paper:
        Importance Estimation for Neural Network Pruning,
        http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf

        So if sparsity of conv1 is 0.2, the expected masks should mask out filter 0, this can be verified through:
        `all(torch.sum(mask1['weight_mask'], (1, 2, 3)).numpy() == np.array([0., 25., 25., 25., 25.]))`

        If sparsity of conv2 is 0.6, the expected masks should mask out filter 4,5,6,7,8,9 this can be verified through:
        `all(torch.sum(mask2['weight_mask'], (1, 2, 3)).numpy() == np.array([125., 125., 125., 125., 0., 0., 0., 0., 0., 0., ]))`
        """

        w1 = np.array([np.zeros((1, 5, 5)), np.ones((1, 5, 5)), np.ones((1, 5, 5)) * 2,
                      np.ones((1, 5, 5)) * 3, np.ones((1, 5, 5)) * 4])
        w2 = np.array([[[[i + 1] * 5] * 5] * 5 for i in range(10)[::-1]])

        grad1 = np.array([np.ones((1, 5, 5)) * -1, np.ones((1, 5, 5)) * 1, np.ones((1, 5, 5)) * -1,
                      np.ones((1, 5, 5)) * 1, np.ones((1, 5, 5)) * -1])

        grad2 = np.array([[[[(-1)**i] * 5] * 5] * 5 for i in range(10)])

        config_list = [{'sparsity': 0.2, 'op_types': ['Conv2d'], 'op_names': ['conv1']},
                       {'sparsity': 0.6, 'op_types': ['Conv2d'], 'op_names': ['conv2']}]

        model = TorchModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        pruner = torch_pruner.TaylorFOWeightFilterPruner(model, config_list, optimizer, statistics_batch_num=1)
        
        x = torch.rand((1, 1, 28, 28), requires_grad=True)
        model.conv1.module.weight.data = torch.tensor(w1).float()
        model.conv2.module.weight.data = torch.tensor(w2).float()

        y = model(x)
        y.backward(torch.ones_like(y))

        model.conv1.module.weight.grad.data = torch.tensor(grad1).float()
        model.conv2.module.weight.grad.data = torch.tensor(grad2).float()
        optimizer.step()

        mask1 = pruner.calc_mask(model.conv1)
        mask2 = pruner.calc_mask(model.conv2)
        assert all(torch.sum(mask1['weight_mask'], (1, 2, 3)).numpy() == np.array([0., 25., 25., 25., 25.]))
        assert all(torch.sum(mask2['weight_mask'], (1, 2, 3)).numpy() == np.array([125., 125., 125., 125., 0., 0., 0., 0., 0., 0., ]))

    def test_torch_QAT_quantizer(self):
        model = TorchModel()
        config_list = [{
            'quant_types': ['weight'],
            'quant_bits': 8,
            'op_types': ['Conv2d', 'Linear']
        }, {
            'quant_types': ['output'],
            'quant_bits': 8,
            'quant_start_step': 0,
            'op_types': ['ReLU']
        }]
        model.relu = torch.nn.ReLU()
        quantizer = torch_quantizer.QAT_Quantizer(model, config_list)
        quantizer.compress()

        # test quantize
        # range not including 0
        eps = 1e-7
        weight = torch.tensor([[1, 2], [3, 5]]).float()
        model.conv2.module.old_weight.data = weight
        quantizer.quantize_weight(model.conv2)
        assert math.isclose(model.conv2.module.scale, 5 / 255, abs_tol=eps)
        assert model.conv2.module.zero_point == 0
        # range including 0
        weight = torch.tensor([[-1, 2], [3, 5]]).float()
        model.conv2.module.old_weight.data = weight
        quantizer.quantize_weight(model.conv2)
        assert math.isclose(model.conv2.module.scale, 6 / 255, abs_tol=eps)
        assert model.conv2.module.zero_point in (42, 43)
        # test value of weight and bias after quantization
        weight = torch.tensor([[1.1287, 2.3456], [3.7814, 5.9723]])
        weight_valid = torch.tensor([[1.1242, 2.3421], [3.7707, 5.9723]])
        bias = torch.tensor([2.3432, 3.4342, 1.3414, 5.2341])
        bias_valid = torch.tensor([2.3432, 3.4342, 1.3414, 5.2341])
        model.conv2.module.old_weight.data = weight
        model.conv2.module.bias.data = bias
        quantizer.quantize_weight(model.conv2)
        assert torch.all(torch.isclose(model.conv2.module.weight.data, weight_valid, rtol=1e-4))
        assert torch.all(torch.isclose(model.conv2.module.bias.data, bias_valid, rtol=1e-7))

        # test ema
        eps = 1e-7
        x = torch.tensor([[-0.2, 0], [0.1, 0.2]])
        out = model.relu(x)
        assert math.isclose(model.relu.module.tracked_min_biased, 0, abs_tol=eps)
        assert math.isclose(model.relu.module.tracked_max_biased, 0.002, abs_tol=eps)

        quantizer.step_with_optimizer()
        x = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
        out = model.relu(x)
        assert math.isclose(model.relu.module.tracked_min_biased, 0.002, abs_tol=eps)
        assert math.isclose(model.relu.module.tracked_max_biased, 0.00998, abs_tol=eps)

    def test_torch_pruner_validation(self):
        # test bad configuraiton
        pruner_classes = [torch_pruner.__dict__[x] for x in \
            ['LevelPruner', 'SlimPruner', 'FPGMPruner', 'L1FilterPruner', 'L2FilterPruner', 'AGPPruner',\
            'ActivationMeanRankFilterPruner', 'ActivationAPoZRankFilterPruner']]

        bad_configs = [
            [
                {'sparsity': '0.2'},
                {'sparsity': 0.6 }
            ],
            [
                {'sparsity': 0.2},
                {'sparsity': 1.6 }
            ],
            [
                {'sparsity': 0.2, 'op_types': 'default'},
                {'sparsity': 0.6 }
            ],
            [
                {'sparsity': 0.2 },
                {'sparsity': 0.6, 'op_names': 'abc' }
            ]
        ]
        model = TorchModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for pruner_class in pruner_classes:
            for config_list in bad_configs:
                try:
                    pruner_class(model, config_list, optimizer)
                    print(config_list)
                    assert False, 'Validation error should be raised for bad configuration'
                except schema.SchemaError:
                    pass
                except:
                    print('FAILED:', pruner_class, config_list)
                    raise

    def test_torch_quantizer_validation(self):
        # test bad configuraiton
        quantizer_classes = [torch_quantizer.__dict__[x] for x in \
            ['NaiveQuantizer', 'QAT_Quantizer', 'DoReFaQuantizer', 'BNNQuantizer']]

        bad_configs = [
            [
                {'bad_key': 'abc'}
            ],
            [
                {'quant_types': 'abc'}
            ],
            [
                {'quant_bits': 34}
            ],
            [
                {'op_types': 'default'}
            ],
            [
                {'quant_bits': {'abc': 123}}
            ]
        ]
        model = TorchModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for quantizer_class in quantizer_classes:
            for config_list in bad_configs:
                try:
                    quantizer_class(model, config_list, optimizer)
                    print(config_list)
                    assert False, 'Validation error should be raised for bad configuration'
                except schema.SchemaError:
                    pass
                except:
                    print('FAILED:', quantizer_class, config_list)
                    raise

if __name__ == '__main__':
    main()
