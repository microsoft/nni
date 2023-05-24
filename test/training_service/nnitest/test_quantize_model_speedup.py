# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from torchvision import datasets, transforms
import unittest
from unittest import TestCase, main

from nni.compression.quantization import QATQuantizer
from nni.compression.quantization_speedup import ModelSpeedupTensorRT

torch.manual_seed(0)

class BackboneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
        self.relu1 = torch.nn.ReLU6()
        self.relu2 = torch.nn.ReLU6()
        self.relu3 = torch.nn.ReLU6()
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, x.size()[1:].numel())
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class QuantizationSpeedupTestCase(TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=trans),
            batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=trans),
            batch_size=1000, shuffle=True)

    def _train(self, model, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(self.train_loader), loss.item()))

    def _test(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)

        print('Loss: {}  Accuracy: {}%)\n'.format(
            test_loss, 100 * correct / len(self.test_loader.dataset)))

    def _test_trt(self, engine):
        test_loss = 0
        correct = 0
        time_elasped = 0
        for data, target in self.test_loader:
            output, time = engine.inference(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            time_elasped += time
        test_loss /= len(self.test_loader.dataset)

        print('Loss: {}  Accuracy: {}%'.format(
            test_loss, 100 * correct / len(self.test_loader.dataset)))
        print("Inference elapsed_time (whole dataset): {}s".format(time_elasped))

    def test_post_training_quantization_speedup(self):
        model = BackboneModel()

        configure_list = {
            'conv1':{'weight_bits':8, 'output_bits':8},
            'conv2':{'weight_bits':32, 'output_bits':32},
            'fc1':{'weight_bits':16, 'output_bits':16},
            'fc2':{'weight_bits':8, 'output_bits':8}
        }

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        model.to(self.device)
        for epoch in range(1):
            print('# Epoch {} #'.format(epoch))
            self._train(model, optimizer)
            self._test(model)

        batch_size = 32
        input_shape = (batch_size, 1, 28, 28)
        calibration_path = "calibration.cache"
        onnx_path = "default_model.onnx"

        engine = ModelSpeedupTensorRT(model, input_shape, config=configure_list, calib_data_loader=self.train_loader, batchsize=batch_size)
        engine.compress()
        self._test_trt(engine)
        os.remove(calibration_path)
        os.remove(onnx_path)
    
    def test_qat_quantization_speedup(self):
        model = BackboneModel()

        configure_list = [{
                'quant_types': ['input', 'weight'],
                'quant_bits': {'input':8, 'weight':8},
                'op_names': ['conv1']
            }, {
                'quant_types': ['output'],
                'quant_bits': {'output':8},
                'op_names': ['relu1']
            }, {
                'quant_types': ['input', 'weight'],
                'quant_bits': {'input':8, 'weight':8},
                'op_names': ['conv2']
            }, {
                'quant_types': ['output'],
                'quant_bits': {'output':8},
                'op_names': ['relu2']
            }
        ]

        # finetune the model by using QAT
        dummy_input = torch.randn(1, 1, 28, 28)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        quantizer = QATQuantizer(model, configure_list, optimizer, dummy_input)
        quantizer.compress()

        model.to(self.device)
        for epoch in range(1):
            print('# Epoch {} #'.format(epoch))
            self._train(model, optimizer)
            self._test(model)

        model_path = "mnist_model.pth"
        calibration_path = "mnist_calibration.pth"
        calibration_config = quantizer.export_model(model_path, calibration_path)

        self._test(model)

        print("calibration_config: ", calibration_config)

        batch_size = 32
        input_shape = (batch_size, 1, 28, 28)

        engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=batch_size)
        engine.compress()

        self._test_trt(engine)

        os.remove(model_path)
        os.remove(calibration_path)
    
    def test_export_load_quantized_model_vgg16(self):
        model = vgg16()

        configure_list = {
            'features.0':{'weight_bits':8, 'output_bits':8},
            'features.1':{'weight_bits':32, 'output_bits':32},
            'features.2':{'weight_bits':16, 'output_bits':16},
            'features.4':{'weight_bits':8, 'output_bits':8},
            'features.7':{'weight_bits':8, 'output_bits':8},
            'features.8':{'weight_bits':8, 'output_bits':8},
            'features.11':{'weight_bits':8, 'output_bits':8}
        }

        model.to(self.device)

        batch_size = 1
        input_shape = (batch_size, 3, 224, 224)
        dummy_input = torch.randn(input_shape).to(self.device)

        output_torch = model(dummy_input)

        engine = ModelSpeedupTensorRT(model, input_shape, config=configure_list, calib_data_loader=dummy_input, batchsize=batch_size)
        engine.compress()
        output, _ = engine.inference(dummy_input)

        # verify result shape
        assert(output.shape == output_torch.shape)

        export_path = "vgg16_trt.engine"
        calibration_path = "calibration.cache"
        engine.export_quantized_model(export_path)
        engine.load_quantized_model(export_path)
        output, _ = engine.inference(dummy_input)

        assert(output.shape == output_torch.shape)

        os.remove(export_path)
        os.remove(calibration_path)
    
if __name__ == '__main__':
    main()
