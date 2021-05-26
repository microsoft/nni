import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

import sys
sys.path.append('../models')
from mnist.naive import NaiveModel


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))

def test_trt(engine, test_loader):
    test_loss = 0
    correct = 0
    time_elasped = 0
    for data, target in test_loader:
        output, time = engine.inference(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        time_elasped += time
    test_loss /= len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))
    print("Inference elapsed_time (whole dataset): {}s".format(time_elasped))

def post_training_quantization_example(train_loader, test_loader, device):
    model = NaiveModel()

    config = {
        'conv1':{'weight_bit':8, 'activation_bit':8},
        'conv2':{'weight_bit':32, 'activation_bit':32},
        'fc1':{'weight_bit':16, 'activation_bit':16},
        'fc2':{'weight_bit':8, 'activation_bit':8}
    }

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    model.to(device)
    for epoch in range(1):
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)

    batch_size = 32
    input_shape = (batch_size, 1, 28, 28)

    engine = ModelSpeedupTensorRT(model, input_shape, config=config, calib_data_loader=train_loader, batchsize=batch_size)
    engine.compress()
    test_trt(engine, test_loader)

def quantization_aware_training_example(train_loader, test_loader, device):
    model = NaiveModel()

    configure_list = [{
            'quant_types': ['weight', 'output'],
            'quant_bits': {'weight':8, 'output':8},
            'op_names': ['conv1']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output':8},
            'op_names': ['relu1']
        }, {
            'quant_types': ['weight', 'output'],
            'quant_bits': {'weight':8, 'output':8},
            'op_names': ['conv2']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output':8},
            'op_names': ['relu2']
        }
    ]

    # finetune the model by using QAT
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    quantizer = QAT_Quantizer(model, configure_list, optimizer)
    quantizer.compress()

    model.to(device)
    for epoch in range(1):
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)

    model_path = "mnist_model.pth"
    calibration_path = "mnist_calibration.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)

    test(model, device, test_loader)

    print("calibration_config: ", calibration_config)

    batch_size = 32
    input_shape = (batch_size, 1, 28, 28)

    engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=batch_size)
    engine.compress()

    test_trt(engine, test_loader)

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=trans),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=trans),
        batch_size=1000, shuffle=True)

    # post-training quantization on TensorRT
    post_training_quantization_example(train_loader, test_loader, device)

    # combine NNI quantization algorithm QAT with backend framework TensorRT
    quantization_aware_training_example(train_loader, test_loader, device)

if __name__ == '__main__':
    main()