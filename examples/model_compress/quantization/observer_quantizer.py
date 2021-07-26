import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from nni.algorithms.compression.pytorch.quantization import ObserverQuantizer
import sys
sys.path.append('../models')
from mnist.naive import NaiveModel


def train(model, device, train_loader, optimizer):
    model.to(device)
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


def calibration(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            model(data)


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

    model = NaiveModel()
    configure_list = [{
            'quant_types': ['weight', 'input'],
            'quant_bits': {'weight': 8, 'input': 8},
            'op_names': ['conv1'],
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8, },
            'op_names': ['relu1'],
        }, {
            'quant_types': ['weight', 'input'],
            'quant_bits': {'weight': 8, 'input': 8},
            'op_names': ['conv2'],
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_names': ['relu2'],
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_names': ['max_pool2'],
        }
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Train the model to get a baseline performance
    for epoch in range(5):
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer)

    test(model, device, test_loader)

    # Construct the ObserverQuantizer. Note that currently ObserverQuantizer only works
    # in evaluation mode.
    quantizer = ObserverQuantizer(model.eval(), configure_list, optimizer)
    # Use the test data set to do calibration, this will not change the model parameters
    calibration(model, device, test_loader)
    # obtain the quantization information and switch the model to "accuracy verification" mode
    quantizer.compress()

    # measure the accuracy of the quantized model.
    test(model, device, test_loader)

    model_path = "mnist_model.pth"
    calibration_path = "mnist_calibration.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)
    print("calibration_config: ", calibration_config)

    # For now the quantization settings of ObserverQuantizer does not match the TensorRT,
    # so TensorRT conversion are not supported
    # current settings:
    # weight      : per_tensor_symmetric, qint8
    # activation  : per_tensor_affine, quint8, reduce_range=True


if __name__ == '__main__':
    main()
