import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

class Mnist(torch.nn.Module):
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
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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

    model = Mnist()

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

if __name__ == '__main__':
    main()