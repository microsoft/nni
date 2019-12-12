from nni.compression.torch import AGP_Pruner
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


class Mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
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


def main():
    torch.manual_seed(0)
    device = torch.device('cpu')

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=trans),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=trans),
        batch_size=1000, shuffle=True)

    model = Mnist()
    model.to(device)

    '''you can change this to LevelPruner to implement it
    pruner = LevelPruner(configure_list)
    '''
    configure_list = [{
        'initial_sparsity': 0,
        'final_sparsity': 0.8,
        'start_epoch': 0,
        'end_epoch': 10,
        'frequency': 1,
        'op_types': ['default']
    }]

    pruner = AGP_Pruner(model, configure_list)
    model = pruner.compress()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(10):
        pruner.update_epoch(epoch)
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
    pruner.export_model('model.pth', 'mask.pth', 'model.onnx', [1, 1, 28, 28])


if __name__ == '__main__':
    main()
