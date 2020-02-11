import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from nni.compression.torch import LotteryTicketPruner

class fc1(nn.Module):

    def __init__(self, num_classes=10):
        super(fc1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(model, train_loader, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for imgs, targets in train_loader:
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
        optimizer.step()
    return train_loss.item()

def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    traindataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    testdataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=60, shuffle=True, num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=60, shuffle=False, num_workers=0, drop_last=True)

    model = fc1().to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1.2e-3)
    criterion = nn.CrossEntropyLoss()

    configure_list = [{
        'prune_iterations': 5,
        'sparsity': 0.96,
        'op_types': ['default']
    }]
    pruner = LotteryTicketPruner(model, configure_list, optimizer)
    pruner.compress()

    #model = nn.DataParallel(model)

    for i in pruner.get_prune_iterations():
        pruner.prune_iteration_start()
        loss = 0
        accuracy = 0
        for epoch in range(10):
            loss = train(model, train_loader, optimizer, criterion)
            accuracy = test(model, test_loader, criterion)
            print('current epoch: {0}, loss: {1}, accuracy: {2}'.format(epoch, loss, accuracy))
        print('prune iteration: {0}, loss: {1}, accuracy: {2}'.format(i, loss, accuracy))
    pruner.export_model('model.pth', 'mask.pth')
