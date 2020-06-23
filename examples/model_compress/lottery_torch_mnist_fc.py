import argparse
import copy
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
    """
    THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS (https://arxiv.org/pdf/1803.03635.pdf)

    The Lottery Ticket Hypothesis. A randomly-initialized, dense neural network contains a subnetwork that is
    initialized such that—when trained in isolation—it can match the test accuracy of the original network after
    training for at most the same number of iterations.

    Identifying winning tickets. We identify a winning ticket by training a network and pruning its
    smallest-magnitude weights. The remaining, unpruned connections constitute the architecture of the
    winning ticket. Unique to our work, each unpruned connection’s value is then reset to its initialization
    from original network before it was trained. This forms our central experiment:
        1. Randomly initialize a neural network f(x; θ0) (where θ0 ∼ Dθ).
        2. Train the network for j iterations, arriving at parameters θj .
        3. Prune p% of the parameters in θj , creating a mask m.
        4. Reset the remaining parameters to their values in θ0, creating the winning ticket f(x; mθ0).
    As described, this pruning approach is one-shot: the network is trained once, p% of weights are
    pruned, and the surviving weights are reset. However, in this paper, we focus on iterative pruning,
    which repeatedly trains, prunes, and resets the network over n rounds; each round prunes p**(1/n) % of
    the weights that survive the previous round. Our results show that iterative pruning finds winning tickets
    that match the accuracy of the original network at smaller sizes than does one-shot pruning.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_epochs", type=int, default=10, help="training epochs")
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    traindataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    testdataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=60, shuffle=True, num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=60, shuffle=False, num_workers=0, drop_last=True)

    model = fc1().to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1.2e-3)
    criterion = nn.CrossEntropyLoss()

    # Record the random intialized model weights
    orig_state = copy.deepcopy(model.state_dict())

    # train the model to get unpruned metrics
    for epoch in range(args.train_epochs):
        train(model, train_loader, optimizer, criterion)
    orig_accuracy = test(model, test_loader, criterion)
    print('unpruned model accuracy: {}'.format(orig_accuracy))

    # reset model weights and optimizer for pruning
    model.load_state_dict(orig_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.2e-3)

    # Prune the model to find a winning ticket
    configure_list = [{
        'prune_iterations': 5,
        'sparsity': 0.96,
        'op_types': ['default']
    }]
    pruner = LotteryTicketPruner(model, configure_list, optimizer)
    pruner.compress()

    best_accuracy = 0.
    best_state_dict = None

    for i in pruner.get_prune_iterations():
        pruner.prune_iteration_start()
        loss = 0
        accuracy = 0
        for epoch in range(args.train_epochs):
            loss = train(model, train_loader, optimizer, criterion)
            accuracy = test(model, test_loader, criterion)
            print('current epoch: {0}, loss: {1}, accuracy: {2}'.format(epoch, loss, accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # state dict of weights and masks
                best_state_dict = copy.deepcopy(model.state_dict())
        print('prune iteration: {0}, loss: {1}, accuracy: {2}'.format(i, loss, accuracy))

    if best_accuracy > orig_accuracy:
        # load weights and masks
        pruner.bound_model.load_state_dict(best_state_dict)
        # reset weights to original untrained model and keep masks unchanged to export winning ticket
        pruner.load_model_state_dict(orig_state)
        pruner.export_model('model_winning_ticket.pth', 'mask_winning_ticket.pth')
        print('winning ticket has been saved: model_winning_ticket.pth, mask_winning_ticket.pth')
    else:
        print('winning ticket is not found in this run, you can run it again.')
