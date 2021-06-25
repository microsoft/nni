from typing import Callable, List, Dict

from torch.nn import Module

from pruning_scheduler import PruningScheduler
from sparsity_generator import SimulatedAnnealingSparsityGenerator


class SimulatedAnnealingPruningScheduler(PruningScheduler):
    def __init__(self, model: Module, config_list: List[Dict], evaluator: Callable[[Module], float],
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35):
        sparsity_generator = SimulatedAnnealingSparsityGenerator(model, config_list, evaluator, start_temperature,
                                                                 stop_temperature, cool_down_rate,
                                                                 perturbation_magnitude)
        super().__init__(model, config_list, iteration_num=None, sparsity_generator=sparsity_generator)

    def compress(self, log_dir: str = './log', tag: str = 'default'):
        return super().compress(log_dir=log_dir, tag=tag, consistent=False)


if __name__ == '__main__':
    import functools
    import torch
    from torchvision import datasets, transforms
    from examples.model_compress.models.mnist.lenet import LeNet
    from nni.algorithms.compression.pytorch.pruning.one_shot_pruner import L1FilterPruner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    config_list = [{
        'sparsity': 0.8,
        'op_types': ['Conv2d']
    }]

    def finetuner(model, optimizer, train_loader, epoch):
        model.train()
        criterion = torch.nn.NLLLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def evaluator(model, test_loader):
        model.eval()
        criterion = torch.nn.NLLLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        acc = 100 * correct / len(test_loader.dataset)

        print('Test Loss: {}  Accuracy: {}%\n'.format(
            test_loss, acc))
        return acc

    def get_optimizer(model):
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./log/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./log/data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=200, shuffle=True)

    dummy_input = torch.randn([200, 1, 28, 28]).to(device)

    # pre-training
    best_acc = 0
    for i in range(1):
        finetuner(model, get_optimizer(model), train_loader, i)
        acc = evaluator(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            state_dict = model.state_dict()

    model.load_state_dict(state_dict)

    _evaluator = functools.partial(evaluator, test_loader=test_loader)
    scheduler = SimulatedAnnealingPruningScheduler(model, config_list, _evaluator)
    scheduler.set_pruner(L1FilterPruner)
    scheduler.compress()
    print(scheduler.get_best_config_list())
