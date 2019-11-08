import torch
import torch.optim as optim

from nni.nas.pytorch.trainer import Trainer
from nni.nas.utils import AverageMeterGroup, auto_device
from .mutator import EnasMutator


class EnasTrainer(Trainer):
    def __init__(self, model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset_train, dataset_valid, lr_scheduler=None,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None,
                 entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999,
                 mutator_lr=0.00035):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.reward_function = reward_function
        self.mutator = mutator
        if self.mutator is None:
            self.mutator = EnasMutator(model)
        self.optim = optimizer
        self.mut_optim = optim.Adam(self.mutator.parameters(), lr=mutator_lr)
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.device = auto_device() if device is None else device
        self.log_frequency = log_frequency
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.

        self.model.to(self.device)
        self.loss.to(self.device)
        self.mutator.to(self.device)

        n_train = len(self.dataset_train)
        split = n_train // 10
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:-split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[-split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=workers)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=batch_size,
                                                       num_workers=workers)

    def train_epoch(self, epoch):
        self.model.train()
        self.mutator.train()

        for phase in ["model", "mutator"]:
            if phase == "model":
                self.model.train()
                self.mutator.eval()
            else:
                self.model.eval()
                self.mutator.train()
            loader = self.train_loader if phase == "model" else self.valid_loader
            meters = AverageMeterGroup()
            for step, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optim.zero_grad()
                self.mut_optim.zero_grad()

                with self.mutator.forward_pass():
                    logits = self.model(x)
                metrics = self.metrics(logits, y)

                if phase == "model":
                    loss = self.loss(logits, y)
                    loss.backward()
                    self.optim.step()
                else:
                    reward = self.reward_function(logits, y)
                    if self.entropy_weight is not None:
                        reward += self.entropy_weight * self.mutator.sample_entropy
                    self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
                    self.baseline = self.baseline.detach().item()
                    loss = self.mutator.sample_log_prob * (reward - self.baseline)
                    if self.skip_weight:
                        loss += self.skip_weight * self.mutator.sample_skip_penalty
                    loss.backward()
                    self.mut_optim.step()
                    metrics["reward"] = reward
                metrics["loss"] = loss.item()
                meters.update(metrics)

                if self.log_frequency is not None and step % self.log_frequency == 0:
                    print("Epoch {} {} Step [{}/{}]  {}".format(epoch, phase.capitalize(), step,
                                                                len(loader), meters))
                    # print(self.mutator._selected_layers)
                    # print(self.mutator._selected_inputs)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def validate_epoch(self, epoch):
        pass

    def train(self):
        for epoch in range(self.num_epochs):
            # training
            print("Epoch {} Training".format(epoch))
            self.train_epoch(epoch)

            # validation
            print("Epoch {} Validating".format(epoch))
            self.validate_epoch(epoch)

    def export(self):
        pass
