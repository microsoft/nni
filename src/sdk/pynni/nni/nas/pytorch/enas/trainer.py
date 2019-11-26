# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
import torch.optim as optim

from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup
from .mutator import EnasMutator

logger = logging.getLogger(__name__)


class EnasTrainer(Trainer):
    def __init__(self, model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None, callbacks=None,
                 entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999,
                 mutator_lr=0.00035, mutator_steps_aggregate=20, mutator_steps=50, aux_weight=0.4):
        super().__init__(model, mutator if mutator is not None else EnasMutator(model),
                         loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid,
                         batch_size, workers, device, log_frequency, callbacks)
        self.reward_function = reward_function
        self.mutator_optim = optim.Adam(self.mutator.parameters(), lr=mutator_lr)

        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.
        self.mutator_steps_aggregate = mutator_steps_aggregate
        self.mutator_steps = mutator_steps
        self.aux_weight = aux_weight

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

    def train_one_epoch(self, epoch):
        # Sample model and train
        self.model.train()
        self.mutator.eval()
        meters = AverageMeterGroup()
        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                self.mutator.reset()
            logits = self.model(x)

            if isinstance(logits, tuple):
                logits, aux_logits = logits
                aux_loss = self.loss(aux_logits, y)
            else:
                aux_loss = 0.
            metrics = self.metrics(logits, y)
            loss = self.loss(logits, y)
            loss = loss + self.aux_weight * aux_loss
            loss.backward()
            self.optimizer.step()
            metrics["loss"] = loss.item()
            meters.update(metrics)

            if self.log_frequency is not None and step % self.log_frequency == 0:
                logger.info("Model Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), meters)

        # Train sampler (mutator)
        self.model.eval()
        self.mutator.train()
        meters = AverageMeterGroup()
        mutator_step, total_mutator_steps = 0, self.mutator_steps * self.mutator_steps_aggregate
        while mutator_step < total_mutator_steps:
            for step, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)

                self.mutator.reset()
                with torch.no_grad():
                    logits = self.model(x)
                metrics = self.metrics(logits, y)
                reward = self.reward_function(logits, y)
                if self.entropy_weight is not None:
                    reward += self.entropy_weight * self.mutator.sample_entropy
                self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
                self.baseline = self.baseline.detach().item()
                loss = self.mutator.sample_log_prob * (reward - self.baseline)
                if self.skip_weight:
                    loss += self.skip_weight * self.mutator.sample_skip_penalty
                metrics["reward"] = reward
                metrics["loss"] = loss.item()
                metrics["ent"] = self.mutator.sample_entropy.item()
                metrics["baseline"] = self.baseline
                metrics["skip"] = self.mutator.sample_skip_penalty

                loss = loss / self.mutator_steps_aggregate
                loss.backward()
                meters.update(metrics)

                if mutator_step % self.mutator_steps_aggregate == 0:
                    self.mutator_optim.step()
                    self.mutator_optim.zero_grad()

                if self.log_frequency is not None and step % self.log_frequency == 0:
                    logger.info("RL Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1, self.num_epochs,
                                mutator_step // self.mutator_steps_aggregate + 1, self.mutator_steps, meters)
                mutator_step += 1
                if mutator_step >= total_mutator_steps:
                    break

    def validate_one_epoch(self, epoch):
        pass
