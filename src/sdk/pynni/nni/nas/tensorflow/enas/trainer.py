# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import logging
from itertools import cycle

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from .mutator import EnasMutator

logger = logging.getLogger(__name__)


#workers = 4
log_frequency = 10
entropy_weight = 0.0001
skip_weight = 0.8
baseline_decay = 0.999
child_steps = 500
mutator_lr = 0.00035
mutator_steps_aggregate = 20
mutator_steps = 50
test_arc_per_epoch = 1


class EnasTrainer:
    def __init__(self, model, loss, metrics, reward_function, optimizer, batch_size, num_epochs,
                 dataset_train, dataset_valid, dataset_test):  # FIXME: how to split a built Dataset?
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.reward_function = reward_function
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_loader = iter(dataset_train)
        self.valid_loader = iter(dataset_valid)
        self.test_loader = iter(dataset_test)

        self.mutator = EnasMutator(model)
        self.mutator_optim = Adam(learning_rate=mutator_lr)

        self.baseline = 0.


    def train(self, validate=True):
        for epoch in range(self.num_epochs):
            logger.info("Epoch %d Training", epoch + 1)
            self.train_one_epoch(epoch)
            logger.info("Epoch %d Validating", epoch + 1)
            self.validate_one_epoch(epoch)


    def validate(self):
        self.validate_one_epoch(-1)


    def train_one_epoch(self, epoch):
        # Sample model and train

        meters = []
        for step in range(1, child_steps + 1):
            x, y = next(train_loader)
            with tf.GradientTape() as tape:
                tape.watch(x)
                logits = self.model(x, training=True)

                if isinstance(logits, tuple):
                    logits, aux_logits = logits
                    aux_loss = self.loss(aux_logits, y)
                else:
                    aux_loss = 0.
                metrics = self.metrics(logits, y)
                loss = self.loss(logits, y)
                loss = loss + self.aux_weight * aux_loss
            grads = tape.gradient(loss, self.model.trainable_weights)
            grads = tf.clip_by_norm(grads, 5.)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            meters.append(loss[0])

            if log_frequency and step % log_frequency == 0:
                logger.info("Model Epoch [%d/%d] Step [%d/%d]  loss %s", epoch + 1,
                            self.num_epochs, step, child_steps, sum(meters) / len(meters))

        # Train sampler (mutator)
        meters = defaultdict(list)
        for mutator_step in range(1, mutator_steps + 1):
            with GradientTape() as tape:
                tape.watch(self.baseline)
                for step in range(1, mutator_steps_aggregate + 1):
                    x, y = next(valid_loader)

                    self.mutator.reset()
                    logits = self.model(x, training=False)
                    metrics = self.metrics(logits, y)
                    reward = self.reward_function(logits, y)
                    tape.watch(reward)
                    reward += entropy_weight * self.mutator.sample_entropy[0]
                    self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
                    loss = self.mutator.sample_log_prob * (reward - self.baseline)
                    loss += skip_weight * self.mutator.sample_skip_penalty
                    loss /= mutator_steps_aggregate

                    meters["reward"].append(reward)
                    meters["loss"].append(loss)
                    meters["ent"].append(self.mutator.sample_entropy)
                    meters["log_prob"].append(self.mutator.sample_log_prob)
                    meters["baseline"].append(self.baseline)
                    meters["skip"].append(self.mutator.sample_skip_penalty)

                    cur_step = step + (mutator_step - 1) * mutator_steps_aggregate
                    if log_frequency and cur_step % log_frequency == 0:
                        logger.info("RL Epoch [%d/%d] Step [%d/%d] [%d/%d]  ", epoch + 1, self.num_epochs,
                                    mutator_step, mutator_steps, step, mutator_steps_aggregate)

            grads = tape.gradient(loss, self.mutator.trainable_weights)
            grads = tf.clip_by_norm(grads, 5.)
            self.mutator_optim.apply_gradients(zip(grads, self.mutator.trainable_weights))


    def validate_one_epoch(self, epoch):
        for arc_id in range(test_arc_per_epoch):
            loss_list = []
            for x, y in self.test_loader:
                self.mutator.reset()
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits, _ = logits
                metrics = self.metrics(logits, y)
                loss = self.loss(logits, y)
                loss_list.append(loss[0])

            logger.info("Test Epoch [%d/%d] Arc [%d/%d] Average loss %s",
                        epoch + 1, self.num_epochs, arc_id + 1, test_arc_per_epoch,
                        sum(loss_list) / len(loss_list))
