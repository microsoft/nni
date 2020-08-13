# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from nni.nas.tensorflow.utils import AverageMeterGroup, fill_zero_grads

from .mutator import EnasMutator

logger = logging.getLogger(__name__)


log_frequency = 100
entropy_weight = 0.0001
skip_weight = 0.8
baseline_decay = 0.999
child_steps = 500
mutator_lr = 0.00035
mutator_steps = 50
mutator_steps_aggregate = 20
aux_weight = 0.4
test_arc_per_epoch = 1


class EnasTrainer:
    def __init__(self, model, loss, metrics, reward_function, optimizer, batch_size, num_epochs,
                 dataset_train, dataset_valid):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.reward_function = reward_function
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        x, y = dataset_train
        split = int(len(x) * 0.9)
        self.train_set = tf.data.Dataset.from_tensor_slices((x[:split], y[:split]))
        self.valid_set = tf.data.Dataset.from_tensor_slices((x[split:], y[split:]))
        self.test_set = tf.data.Dataset.from_tensor_slices(dataset_valid)

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
        train_loader, valid_loader = self._create_train_loader()

        # Sample model and train
        meters = AverageMeterGroup()

        for step in range(1, child_steps + 1):
            x, y = next(train_loader)
            self.mutator.reset()

            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                if isinstance(logits, tuple):
                    logits, aux_logits = logits
                    aux_loss = self.loss(aux_logits, y)
                else:
                    aux_loss = 0.
                metrics = self.metrics(y, logits)
                loss = self.loss(y, logits) + aux_weight * aux_loss

            grads = tape.gradient(loss, self.model.trainable_weights)
            grads = fill_zero_grads(grads, self.model.trainable_weights)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            metrics['loss'] = tf.reduce_mean(loss).numpy()
            meters.update(metrics)

            if log_frequency and step % log_frequency == 0:
                logger.info("Model Epoch [%d/%d] Step [%d/%d]  %s", epoch + 1,
                            self.num_epochs, step, child_steps, meters)

        # Train sampler (mutator)
        meters = AverageMeterGroup()
        for mutator_step in range(1, mutator_steps + 1):
            grads_list = []
            for step in range(1, mutator_steps_aggregate + 1):
                with tf.GradientTape() as tape:
                    x, y = next(valid_loader)
                    self.mutator.reset()

                    logits = self.model(x, training=False)
                    metrics = self.metrics(y, logits)
                    reward = self.reward_function(y, logits) + entropy_weight * self.mutator.sample_entropy
                    self.baseline = self.baseline * baseline_decay + reward * (1 - baseline_decay)
                    loss = self.mutator.sample_log_prob * (reward - self.baseline)
                    loss += skip_weight * self.mutator.sample_skip_penalty

                    meters.update({
                        'reward': reward,
                        'loss': tf.reduce_mean(loss).numpy(),
                        'ent': self.mutator.sample_entropy.numpy(),
                        'log_prob': self.mutator.sample_log_prob.numpy(),
                        'baseline': self.baseline,
                        'skip': self.mutator.sample_skip_penalty,
                    })

                    cur_step = step + (mutator_step - 1) * mutator_steps_aggregate
                    if log_frequency and cur_step % log_frequency == 0:
                        logger.info("RL Epoch [%d/%d] Step [%d/%d] [%d/%d]  %s", epoch + 1, self.num_epochs,
                                    mutator_step, mutator_steps, step, mutator_steps_aggregate,
                                    meters)

                grads = tape.gradient(loss, self.mutator.trainable_weights)
                grads = fill_zero_grads(grads, self.mutator.trainable_weights)
                grads_list.append(grads)
            total_grads = [tf.math.add_n(weight_grads) for weight_grads in zip(*grads_list)]
            total_grads, _ = tf.clip_by_global_norm(total_grads, 5.0)
            self.mutator_optim.apply_gradients(zip(total_grads, self.mutator.trainable_weights))

    def validate_one_epoch(self, epoch):
        test_loader = self._create_validate_loader()

        for arc_id in range(test_arc_per_epoch):
            meters = AverageMeterGroup()
            for x, y in test_loader:
                self.mutator.reset()
                logits = self.model(x, training=False)
                if isinstance(logits, tuple):
                    logits, _ = logits
                metrics = self.metrics(y, logits)
                loss = self.loss(y, logits)
                metrics['loss'] = tf.reduce_mean(loss).numpy()
                meters.update(metrics)

            logger.info("Test Epoch [%d/%d] Arc [%d/%d] Summary  %s",
                        epoch + 1, self.num_epochs, arc_id + 1, test_arc_per_epoch,
                        meters.summary())


    def _create_train_loader(self):
        train_set = self.train_set.shuffle(1000000).repeat().batch(self.batch_size)
        test_set = self.valid_set.shuffle(1000000).repeat().batch(self.batch_size)
        return iter(train_set), iter(test_set)

    def _create_validate_loader(self):
        return iter(self.test_set.shuffle(1000000).batch(self.batch_size))
