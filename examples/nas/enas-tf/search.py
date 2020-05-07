# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from nni.nas.tensorflow import enas
import datasets
from macro import GeneralNetwork
from utils import accuracy, reward_accuracy



dataset_train, dataset_valid, dataset_test = datasets.get_dataset("cifar10")
model = GeneralNetwork()

criterion = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
optimizer = SGD(learning_rate=0.05, momentum=0.9)

trainer = enas.EnasTrainer(model,
                           loss=criterion,
                           metrics=accuracy,
                           reward_function=reward_accuracy,
                           optimizer=optimizer,
                           batch_size=64,
                           num_epochs=310,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           dataset_test=dataset_test)
trainer.train()
