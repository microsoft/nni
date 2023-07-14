"""
Create Pytorch Native Evaluator
===============================

If you are using a native pytorch training loop to train your model, this example could help you getting start quickly.
In this example, you will learn how to create a pytorch native evaluator step by step.

Prepare ``training_func``
-------------------------

``training_func`` has six required parameters.
Maybe you don't need some parameters such as ``lr_scheduler``, but you still need to reflect the complete six parameters on the interface.

For some reason, ``dataloader`` is not exposed on ``training_func`` as part of the interface,
so it is necessary to directly create or reference an dataloader in ``training_func`` inner.

Here is an simple ``training_func``.
"""

from typing import Any, Callable
import torch
from examples.compression.models import prepare_dataloader

def training_func(model: torch.nn.Module, optimizer: torch.optim.Optimizer, training_step: Callable[[Any, torch.nn.Module], torch.Tensor],
                  lr_scheduler: torch.optim.lr_scheduler._LRScheduler, max_steps: int, max_epochs: int):
    # create a train dataloader (and test dataloader if needs)
    train_dataloader, test_dataloader = prepare_dataloader()

    # deal with training duration, NNI prefers to prioritize the largest number of steps
    # at least `max_steps` or `max_epochs` will be given
    assert max_steps is not None or max_epochs is not None
    total_steps = max_steps if max_steps else max_epochs * len(train_dataloader)
    total_epochs = total_steps // len(train_dataloader) + (0 if total_steps % len(train_dataloader) == 0 else 1)

    # here is a common training loop
    current_step = 0
    for _ in range(total_epochs):
        for batch in train_dataloader:
            loss = training_step(batch, model)
            loss.backward()
            optimizer.step()

            # if reach the total steps, exit from the training loop
            current_step = current_step + 1
            if current_step >= total_steps:
                return

        # if you are using a epoch-wise scheduler, call it here
        lr_scheduler.step()

# %%
# Now we have a basic training function that can generate loss by ``model`` and ``training_step``,
# optimize the model by ``optimizer`` and ``lr_scheduler``, terminate the training loop by ``max_steps`` and ``max_epochs``.
#
# Prepare ``optimizers`` and ``lr_schedulers``
# --------------------------------------------
#
# ``optimizers`` is a required parameter and ``lr_schedulers`` is an optional parameter.
# ``optimizers`` can be a optimizer instance or a list of optimziers and ``lr_schedulers`` can be a lr scheduler instance or a list of lr schedulers or ``None``.
#
# Note that each ``optimizer`` and ``lr_scheduler`` should be a subclass of ``torch.optim.Optimizer`` or ``torch.optim.lr_scheduler._LRScheduler``
# (``torch.optim.lr_scheduler.LRScheduler`` in ``torch >= 2.0``), and the class should be wrapped by ``nni.trace``.
# ``nni.trace`` is important for NNI refreshing the optimizer, because compression will register new module parameters that need to optimize.

import nni
from examples.compression.models import build_resnet18

# create a resnet18 model as an exmaple
model = build_resnet18()

optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
lr_scheduler = nni.trace(torch.optim.lr_scheduler.LambdaLR)(optimizer, lr_lambda=lambda epoch: 1 / epoch)

# %%
# Now we have a traced optimizer and a traced lr scheduler.
#
# Prepare ``training_step``
# -------------------------
#
# Training step should have two required parameters ``batch`` and ``model``,
# return value is a loss tensor or a list with the first element loss or a dict with key ``loss``.

import torch.nn.functional as F

def training_step(batch: Any, model: torch.nn.Module, *args, **kwargs):
    output = model(batch[0])
    loss = F.cross_entropy(output, batch[1])
    return loss

# Init ``TorchEvaluator``
# -----------------------

from nni.compression import TorchEvaluator

evaluator = TorchEvaluator(training_func, optimizer, training_step, lr_scheduler)
