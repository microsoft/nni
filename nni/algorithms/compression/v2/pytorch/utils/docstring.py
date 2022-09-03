# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


_EVALUATOR_DOCSTRING = r"""NNI will use the evaluator to intervene in the model training process,
        so as to perform training-aware model compression.
        All training-aware model compression will use the evaluator as the entry for intervention training in the future.
        Usually you just need to wrap some classes with ``nni.trace`` or package the training process as a function to initialize the evaluator.
        Please refer :doc:`/compression/compression_evaluator` for a full tutorial on how to initialize a ``evaluator``.

        The following are two simple examples, if you use pytorch_lightning, please refer to :class:`nni.compression.pytorch.LightningEvaluator`,
        if you use native pytorch, please refer to :class:`nni.compression.pytorch.TorchEvaluator`::

            # LightningEvaluator example
            import pytorch_lightning
            lightning_trainer = nni.trace(pytorch_lightning.Trainer)(max_epochs=1, max_steps=50, logger=TensorBoardLogger(...))
            lightning_data_module = nni.trace(pytorch_lightning.LightningDataModule)(...)

            from nni.compression.pytorch import LightningEvaluator
            evaluator = LightningEvaluator(lightning_trainer, lightning_data_module)

            # TorchEvaluator example
            import torch
            import torch.nn.functional as F

            def training_model(model, optimizer, criterion, lr_scheduler, max_steps, max_epochs, *args, **kwargs):
                # max_steps, max_epochs might be None, which means unlimited training time,
                # so here we need set a default termination condition (by default, total_epochs=10, total_steps=100000).
                total_epochs = max_epochs if max_epochs else 10
                total_steps = max_steps if max_steps else 100000
                current_step = 0

                # init dataloader
                train_dataloader = ...

                for epoch in range(total_epochs):
                    ...
                    for input_data, target in train_dataloader:
                        optimizer.zero_grad()
                        result = model(input_data)
                        loss = criterion(result, target)
                        loss.backward()
                        optimizer.step()
                        current_step += 1
                        if current_step >= total_steps:
                            return
                    lr_scheduler.step()

            traced_optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01)
            criterion = F.nll_loss

            from nni.compression.pytorch import TorchEvaluator
            evaluator = TorchEvaluator(training_func=training_model, optimziers=traced_optimizer, criterion=criterion)
    """
