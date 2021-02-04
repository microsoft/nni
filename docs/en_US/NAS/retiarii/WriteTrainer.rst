Customize A New Trainer
=======================

Trainers are necessary to evaluate the performance of new explored models. In NAS scenario, this further divides into two use cases:

1. **Classic trainers**: trainers that are used to train and evaluate one single model.
2. **One-shot trainers**: trainers that handle training and searching simultaneously, from an end-to-end perspective.

Classic trainers
----------------

It's recommended to use PyTorch-Lightning...

All classic trainers need to inherit ``nni.retiarii.trainer.BaseTrainer``, implement the ``fit`` method and decorated with ``@register_trainer`` if it is intended to be used together with Retiarii. The decorator serialize the trainer that is used and its argument to fit for the requirements of NNI.

The init function of trainer should take model as its first argument, and the rest of the arguments should be named (``*args`` and ``**kwargs`` may not work as expected) and JSON serializable. This means, currently, passing a complex object like ``torchvision.datasets.ImageNet()`` is not supported. Trainer should use NNI standard API to communicate with tuning algorithms. This includes ``nni.report_intermediate_result`` for periodical metrics and ``nni.report_final_result`` for final metrics.

An example is as follows:

.. code-block::python

    from nni.retiarii import register_trainer
    from nni.retiarii.trainer import BaseTrainer

    @register_trainer
    class MnistTrainer(BaseTrainer):
        def __init__(self, model, optimizer_class_name='SGD', learning_rate=0.1):
            super().__init__()
            self.model = model
            self.criterion = nn.CrossEntropyLoss()
            self.train_dataset = MNIST(train=True)
            self.valid_dataset = MNIST(train=False)
            self.optimizer = getattr(torch.optim, optimizer_class_name)(lr=learning_rate)

        def validate():
            pass

        def fit(self) -> None:
            for i in range(10):  # number of epochs:
                for x, y in DataLoader(self.dataset):
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    loss.backward()
                    self.optimizer.step()
            acc = self.validate()  # get validation accuracy
            nni.report_final_result(acc)

One-shot trainers
-----------------

One-shot trainers should inheirt ``nni.retiarii.trainer.BaseOneShotTrainer``, and need to implement ``fit()`` (used to conduct the fitting and searching process) and ``export()`` method (used to return the searched best architecture).

Writing a one-shot trainer is very different to classic trainers. First of all, there are no more restrictions on init method arguments, any Python arguments are acceptable. Secondly, the model feeded into one-shot trainers might be a model with Retiarii-specific modules, such as LayerChoice and InputChoice. Such model cannot directly forward-propagate and trainers need to decide how to handle those modules.

A typical example is DartsTrainer, where learnable-parameters are used to combine multiple choices in LayerChoice. Retiarii provides ease-to-use utility functions for module-replace purposes, namely ``replace_layer_choice``, ``replace_input_choice``. A simplified example is as follows: 

.. code-block::python

    from nni.retiarii.trainer.pytorch import BaseOneShotTrainer
    from nni.retiarii.trainer.pytorch.utils import replace_layer_choice, replace_input_choice


    class DartsLayerChoice(nn.Module):
        def __init__(self, layer_choice):
            super(DartsLayerChoice, self).__init__()
            self.name = layer_choice.key
            self.op_choices = nn.ModuleDict(layer_choice.named_children())
            self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

        def forward(self, *args, **kwargs):
            op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
            alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
            return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)


    class DartsTrainer(BaseOneShotTrainer):

        def __init__(self, model, loss, metrics, optimizer):
            self.model = model
            self.loss = loss
            self.metrics = metrics
            self.num_epochs = 10

            self.nas_modules = []
            replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)

            ... # init dataloaders and optimizers

        def fit(self):
            for i in range(self.num_epochs):
                for (trn_X, trn_y), (val_X, val_y) in zip(self.train_loader, self.valid_loader):
                    self.train_architecture(val_X, val_y)
                    self.train_model_weight(trn_X, trn_y)

        @torch.no_grad()
        def export(self):
            result = dict()
            for name, module in self.nas_modules:
                if name not in result:
                    result[name] = select_best_of_module(module)
            return result

The full code of DartsTrainer is available to Retiarii source code. Please have a check at :githublink:`nni/retiarii/trainer/pytorch/darts.py`.
