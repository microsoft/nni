Customize Exploration Strategy
==============================

Customize Multi-trial Strategy
------------------------------

If users want to innovate a new exploration strategy, they can easily customize a new one following the interface provided by NNI. Specifically, users should inherit the base strategy class ``BaseStrategy``, then implement the member function ``run``. This member function takes ``base_model`` and ``applied_mutators`` as its input arguments. It can simply apply the user specified mutators in ``applied_mutators`` onto ``base_model`` to generate a new model. When a mutator is applied, it should be bound with a sampler (e.g., ``RandomSampler``). Every sampler implements the ``choice`` function which chooses value(s) from candidate values. The ``choice`` functions invoked in mutators are executed with the sampler.

Below is a very simple random strategy, which makes the choices completely random.

.. code-block:: python

    from nni.retiarii import Sampler

    class RandomSampler(Sampler):
        def choice(self, candidates, mutator, model, index):
            return random.choice(candidates)

    class RandomStrategy(BaseStrategy):
        def __init__(self):
            self.random_sampler = RandomSampler()

        def run(self, base_model, applied_mutators):
            _logger.info('stargety start...')
            while True:
                avail_resource = query_available_resources()
                if avail_resource > 0:
                    model = base_model
                    _logger.info('apply mutators...')
                    _logger.info('mutators: %s', str(applied_mutators))
                    for mutator in applied_mutators:
                        mutator.bind_sampler(self.random_sampler)
                        model = mutator.apply(model)
                    # run models
                    submit_models(model)
                else:
                    time.sleep(2)

You can find that this strategy does not know the search space beforehand, it passively makes decisions every time ``choice`` is invoked from mutators. If a strategy wants to know the whole search space before making any decision (e.g., TPE, SMAC), it can use ``dry_run`` function provided by ``Mutator`` to obtain the space. An example strategy can be found :githublink:`here <nni/retiarii/strategy/tpe_strategy.py>`.

After generating a new model, the strategy can use our provided APIs (e.g., ``submit_models``, ``is_stopped_exec``) to submit the model and get its reported results.

References
^^^^^^^^^^

..  autoclass:: nni.retiarii.Sampler
    :members:
    :noindex:

..  autoclass:: nni.retiarii.strategy.BaseStrategy
    :members:
    :noindex:

Customize a New One-shot Trainer (legacy)
-----------------------------------------

One-shot trainers should inherit :class:`nni.retiarii.oneshot.BaseOneShotTrainer`, and need to implement ``fit()`` (used to conduct the fitting and searching process) and ``export()`` method (used to return the searched best architecture).

Writing a one-shot trainer is very different to single-arch evaluator. First of all, there are no more restrictions on init method arguments, any Python arguments are acceptable. Secondly, the model fed into one-shot trainers might be a model with Retiarii-specific modules, such as LayerChoice and InputChoice. Such model cannot directly forward-propagate and trainers need to decide how to handle those modules.

A typical example is DartsTrainer, where learnable-parameters are used to combine multiple choices in LayerChoice. Retiarii provides ease-to-use utility functions for module-replace purposes, namely ``replace_layer_choice``, ``replace_input_choice``. A simplified example is as follows: 

.. code-block:: python

    from nni.retiarii.oneshot import BaseOneShotTrainer
    from nni.retiarii.oneshot.pytorch import replace_layer_choice, replace_input_choice


    class DartsLayerChoice(nn.Module):
        def __init__(self, layer_choice):
            super(DartsLayerChoice, self).__init__()
            self.name = layer_choice.label
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

The full code of DartsTrainer is available to Retiarii source code. Please have a check at :githublink:`DartsTrainer <nni/retiarii/oneshot/pytorch/darts.py>`.

References
^^^^^^^^^^

..  autoclass:: nni.retiarii.oneshot.BaseOneShotTrainer
    :members:
    :noindex:
