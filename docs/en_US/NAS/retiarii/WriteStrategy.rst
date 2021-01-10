Customize A New Strategy
========================

To write a new strategy, you should inherit the base strategy class ``BaseStrategy``, then implement the member function ``run``. This member function takes ``base_model`` and ``applied_mutators`` as its input arguments. It can simply apply the user specified mutators in ``applied_mutators`` onto ``base_model`` to generate a new model. When a mutator is applied, it should be bound with a sampler (e.g., ``RandomSampler``). Every sampler implements the ``choice`` function which chooses value(s) from candidate values. The ``choice`` functions invoked in mutators are executed with the sampler.

Below is a very simple random strategy, the complete code can be found :githublink:`here <nni/retiarii/strategies/random_strategy.py>`.

.. code-block:: python

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

You can find that this strategy does not know the search space beforehand, it passively makes decisions every time ``choice`` is invoked from mutators. If a strategy wants to know the whole search space before making any decision (e.g., TPE, SMAC), it can use ``dry_run`` function provided by ``Mutator`` to obtain the space. An example strategy can be found :githublink:`here <nni/retiarii/strategies/tpe_strategy.py>`.

After generating a new model, the strategy can use our provided APIs (e.g., ``submit_models``, ``is_stopped_exec``) to submit the model and get its reported results. More APIs can be found in `API References <./ApiReference.rst>`__.