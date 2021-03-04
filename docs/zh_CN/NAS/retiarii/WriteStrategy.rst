自定义 Strategy
========================

要编写新策略，应该继承基本策略类 ``BaseStrategy``，然后实现成员函数 ``run``。 此成员函数将 ``base_model`` 和 ``applied_mutators`` 作为输入参数， 并将用户在 ``applied_mutators`` 中指定的 Mutator 应用到 ``base_model`` 中生成新模型。 当应用一个 Mutator 时，应该与一个 sampler 绑定（例如，``RandomSampler``）。 每个 sampler 都实现了从候选值中选择值的 ``choice`` 函数。 在 Mutator 中调用 ``choice`` 函数是用 sampler 执行的。

下面是一个非常简单的随机策略，它使选择完全随机。

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
                    # 运行模型
                    submit_models(model)
                else:
                    time.sleep(2)

您会发现此策略事先并不知道搜索空间，每次从 Mutator 调用 ``choice`` 时都会被动地做出决定。 如果一个策略想在做出任何决策（如 TPE、SMAC）之前知道整个搜索空间，它可以使用 ``Mutator`` 提供的 ``dry_run`` 函数来获取搜索空间。 可以在 :githublink:`这里 <nni/retiarii/strategy/tpe_strategy.py>` 找到一个示例策略。

生成新模型后，该策略可以使用 NNI 提供的API（例如 ``submit_models``, ``is_stopped_exec``）提交模型并获取其报告的结果。 更多的 API 在 `API 参考 <./ApiReference.rst>`__ 中。