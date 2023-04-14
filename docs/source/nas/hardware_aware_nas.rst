Hardware-aware NAS
==================

Hardware-aware NAS is a technique to search for architectures under the constraints of a given hardware platform. Deploying a neural network on a specific hardware is challenging because different hardwares have different preferences for operations, combinations of operations. Different use scenarios might also pose different levels of requirements. Some strict scenarios might require the networks to be smaller and faster. Our hardware-aware NAS supports searching for architectures under the constraints of latency, model size or FLOPs. This document demonstrates how to use multi-trial strategy or one-shot strategy combining those constraints.

Profiler
--------

:class:`~nni.nas.profiler.Profiler` is designed to efficiently compute metrics like latency for models within the same model space. Specifically, it is first initialized with a model space, in which it precomputes some data, and for any sample in the model space, it can quickly give out a metric::

    class MyModelSpace(ModelSpace):
        ...
      
    from nni.nas.profiler.pytorch.flops import FlopsProfiler
    # initialization
    profiler = FlopsProfiler(net, torch.randn(3))  # randn(3) is a dummy input. It could be a tensor or a tuple of tensors.
    # compute flops for a sample
    flops = profiler.profile({'layer1': 'conv'})

NNI currently supports the following types of profilers:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Brief Description
   * - :class:`~nni.nas.profiler.pytorch.flops.FlopsProfiler`
     - Profile the FLOPs of a model
   * - :class:`~nni.nas.profiler.pytorch.flops.NumParamsProfiler`
     - Profile the number of parameters of a model
   * - :class:`~nni.nas.profiler.pytorch.nn_meter.NnMeterProfiler`
     - Profile the estimated latency of a model with nn-meter

Hardware-aware multi-trial search
---------------------------------

When using multi-trial strategies, the most intuitive approach to combine a hardware-aware constraint is to filter out those outside the constraints. This can be done via strategy middleware :class:`~nni.nas.strategy.middleware.Filter`. An example is as follows::

    from nni.nas.strategy import Random
    from nni.nas.strategy.middleware import Filter, Chain
    from nni.nas.profiler.pytorch.flops import FlopsProfiler

    profiler = FlopsProfiler(model_space, dummy_input)

    strategy = Chain(
        Random(),
        Filter(lambda sample: profiler.profile(sample) < 300e6)
    )

The example here uses a random strategy to randomly generate models, but the FLOPs is restricted to less than 300M. Models over the limitation will be directly discarded without training. :class:`~nni.nas.strategy.middleware.Filter` can be also set to a mode to return a bad metric for models out of constraint, so that the strategy can learn to avoid sampling such models. The profiler can also be replaced with any profiler listed above or customized.

Hardware-aware one-shot strategy
--------------------------------

One-shot strategy can be also combined with metrics on hardware. There are usually two approaches.

The first approach is to add a special regularization term to the loss within one-shot strategy, penalizing the sampling of models that does not fit the constraints. To do this, a penalty term (either :class:`~nni.nas.oneshot.pytorch.profiler.ExpectationProfilerPenalty` for differentiable algorithms, or :class:`~nni.nas.oneshot.pytorch.profiler.SampleProfilerPenalty` for sampling-based algorithms). Example below::

    from nni.nas.strategy import ProxylessNAS
    from nni.nas.oneshot.pytorch.profiler import ExpectationProfilerPenalty
    # For sampling-based algorithms like ENAS, use `SampleProfilerPenalty` here.
    # Please see the document for each algorithm for the type of penalties they have supported.

    profiler = FlopsProfiler(model_space, dummy_input)
    penalty = ExpectationProfilerPenalty(profiler, 300e6)  # 300M is the expected profiler here. Exceeding it will be penalized.
    strategy = ProxylessNAS(penalty=penalty)

Another approach is similar to what we've done for multi-trial strategies: to directly prevent models out of constraints from being sampled. To do this, use :class:`~nni.nas.oneshot.pytorch.profiler.RangeProfilerFilter`. Example::

    from nni.nas.strategy import ENAS
    from nni.nas.oneshot.pytorch.profiler import RangeProfilerFilter

    profiler = FlopsProfiler(model_space, dummy_input)
    penalty = RangeProfilerFilter(profiler, 200e6, 300e6)  # Only flops between 200M and 300M are considered legal.
    strategy = ENAS(filter=filter)

.. tip:: The penalty and filter here are specialized for one-shot strategies, please do not use them in multi-trial strategies.

Best practices of hardware-aware NAS
------------------------------------

The hardware-aware part in NAS is probably the most complex component within the whole NNI NAS framework. It's expected that users might encounter technical issues when using hardware-aware NAS. A full troubleshooting guide is still under preparation. For now, we recommend the following practices, briefly.

1. Make sure shape inference succeeds. In order to make profiler to work, we will dry run the model space and infer a symbolic shape for inputs and outputs of every submodule. Built-in implementations only support a limited set of operations when inferencing shapes. If errors like ``no shape formula``, please register the shape formula following the prompt, or decorate the whole module as a leaf module that doesn't need to be opened. Note that if the shape inference doesn't open a module, its FLOPs and latency might also need to compute as a whole. You might also need to write FLOPs / latency formula for the module.
2. Try with FLOPs first. In our experience, complex profilers like nn-Meter might make it harder to debug when something goes wrong. Remember to examine whether the FLOPs profiler returns a reasonable result. This can be done by manually invoking ``profiler.profile(sample)``.
3. :class:`~nni.nas.profiler.pytorch.nn_meter.NnMeterProfiler` will expand all the possible modules when it considers a module space as a leaf module (note that nn-meter has its own leaf module settings and do not follow what has been set for shape inference). If the submodule contains too many combinations. The profiler might hang when preprocessing. Try using ``logging.getLogger('nni').setLevel(logging.DEBUG)`` to print debug logs, so as to identify the cause of the issue.
4. For a specific model space and a specific hardware, you can also build your own profiler with :class:`~nni.nas.profiler.Profiler`. As long as they follow the interface of :class:`~nni.nas.profiler.Profiler`, the inner implementation doesn't matter. Users can use lookup tables, build predictors, or even connecting to the real device for profiling. If the interface is compatible, it's possible to use it combining our built-in strategies. It's usually the recommended method when your model space is too complex for the general shape inference to work, or you are targetting at the specialized hardware we do not yet support.
