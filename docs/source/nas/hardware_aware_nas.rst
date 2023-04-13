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

The example here uses a random strategy to randomly generate models, but the FLOPs is restricted to less than 300M. Models over the limitation will be directly discarded without training. :class:`~nni.nas.strategy.middleware.Filter` can be also set to a mode to return a bad metric for models out of constraint, so that the strategy can learn to avoid sampling such models.

Hardware-aware one-shot strategy
--------------------------------

One-shot strategy can be also combined with metrics on hardware.
