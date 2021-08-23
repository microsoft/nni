Hardware-aware NAS
==================

.. contents::

EndToEnd Multi-trial SPOS Demo
------------------------------

To empower affordable DNN on the edge and mobile devices, hardware-aware NAS searches both high accuracy and low latency models. In particular, the search algorithm only considers the models within the target latency constraints during the search process.

To run this demo, first install nn-Meter from source code (Github repo link: https://github.com/microsoft/nn-Meter. Currently we haven't released this package, so development installation is required).

.. code-block:: bash

  python setup.py develop

Then run multi-trail SPOS demo:

.. code-block:: bash

  python ${NNI_ROOT}/examples/nas/oneshot/spos/multi_trial.py

How the demo works
------------------

To support hardware-aware NAS, you first need a `Strategy` that supports filtering the models by latency. We provide such a filter named `LatencyFilter` in NNI and initialize a `Random` strategy with the filter:

.. code-block:: python

  simple_strategy = strategy.Random(model_filter=LatencyFilter(threshold=100, predictor=base_predictor))

``LatencyFilter`` will predict the models\' latency by using nn-Meter and filter out the models whose latency are larger than the threshold (i.e., ``100`` in this example).
You can also build your own strategies and filters to support more flexible NAS such as sorting the models according to latency.

Then, pass this strategy to ``RetiariiExperiment``:

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, strategy=simple_strategy)

  exp_config = RetiariiExeConfig('local')
  ...
  exp_config.dummy_input = [1, 3, 32, 32]

  exp.run(exp_config, port)

In ``exp_config``, ``dummy_input`` is required for tracing shape info.
