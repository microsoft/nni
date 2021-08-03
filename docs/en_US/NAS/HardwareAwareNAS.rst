Hardware-aware NAS
==================

.. contents::

EndToEnd Multi-trial SPOS Demo
------------------------------

Basically, this demo will select the model whose latency satisfy constraints to train.

To run this demo, first install nn-Meter from source code (Github repo link: https://github.com/microsoft/nn-Meter. Currently we haven't released this package, so development installation is required).

.. code-block:: bash

  python setup.py develop

Then run multi-trail SPOS demo:

.. code-block:: bash

  python ${NNI_ROOT}/examples/nas/oneshot/spos/multi_trial.py

How the demo works
------------------

To support latency-aware NAS, you first need a `Strategy` that supports filtering the models by latency. We provide such a filter named `LatencyFilter` in NNI and initialize a `Random` strategy with the filter:

.. code-block:: python

  simple_strategy = strategy.Random(model_filter=LatencyFilter(100)

``LatencyFilter`` will predict the models\' latency by using nn-Meter and filter out the models whose latency are larger than the threshold (i.e., ``100`` in this example).
You can also build your own strategies and filters to support more flexible NAS such as sorting the models according to latency.

Then, pass this strategy to ``RetiariiExperiment`` along with some additional arguments: ``parse_shape=True, dummy_input=dummy_input``:

.. code-block:: python

  RetiariiExperiment(base_model, trainer, [], simple_strategy, True, dummy_input)

Here, ``parse_shape=True`` means extracting shape info from the torch model as it is required by nn-Meter to predict latency. ``dummy_input`` is required for tracing shape info.
