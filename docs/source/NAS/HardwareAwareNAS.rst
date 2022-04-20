Hardware-aware NAS
==================

.. contents::

End-to-end Multi-trial SPOS Demo
--------------------------------

To empower affordable DNN on the edge and mobile devices, hardware-aware NAS searches both high accuracy and low latency models. In particular, the search algorithm only considers the models within the target latency constraints during the search process.

To run this demo, first install nn-Meter by running:

.. code-block:: bash

  pip install nn-meter

Then run multi-trail SPOS demo:

.. code-block:: bash

  python ${NNI_ROOT}/examples/nas/oneshot/spos/multi_trial.py


How the demo works
^^^^^^^^^^^^^^^^^^

To support hardware-aware NAS, you first need a ``Strategy`` that supports filtering the models by latency. We provide such a filter named ``LatencyFilter`` in NNI and initialize a ``Random`` strategy with the filter:

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


End-to-end ProxylessNAS with Latency Constraints
------------------------------------------------

`ProxylessNAS <https://arxiv.org/pdf/1812.00332.pdf>`__ is a hardware-aware one-shot NAS algorithm. ProxylessNAS applies the expected latency of the model to build a differentiable metric and design efficient neural network architectures for hardware. The latency loss is added as a regularization term for architecture parameter optimization. In this example, nn-Meter provides a latency estimator to predict expected latency for the mixed operation on other types of mobile and edge hardware. 

To run the one-shot ProxylessNAS demo, first install nn-Meter by running:

.. code-block:: bash

  pip install nn-meter

Then run one-shot ProxylessNAS demo:

```bash
python ${NNI_ROOT}/examples/nas/oneshot/proxylessnas/main.py --applied_hardware <hardware> --reference_latency <reference latency (ms)>
```

How the demo works
^^^^^^^^^^^^^^^^^^

In the implementation of ProxylessNAS ``trainer``, we provide a ``HardwareLatencyEstimator`` which currently builds a lookup table, that stores the measured latency of each candidate building block in the search space. The latency sum of all building blocks in a candidate model will be treated as the model inference latency. The latency prediction is obtained by ``nn-Meter``. ``HardwareLatencyEstimator`` predicts expected latency for the mixed operation based on the path weight of `ProxylessLayerChoice`. With leveraging ``nn-Meter`` in NNI, users can apply ProxylessNAS to search efficient DNN models on more types of edge devices. 

Despite of ``applied_hardware`` and ``reference_latency``, There are some other parameters related to hardware-aware ProxylessNAS training in this :githublink:`example <examples/nas/oneshot/proxylessnas/main.py>`:

* ``grad_reg_loss_type``: Regularization type to add hardware related loss. Allowed types include ``"mul#log"`` and ``"add#linear"``. Type of ``mul#log`` is calculate by ``(torch.log(expected_latency) / math.log(reference_latency)) ** beta``. Type of ``"add#linear"`` is calculate by ``reg_lambda * (expected_latency - reference_latency) / reference_latency``. 
* ``grad_reg_loss_lambda``: Regularization params, is set to ``0.1`` by default.
* ``grad_reg_loss_alpha``: Regularization params, is set to ``0.2`` by default.
* ``grad_reg_loss_beta``: Regularization params, is set to ``0.3`` by default.
* ``dummy_input``: The dummy input shape when applied to the target hardware. This parameter is set as (1, 3, 224, 224) by default.
