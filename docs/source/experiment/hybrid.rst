Hybrid Training Service
=======================

Hybrid training service is for aggregating different types of computation resources into a virtually unified resource pool, in which trial jobs are dispatched. Hybrid training service is for collecting user's all available computation resources to jointly work on an AutoML task, it is flexibile enough to switch among different types of computation resources. For example, NNI could submit trial jobs to multiple remote machines and AML simultaneously.

Prerequisite
------------

NNI has supported :doc:`./local`, :doc:`./remote`, :doc:`./openpai`, :doc:`./aml`, :doc:`./kubeflow`, :doc:`./frameworkcontroller`, for hybrid training service. Before starting an experiment using using hybrid training service, users should first setup their chosen (sub) training services (e.g., remote training service) according to each training service's own document page.

Usage
-----

Unlike other training services (e.g., ``platform: remote`` in remote training service), there is no dedicated keyword for hybrid training service, users can simply list the configurations of their chosen training services under the ``trainingService`` field. Below is an example of a hybrid training service containing remote training service and local training service in experiment configuration yaml.

.. code-block:: yaml

    # the experiment config yaml file
    ...
    trainingService:
      - platform: remote
        machineList:
          - host: 127.0.0.1 # your machine's IP address
            user: bob
            password: bob
      - platform: local
    ...

A complete example configuration file can be found in :githublink:`examples/trials/mnist-pytorch/config_hybrid.yml`.