AML Training Service
====================

To run your trials on `AzureML <https://azure.microsoft.com/en-us/services/machine-learning/>`__, you can use AML training service. AML training service can programmatically submit runs to AzureML platform and collect their metrics.

Prerequisite
------------

1. Create an Azure account/subscription using this `link <https://azure.microsoft.com/en-us/free/services/machine-learning/>`__. If you already have an Azure account/subscription, skip this step.
2. Install the Azure CLI on your machine, follow the install guide `here <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest>`__.
3. Authenticate to your Azure subscription from the CLI. To authenticate interactively, open a command line or terminal and use the following command:

   .. code-block:: bash

      az login

4. Log into your Azure account with a web browser and create a Machine Learning resource. You will need to choose a resource group and specific a workspace name. Then download ``config.json`` which will be used later.

   .. image:: ../../../img/aml_workspace.png

5. Create an AML cluster as the compute target.

   .. image:: ../../../img/aml_cluster.png

6. Open a command line and install AML package environment.

   .. code-block:: bash

      python3 -m pip install azureml
      python3 -m pip install azureml-sdk

Usage
-----

We show an example configuration here with YAML (Python configuration should be similar).

.. code-block:: yaml

   trialConcurrency: 1
   maxTrialNumber: 10
   ...
   trainingService:
     platform: aml
     dockerImage: msranni/nni
     subscriptionId: ${your subscription ID}
     resourceGroup: ${your resource group}
     workspaceName: ${your workspace name}
     computeTarget: ${your compute target}

Configuration References
------------------------

Compared with :doc:`local` and :doc:`remote`, OpenPAI training service supports the following additional configurations.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Field name
     - Description
   * - dockerImage
     - Required field. The docker image name used in job. If you don't want to build your own, NNI has provided a docker image `msranni/nni <https://hub.docker.com/r/msranni/nni>`__, which is up-to-date with every NNI release.
   * - subscriptionId
     - Required field. The subscription id of your account, can be found in ``config.json`` described above.
   * - resourceGroup
     - Required field. The resource group of your account, can be found in ``config.json`` described above.
   * - workspaceName
     - Required field. The workspace name of your account, can be found in ``config.json`` described above.
   * - computeTarget
     - Required field. The compute cluster name you want to use in your AML workspace. See `reference <https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target>`__ and Step 5 above.
   * - maxTrialNumberPerGpu
     - Optional field. Default 1. Used to specify the max concurrency trial number on a GPU device.
   * - useActiveGpu
     - Optional field. Default false. Used to specify whether to use a GPU if there is another process. By default, NNI will use the GPU only if there is no other active process in the GPU. See :doc:`local` for details.

Monitor your trial on the cloud by using AML studio
---------------------------------------------------

To see your trial job's detailed status on the cloud, you need to visit your studio which you create at Step 5 above. Once the job completes, go to the **Outputs + logs** tab. There you can see a ``70_driver_log.txt`` file, This file contains the standard output from a run and can be useful when you're debugging remote runs in the cloud. Learn more about aml from `here <https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-hello-world>`__.
