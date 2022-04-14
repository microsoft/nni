Overview
========

NNI has supported many training services listed below. Users can go through each page to learning how to configure the corresponding training service. NNI has high extensibility by design, users can customize new training service for their special resource, platform or needs.


..  list-table::
    :header-rows: 1

    * - Training Service
      - Description
    * - :doc:`Local <local>`
      - The whole experiment runs on your dev machine (i.e., a single local machine)
    * - :doc:`Remote <remote>`
      - The trials are dispatched to your configured SSH servers
    * - :doc:`OpenPAI <openpai>`
      - Running trials on OpenPAI, a DNN model training platform based on Kubernetes
    * - :doc:`Kubeflow <kubeflow>`
      - Running trials with Kubeflow, a DNN model training framework based on Kubernetes
    * - :doc:`AdaptDL <adaptdl>`
      - Running trials on AdaptDL, an elastic DNN model training platform
    * - :doc:`FrameworkController <frameworkcontroller>`
      - Running trials with FrameworkController, a DNN model training framework on Kubernetes
    * - :doc:`AML <aml>`
      - Running trials on Azure Machine Learning (AML) cloud service
    * - :doc:`PAI-DLC <paidlc>`
      - Running trials on PAI-DLC, which is deep learning containers based on Alibaba ACK
    * - :doc:`Hybrid <hybrid>`
      - Support jointly using multiple above training services

.. _training-service-reuse:

Training Service Under Reuse Mode
---------------------------------

Since NNI v2.0, there are two sets of training service implementations in NNI. The new one is called *reuse mode*. When reuse mode is enabled, a cluster, such as a remote machine or a computer instance on AML, will launch a long-running environment, so that NNI will submit trials to these environments iteratively, which saves the time to create new jobs. For instance, using OpenPAI training platform under reuse mode can avoid the overhead of pulling docker images, creating containers, and downloading data repeatedly.

.. note:: In the reuse mode, users need to make sure each trial can run independently in the same job (e.g., avoid loading checkpoints from previous trials).
