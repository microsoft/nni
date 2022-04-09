Overview
========

NNI has supported many training services listed below. Users can go through each page to learning how to configure the corresponding training service. NNI has high extensibility by design, users can customize new training service for their special resource, platform or needs.


..  list-table::
    :header-rows: 1

    * - Training Service
      - Description
    * - Local
      - The whole experiment runs on your dev machine (i.e., a single local machine)
    * - Remote
      - The trials are dispatched to your configured remote servers
    * - OpenPAI
      - Running trials on OpenPAI, a DNN model training platform based on Kubernetes
    * - Kubeflow
      - Running trials with Kubeflow, a DNN model training framework based on Kubernetes
    * - AdaptDL
      - Running trials on AdaptDL, an elastic DNN model training platform
    * - FrameworkController
      - Running trials with FrameworkController, a DNN model training framework on Kubernetes
    * - AML
      - Running trials on AML cloud service
    * - PAI-DLC
      - Running trials on PAI-DLC, which is deep learning containers based on Alibaba ACK
    * - Hybrid
      - Support jointly using multiple above training services