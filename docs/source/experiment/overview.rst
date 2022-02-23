NNI Experiment
==============

An NNI experiment is a unit of one tuning process. For example, it is one run of hyper-parameter tuning on a specific search space, it is one run of neural architecture search on a search space, or it is one run of automatic model compression on user specified goal on latency and accuracy. Usually, the tuning process requires many trials to explore feasible and potentially good-performing models. Thus, an important component of NNI experiment is **training service**, which is a unified interface to abstract diverse computation resources (e.g., local machine, remote servers, AKS). Users can easily run the tuning process on their prefered computation resource and platform. On the other hand, NNI experiment provides **WebUI** to visualize the tuning process to users.

During developing a DNN model, users need manage the tuning process, such as, creating an experiment, adjusting an experiment, kill or rerun a trial in an experiment, dumping experiment data for customized analysis. Also, users may create a new experiment for comparison, or concurrently for new model developing tasks. Thus, NNI provides the functionality of **experiment management**. Users can use :doc:`../reference/nnictl` to interact with experiments.

Before reading the following content, you are recommended to go through the quick start first.

..  toctree::
    :maxdepth: 2

    Training Services <training_service>
    WebUI <webui>
    Experiment Management <exp_management>