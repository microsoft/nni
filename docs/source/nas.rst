#############################################
Retiarii for Neural Architecture Search (NAS)
#############################################

Automatic neural architecture search is taking an increasingly important role on finding better models.
Recent research works have proved the feasibility of automatic NAS, and also found some models that could beat manually tuned models.
Some of representative works are NASNet, ENAS, DARTS, Network Morphism, and Evolution. Moreover, new innovations keep emerging.

However, it takes great efforts to implement NAS algorithms, and it is hard to reuse code base of existing algorithms in a new one.
To facilitate NAS innovations (e.g., design and implement new NAS models, compare different NAS models side-by-side),
an easy-to-use and flexible programming interface is crucial.

Thus, we design `Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__. It is a deep learning framework that supports the exploratory training on a neural network model space, rather than on a single neural network model.
Exploratory training with Retiarii allows user to express various search spaces for *Neural Architecture Search* and *Hyper-Parameter Tuning* with high flexibility.

Some frequently used terminologies in this document:

* *Model search space*: it means a set of models from which the best model is explored/searched. Sometimes we use *search space* or *model space* in short.
* *Exploration strategy*: the algorithm that is used to explore a model search space.
* *Model evaluator*: it is used to train a model and evaluate the model's performance.

Follow the instructions below to start your journey with Retiarii.

..  toctree::
    :maxdepth: 2

    Overview <NAS/Overview>
    Quick Start <NAS/QuickStart>
    Construct Model Space <NAS/construct_space>
    Multi-trial NAS <NAS/multi_trial_nas>
    One-shot NAS <NAS/one_shot_nas>
    Hardware-aware NAS <NAS/HardwareAwareNAS>
    NAS Benchmarks <NAS/Benchmarks>
    NAS API References <NAS/ApiReference>
