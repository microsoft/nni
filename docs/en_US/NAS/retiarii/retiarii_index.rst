#################
Retiarii Overview
#################

`Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ is a deep learning framework that supports the exploratory training on a neural network model space, rather than on a single neural network model. 

Exploratory training with Retiarii allows user to express various search space for **Neural Architecture Search** and **Hyper-Parameter Tuning** with high flexibility. 

As previous NAS and HPO supports, the new framework continued the ability for allowing user to reuse SOTA search algorithms, and to leverage system level optimizations to speed up the search process. 

Follow the instructions below to start your journey with Retiarii.

..  toctree::
    :maxdepth: 2

    Quick Start <Tutorial>
    Write a Model Evaluator <WriteTrainer>
    One-shot NAS <OneshotTrainer>
    Advanced Tutorial <Advanced>
    Customize a New Strategy <WriteStrategy>
    Retiarii APIs <ApiReference>
