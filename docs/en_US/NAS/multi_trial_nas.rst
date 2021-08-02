Multi-trial NAS
===============

In multi-trial NAS, users need model evaluator to evaluate the performance of each sampled model, and need an exploration strategy to sample models from a defined model space. Here, users could use NNI provided model evaluators or write their own model evalutor. They can simply choose a exploration strategy. Advanced users can also customize new exploration strategy. For a simple example about how to run a multi-trial NAS experiment, please refer to `Quick Start <./QuickStart.rst>`__.

..  toctree::
    :maxdepth: 1

    Model Evaluators <ModelEvaluators>
    Customize Model Evaluator <WriteTrainer>
    Exploration Strategies <ExplorationStrategies>
    Customize Exploration Strategies <WriteStrategy>
    Execution Engines <ExecutionEngines>
    Hardware-aware NAS <HardwareAwareNAS>
