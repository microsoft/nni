Batch Tuner on NNI
===

## Batch Tuner

Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in [search space spec](../Tutorial/SearchSpaceSpec.md).

Suggested scenario: If the configurations you want to try have been decided, you can list them in SearchSpace file (using choice) and run them using batch tuner.
