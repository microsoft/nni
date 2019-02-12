# Batch Tuner on NNI

## Batch Tuner

Batch Tuner 能让用户简单的提供几组配置（如，超参选项的组合）。 当所有配置都执行完后，Experiment 即结束。 Batch tuner only supports the type choice in [search space spec](../../../../../docs/SearchSpaceSpec.md).

Suggested sceanrio: If the configurations you want to try have been decided, you can list them in searchspace file (using choice) and run them using batch tuner.