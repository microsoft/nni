# Batch Tuner

## Batch Tuner（批处理 Tuner）

Batch Tuner 能让用户简单的提供几组配置（如，超参选项的组合）。 当所有配置都执行完后，Experiment 即结束。 Batch Tuner 的[搜索空间](SearchSpaceSpec.md)只支持 `choice`。

Suggested scenario: If the configurations you want to try have been decided, you can list them in SearchSpace file (using choice) and run them using batch tuner.