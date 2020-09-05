# Batch Tuner

## Batch Tuner（批处理 Tuner）

Batch Tuner 能让用户简单的提供几组配置（如，超参选项的组合）。 当所有配置都执行完后，Experiment 即结束。 Batch Tuner 的[搜索空间](../Tutorial/SearchSpaceSpec.md)只支持 `choice`。

建议场景：如果 Experiment 配置已确定，可通过 `choice` 将它们罗列到搜索空间文件中运行即可。