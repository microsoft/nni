# **指南** - 自定义 Advisor

*警告：API 可能会在将来的版本中更改。*

Advisor 用于同时需要 Tuner 和 Assessor 方法的自动机器学习算法。 Advisor 与 Tuner 类似，它接收 Trial 的参数请求、最终结果，并生成 Trial 的参数。 另外，它也能像 Assessor 一样接收中间结果、Trial 的最终状态，并可以发送终止 Trial 的命令。 注意，在使用 Advisor 时，不能同时使用 Tuner 和 Assessor。

如果要自定义 Advisor，需要：

**1. 定义从 MsgDispatcherBase 类继承的 Advisor。** 如：

```python
from nni.msg_dispatcher_base import MsgDispatcherBase

class CustomizedAdvisor(MsgDispatcherBase):
    def __init__(self, ...):
        ...
```

**2. 实现所有除了 `handle_request` 外的，以 `handle_` 前缀开始的方法**。 [此文档](https://nni.readthedocs.io/zh/latest/sdk_reference.html#nni.msg_dispatcher_base.MsgDispatcherBase)可帮助理解 `MsgDispatcherBase`。

**3. 在 Experiment 的 YAML 文件中配置好自定义的 Advisor。**

与 Tuner 和 Assessor 类似。 NNI 需要定位到自定义的 Advisor 类，并实例化它，因此需要指定自定义 Advisor 类的文件位置，并将参数值传给 `__init__` 构造函数。

```yaml
advisor:
  codeDir: /home/abc/myadvisor
  classFileName: my_customized_advisor.py
  className: CustomizedAdvisor
  # 任何传入 __init__ 构造函数的参数
  # 都需要声明在 classArgs 字段中，如：
  classArgs:
    arg1: value1
```

**注意：**Advisor 的工作目录是`<home>/nni-experiments/<experiment_id>/log` 可从环境变量 `NNI_LOG_DIRECTORY` 中获取。

## 示例

参考[示例](https://github.com/microsoft/nni/tree/master/examples/tuners/mnist_keras_customized_advisor)。