# **指南** - 自定义 Advisor

*Advisor 用于同时需要 Tuner 和 Assessor 方法的自动机器学习算法。 Advisor 与 Tuner 类似，它接收 Trial 的参数请求、最终结果，并生成 Trial 的参数。 另外，它也能像 Assessor 一样接收中间结果、Trial 的最终状态，并可以发送终止 Trial 的命令。 注意，在使用 Advisor 时，不能同时使用 Tuner 和 Assessor。*

如果要自定义 Advisor，需要：

1. 从 MsgDispatcherBase 类继承并创建新的 Advisor 类
2. 实现所有除了 `handle_request` 外的，以 `handle_` 前缀开始的方法
3. 在 Experiment 的 YAML 文件中配置好自定义的 Advisor

样例如下：

**1) 从 MsgDispatcherBase 类继承并创建新的 Advisor 类**

```python
from nni.msg_dispatcher_base import MsgDispatcherBase

class CustomizedAdvisor(MsgDispatcherBase):
    def __init__(self, ...):
        ...
```

**2) 实现所有除了 `handle_request` 外的，以 `handle_` 前缀开始的方法**

参考 Hyperband 的实现 ([src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py)) 来学习如何实现这些方法。

**3) 在 Experiment 的 YAML 文件中配置好自定义的 Advisor**

与 Tuner 和 Assessor 类似。 NNI 需要定位到自定义的 Advisor 类，并实例化它，因此需要指定自定义 Advisor 类的文件位置，并将参数值传给 \_\_init__ 构造函数。

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