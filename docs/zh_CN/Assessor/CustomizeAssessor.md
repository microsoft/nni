# 自定义 Assessor

NNI 支持自定义 Assessor。

实现自定义的 Assessor，需要如下几步：

1. 继承 Assessor 基类
2. 实现 assess_trial 函数
3. 在 Experiment 的 YAML 文件中配置好自定义的 Assessor

**1. 继承 Assessor 基类**

```python
from nni.assessor import Assessor

class CustomizedAssessor(Assessor):
    def __init__(self, ...):
        ...
```

**2. 实现 assess_trial 函数**

```python
from nni.assessor import Assessor, AssessResult

class CustomizedAssessor(Assessor):
    def __init__(self, ...):
        ...

    def assess_trial(self, trial_history):
        """
        确定是否要停止该 Trial。 必须重载。
        trial_history: 中间结果列表对象。
        返回 AssessResult.Good 或 AssessResult.Bad。
        """
        # 代码实现于此处。
        ...
```

**3. 在 Experiment 的 YAML 文件中配置好自定义的 Assessor**

NNI 需要定位到自定义的 Assessor 类，并实例化它，因此需要指定自定义 Assessor 类的文件位置，并将参数值传给 \_\_init__ 构造函数。

```yaml
assessor:
  codeDir: /home/abc/myassessor
  classFileName: my_customized_assessor.py
  className: CustomizedAssessor
  # 任何传入 __init__ 构造函数的参数，
  # 都需要在 classArgs 字段中指定，如
  classArgs:
    arg1: value1
```

注意在 **2** 中， `trial_history` 对象与 Trial 通过 `report_intermediate_result` 函数返回给 Assessor 的对象完全一致。

Assessor 的工作目录是`<home>/nni-experiments/<experiment_id>/log` 可从环境变量 `NNI_LOG_DIRECTORY` 中获取。

更多示例，可参考：

> * [medianstop-assessor](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/medianstop_assessor)
> * [curvefitting-assessor](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/curvefitting_assessor)