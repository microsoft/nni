# 自定义 Assessor

NNI supports to build an assessor by yourself for tuning demand.

If you want to implement a customized Assessor, there are three things to do:

1. Inherit the base Assessor class
2. Implement assess_trial function
3. Configure your customized Assessor in experiment YAML config file

**1. Inherit the base Assessor class**

```python
from nni.assessor import Assessor

class CustomizedAssessor(Assessor):
    def __init__(self, ...):
        ...
```

**2. Implement assess trial function**

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

**3. Configure your customized Assessor in experiment YAML config file**

NNI needs to locate your customized Assessor class and instantiate the class, so you need to specify the location of the customized Assessor class and pass literal values as parameters to the \_\_init__ constructor.

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

Please noted in **2**. The object `trial_history` are exact the object that Trial send to Assessor by using SDK `report_intermediate_result` function.

More detail example you could see:

> - [medianstop-assessor](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/medianstop_assessor)
> - [curvefitting-assessor](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/curvefitting_assessor)