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
        Determines whether a trial should be killed. Must override.
        trial_history: a list of intermediate result objects.
        Returns AssessResult.Good or AssessResult.Bad.
        """
        # you code implement here.
        ...
```

**3. Configure your customized Assessor in experiment YAML config file**

NNI needs to locate your customized Assessor class and instantiate the class, so you need to specify the location of the customized Assessor class and pass literal values as parameters to the \_\_init__ constructor.

```yaml
assessor:
  codeDir: /home/abc/myassessor
  classFileName: my_customized_assessor.py
  className: CustomizedAssessor
  # Any parameter need to pass to your Assessor class __init__ constructor
  # can be specified in this optional classArgs field, for example 
  classArgs:
    arg1: value1
```

Please noted in **2**. The object `trial_history` are exact the object that Trial send to Assessor by using SDK `report_intermediate_result` function.

More detail example you could see:

> - [medianstop-assessor](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/medianstop_assessor)
> - [curvefitting-assessor](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/curvefitting_assessor)