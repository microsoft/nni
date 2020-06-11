# 自定义 Assessor

*Assessor 从 Trial 中接收中间结果，并决定此 Trial 是否应该终止。 一旦 Trial 满足提前终止条件，Assessor 将终止此 Trial。*

因此，如果要自定义 Assessor，需要：

**1) 继承于 Assessor 基类，创建 Assessor 类**

```python
from nni.assessor import Assessor

class CustomizedAssessor(Assessor):
    def __init__(self, ...):
        ...
```

**2) 实现评估 Trial 的函数**

```python
from nni.assessor import Assessor, AssessResult

class CustomizedAssessor(Assessor):
    def __init__(self, ...):
        ...

    def assess_trial(self, trial_history):
        """
        决定是否应该终止 Trial。 必须重载。
        trial_history: 中间结果列表对象。
        返回 AssessResult.Good 或 AssessResult.Bad。
        """
        # 代码实现于此处。
        ...
```

**3) 实现脚本来运行 Assessor**

```python
import argparse

import CustomizedAssessor

def main():
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    # 在这里解析 Assessor 的参数。
    ...
    FLAGS, unparsed = parser.parse_known_args()

    tuner = CustomizedAssessor(...)
    tuner.run()

main()
```

注意 2) 中， 对象 `trial_history` 和 `report_intermediate_result` 函数返回给 Assessor 的完全一致。

也可以重载 Assessor 的 `run` 函数来控制过程逻辑。

更多示例，可参考：

> - [Base-Assessor](https://msrasrg.visualstudio.com/NeuralNetworkIntelligenceOpenSource/_git/Default?_a=contents&path=%2Fsrc%2Fsdk%2Fpynni%2Fnni%2Fassessor.py&version=GBadd_readme)