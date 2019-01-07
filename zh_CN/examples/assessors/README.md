# 自定义评估器

*评估器从尝试中接收中间结果，并决定此尝试是否应该终止。 一旦尝试满足提前终止条件，评估器将终止此尝试。*

因此，如果要自定义评估器，需要：

**1) 继承于 Assessor 基类，创建评估器类**

```python
from nni.assessor import Assessor

class CustomizedAssessor(Assessor):
    def __init__(self, ...):
        ...
```

**2) 实现评估尝试的函数**

```python
from nni.assessor import Assessor, AssessResult

class CustomizedAssessor(Assessor):
    def __init__(self, ...):
        ...

    def assess_trial(self, trial_history):
        """
        决定是否应该终止尝试。 必须重载。
        trial_history: 中间结果列表对象。
        返回 AssessResult.Good 或 AssessResult.Bad。
        """
        # 代码实现于此处。
        ...
```

**3) 实现脚本来运行评估器**

```python
import argparse

import CustomizedAssesor

def main():
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    # 在这里解析评估器的参数
    ...
    FLAGS, unparsed = parser.parse_known_args()

    tuner = CustomizedAssessor(...)
    tuner.run()

main()
```

注意 2) 中。 对象 ```trial_history``` are exact the object that Trial send to Assesor by using SDK ```report_intermediate_result``` function.

Also, user could override the ```run``` function in Assessor to control the process logic.

More detail example you could see:

> - [Base-Assessor](https://msrasrg.visualstudio.com/NeuralNetworkIntelligenceOpenSource/_git/Default?_a=contents&path=%2Fsrc%2Fsdk%2Fpynni%2Fnni%2Fassessor.py&version=GBadd_readme)