# Define your own Assessor

*Assessor receive intermediate result from Trial and decide whether the Trial should be killed. Once the Trial experiment meets the early stop conditions, the assessor will kill the Trial.*

So, if users want to implement a customized Assessor, they only need to:


**1) Inherit an assessor of a base Assessor class**
```python
from nni.assessor import Assessor

class CustomizedAssessor(Assessor):
    def __init__(self, ...):
        ...
```

**2) Implement assess trial function**
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
**3) Write a script to run Assessor**
```python
import argparse

import CustomizedAssessor

def main():
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    # parse your assessor arg here.
    ...
    FLAGS, unparsed = parser.parse_known_args()

    tuner = CustomizedAssessor(...)
    tuner.run()

main()
```

Please noted in 2). The object `trial_history` are exact the object that Trial send to Assessor by using SDK `report_intermediate_result` function.

Also, user could override the `run` function in Assessor to control the process logic.

More detail example you could see:
> * [Base-Assessor](https://msrasrg.visualstudio.com/NeuralNetworkIntelligenceOpenSource/_git/Default?_a=contents&path=%2Fsrc%2Fsdk%2Fpynni%2Fnni%2Fassessor.py&version=GBadd_readme)
