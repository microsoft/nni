# Customize-Assessor

## Customize Assessor

NNI also support building an assessor by yourself to adjust your tuning demand.

If you want to implement a customized Assessor, there are three things for you to do:

1) Inherit an assessor of a base Assessor class
2) Implement assess_trial function
3) Write a script to run Assessor

**1. Inherit an assessor of a base Assessor class**

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
        Determines whether a trial should be killed. Must override.
        trial_history: a list of intermediate result objects.
        Returns AssessResult.Good or AssessResult.Bad.
        """
        # you code implement here.
        ...
```

**3. Write a script to run Assessor**

```python
import argparse

import CustomizedAssessor

def main():
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    # parse your assessor arg here.
    ...
    FLAGS, unparsed = parser.parse_known_args()

    assessor = CustomizedAssessor(...)
    assessor.run()

main()
```

Please noted in **2**. The object `trial_history` are exact the object that Trial send to Assessor by using SDK `report_intermediate_result` function.

Also, user could override the `run` function in Assessor to control the processing logic.

More detail example you could see:
> * [medianstop-assessor](../src/sdk/pynni/nni/medianstop_assessor)
> * [curvefitting-assessor](../src/sdk/pynni/nni/curvefitting_assessor)