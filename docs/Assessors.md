# Assessors

## Overview

In order to save our computing resources, NNI supports an early stop policy and creates **Assessor** to finish this job.

Assessor receives intermediate result from Trial and decides whether the Trial should be killed. Once the Trial experiment meets the early stop conditions, the assessor will kill the Trial.

In other words, assesor uses the intermediate results and evaluates the results by specific algorithm. If assessor is pessimistic about the final results, assessor will stop this trial and the status of experiement will be `"Early Stoped"`.

In NNI, we support two approaches to set the assessor.

1. Directly use assessor provided by nni sdk

        required fields: builtinAssessorName and classArgs.

2. Customize your own assessor file

        required fields: codeDirectory, classFileName, className and classArgs.

For now, NNI has supported the following assessor algorithms:

|Assessor|Brief introduction to the algorithm|Suggested scenario|Reference|
|---|---|---|---|
|**Medianstop**|Medianstop is a simple early stopping rule mentioned. It stops a pending trial X at step S if the trial’s best objective value by step S is strictly worse than the median value of the running averages of all completed trials’ objectives reported up to step S.|It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress.|Usage: [Medianstop Usage][1] Paper: [Google Vizier: A Service for Black-Box Optimization][2]|
|**Curvefitting**|Curve Fitting Assessor is a LPA(learning, predicting, assessing) algorithm. It stops a pending trial X at step S if the prediction of final epoch's performance worse than the best final performance in the trial history. In this algorithm, we use 12 curves to fit the accuracy curve|It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress. Even better, it's able to handle and assess curves with similar performance.|Usage: [Curvefitting Usage][3] Paper:[Speeding up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves][4]|

## Try Different Assessors

### Example of Builtin Assessor Usage

Our NNI integrates state-of-the-art assessing algorithm. You can easily use our builtin assessors by declare the `builtinAssessorName` and `classArguments` in config file.

For example, if you chose to use "Medianstop" assessor, you can set the `config.yml` like this:

```yaml
assessor:
    builtinAssessorName: Medianstop
    classArgs:
      #choice: maximize, minimize
      optimize_mode: maximize
      # (optional) A trial is determined to be stopped or not, 
      * only after receiving start_step number of reported intermediate results.
      * The default value of start_step is 0.
      start_step: 5
```

If you chose to use "Curvefitting" assessor, you can set the `config.yml` like this:

```yaml
assessor:
    builtinAssessorName: Curvefitting
    classArgs:
      # (required)The total number of epoch.
      # We need to know the number of epoch to determine which point we need to predict.
      epoch_num: 20
      # (optional) choice: maximize, minimize
      # Kindly reminds that if you choose minimize mode, please adjust the value of threshold >= 1.0 (e.g threshold=1.1)
      * The default value of optimize_mode is maximize
      optimize_mode: maximize
      # (optional) A trial is determined to be stopped or not
      # In order to save our computing resource, we start to predict when we have more than start_step(default=6) accuracy points.
      # only after receiving start_step number of reported intermediate results.
      * The default value of start_step is 6.
      start_step: 6
      # (optional) The threshold that we decide to early stop the worse performance curve.
      # For example: if threshold = 0.95, optimize_mode = maximize, best performance in the history is 0.9, then we will stop the trial which predict value is lower than 0.95 * 0.9 = 0.855.
      * The default value of threshold is 0.95.
      threshold: 0.95
```

Note:

* Please write the .yml file in this format correctly.

### Requirements of each Assessor

According to different usage scenarios and requirements, we encourage users to use different assessors to better adjust the hyper-parameters. The following lists the names of our current builtin assessor and the corresponding classArg.

Notes:

* The `classArg` in `Bold` is **Requried**, must be assigned when using.
* Other cases of `classArg` below is **Optional**, we show the default value.
* Keywords **Unsupported** means we don't support this classArg.

|Assessor|builtinAssessorName|optimize_mode|start_step|Unique classArg|
|---|---|---|---|---|
|**Medianstop**|Medianstop|'maximize'|5||
|**Curvefitting**|Curvefitting|'maximize'|6|['**epoch_num**']:int, ['threshold']:float|

## Customize Assessor

If you want to implement a customized Assessor, you only need to:

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

import CustomizedAssesor

def main():
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    # parse your assessor arg here.
    ...
    FLAGS, unparsed = parser.parse_known_args()

    assessor = CustomizedAssessor(...)
    assessor.run()

main()
```

Please noted in **2**. The object ```trial_history``` are exact the object that Trial send to Assesor by using SDK ```report_intermediate_result``` function.

Also, user could override the ```run``` function in Assessor to control the process logic.

More detail example you could see:
> * [Base-Assessor](https://msrasrg.visualstudio.com/NeuralNetworkIntelligenceOpenSource/_git/Default?_a=contents&path=%2Fsrc%2Fsdk%2Fpynni%2Fnni%2Fassessor.py&version=GBadd_readme)

[1]: https://github.com/Microsoft/nni/blob/5b5861e9073ad591e0b761af940c52d930c5007a/docs/HowToChooseTuner.md
[2]: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf
[3]: https://github.com/Microsoft/nni/blob/5b5861e9073ad591e0b761af940c52d930c5007a/docs/HowToChooseTuner.md
[4]: http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf