**How to install customized algorithms as builtin tuners, assessors and advisors**
===

## Overview

NNI provides a lot of [builtin tuners](../Tuner/BuiltinTuner.md), [advisors](../Tuner/BuiltinTuner.md#Hyperband) and [assessors](../Assessor/BuiltinAssessor.md) can be used directly for Hyper Parameter Optimization, and some extra algorithms can be installed via `nnictl package install --name <name>` after NNI is installed. You can check these extra algorithms via `nnictl package list` command.

NNI also provides the ability to build your own [customized tuners](../Tuner/CustomizeTuner.md), [advisors](../Tuner/CustomizeAdvisor.md) and [assessors](../Assessor/CustomizeAssessor.md).

Customized tuners/advisors/assessors can be installed into NNI as builtin algorithms, once they are installed into NNI, you can use your customized algorithms the same way as builtin tuners/advisorsa/assessors in your experiment configuration file. For example, you built a customized tuner and installed it into NNI using a builtin name `mytuner`, then you can use this tuner in your configuration file like below:
```yaml
tuner:
  builtinTunerName: mytuner
```

## Install customized algorithms as builtin tuners, assessors and advisors
You can follow below steps to build a customized tuner/assessor/advisor, and install it into NNI as builtin algorithm.

### 1. Create a customized tuner/assessor/advisor
Reference following instructions to create:
* [customized tuner](../Tuner/CustomizeTuner.md)
* [customized assessor](../Assessor/CustomizeAssessor.md)
* [customized advisor](../Tuner/CustomizeAdvisor.md)

### 2. (Optional) Create a validator to validate classArgs
NNI provides a `ClassArgsValidator` interface for customized algorithms author to validate the classArgs parameters in experiment configuration file which are passed to customized algorithms constructors.
The `ClassArgsValidator` interface is defined as:
```python
class ClassArgsValidator(object):
    def validate_class_args(self, **kwargs):
        """
        The classArgs fields in experiment configuration are packed as a dict and
        passed to validator as kwargs.
        """
        pass
```
For example, you can implement your validator such as:
```python
from schema import Schema, Optional
from nni import ClassArgsValidator

class MedianstopClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            Optional('optimize_mode'): self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('start_step'): self.range('start_step', int, 0, 9999),
        }).validate(kwargs)
```
The validator will be invoked before experiment is started to check whether the classArgs fields are valid for your customized algorithms.

### 3. Prepare package installation source
In order to be installed as builtin tuners, assessors and advisors, the customized algorithms need to be packaged as installable source which can be recognized by `pip` command, under the hood nni calls `pip` command to install the package.
Besides being a common pip source, the package needs to provide meta information in the `classifiers` field.


### 4. Install customized algorithms package into NNI

### Manage packages using `nnictl package`
