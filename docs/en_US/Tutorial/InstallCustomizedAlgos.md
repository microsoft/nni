**How to install customized algorithms as builtin tuners, assessors and advisors**
===

## Overview

NNI provides a lot of [builtin tuners](../Tuner/BuiltinTuner.md), [advisors](../Tuner/HyperbandAdvisor.md) and [assessors](../Assessor/BuiltinAssessor.md) can be used directly for Hyper Parameter Optimization, and some extra algorithms can be installed via `nnictl package install --name <name>` after NNI is installed. You can check these extra algorithms via `nnictl package list` command.

NNI also provides the ability to build your own customized tuners, advisors and assessors. To use the customized algorithm, users can simply follow the spec in experiment config file to properly reference the algorithm, which has been illustrated in the tutorials of [customized tuners](../Tuner/CustomizeTuner.md)/[advisors](../Tuner/CustomizeAdvisor.md)/[assessors](../Assessor/CustomizeAssessor.md).

NNI also allows users to install the customized algorithm as a builtin algorithm, in order for users to use the algorithm in the same way as NNI builtin tuners/advisors/assessors. More importantly, it becomes much easier for users to share or distribute their implemented algorithm to others. Customized tuners/advisors/assessors can be installed into NNI as builtin algorithms, once they are installed into NNI, you can use your customized algorithms the same way as builtin tuners/advisors/assessors in your experiment configuration file. For example, you built a customized tuner and installed it into NNI using a builtin name `mytuner`, then you can use this tuner in your configuration file like below:
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
Format of classifiers field is a following:
```
NNI Package :: <type> :: <builtin name> :: <full class name of tuner> :: <full class name of class args validator>
```
* `type`: type of algorithms, could be one of `tuner`, `assessor`, `advisor`
* `builtin name`: builtin name used in experiment configuration file
* `full class name of tuner`: tuner class name, including its module name, for example: `demo_tuner.DemoTuner`
* `full class name of class args validator`: class args validator class name, including its module name, for example: `demo_tuner.MyClassArgsValidator`

Following is an example of classfiers in package's `setup.py`:

```python
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: ',
        'NNI Package :: tuner :: demotuner :: demo_tuner.DemoTuner :: demo_tuner.MyClassArgsValidator'
    ],
```

Once you have the meta info in `setup.py`, you can build your pip installation source via:
* Run command `python setup.py develop` from the package directory, this command will build the directory as a pip installation source.
* Run command `python setup.py bdist_wheel` from the package directory, this command build a whl file which is a pip installation source.

NNI will look for the classifier starts with `NNI Package` to retrieve the package meta information while the package being installed with `nnictl package install <source>` command.

Reference [customized tuner example](../Tuner/InstallCustomizedTuner.md) for a full example.

### 4. Install customized algorithms package into NNI

If your installation source is prepared as a directory with `python setup.py develop`, you can install the package by following command:

`nnictl package install <installation source directory>`

For example:

`nnictl package install nni/examples/tuners/customized_tuner/`

If your installation source is prepared as a whl file with `python setup.py bdist_wheel`, you can install the package by following command:

`nnictl package install <whl file path>`

For example:

`nnictl package install nni/examples/tuners/customized_tuner/dist/demo_tuner-0.1-py3-none-any.whl`

## 5. Use the installed builtin algorithms in experiment
Once your customized algorithms is installed, you can use it in experiment configuration file the same way as other builtin tuners/assessors/advisors, for example:

```yaml
tuner:
  builtinTunerName: demotuner
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
```


## Manage packages using `nnictl package`

### List installed packages

Run following command to list the installed packages:

```
nnictl package list
+-----------------+------------+-----------+--------=-------------+------------------------------------------+
|      Name       |    Type    | Installed |      Class Name      |               Module Name                |
+-----------------+------------+-----------+----------------------+------------------------------------------+
| demotuner       | tuners     | Yes       | DemoTuner            | demo_tuner                               |
| SMAC            | tuners     | No        | SMACTuner            | nni.smac_tuner.smac_tuner                |
| PPOTuner        | tuners     | No        | PPOTuner             | nni.ppo_tuner.ppo_tuner                  |
| BOHB            | advisors   | Yes       | BOHB                 | nni.bohb_advisor.bohb_advisor            |
+-----------------+------------+-----------+----------------------+------------------------------------------+
```

Run following command to list all packages, including the builtin packages can not be uninstalled.

```
nnictl package list --all
+-----------------+------------+-----------+--------=-------------+------------------------------------------+
|      Name       |    Type    | Installed |      Class Name      |               Module Name                |
+-----------------+------------+-----------+----------------------+------------------------------------------+
| TPE             | tuners     | Yes       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
| Random          | tuners     | Yes       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
| Anneal          | tuners     | Yes       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
| Evolution       | tuners     | Yes       | EvolutionTuner       | nni.evolution_tuner.evolution_tuner      |
| BatchTuner      | tuners     | Yes       | BatchTuner           | nni.batch_tuner.batch_tuner              |
| GridSearch      | tuners     | Yes       | GridSearchTuner      | nni.gridsearch_tuner.gridsearch_tuner    |
| NetworkMorphism | tuners     | Yes       | NetworkMorphismTuner | nni.networkmorphism_tuner.networkmo...   |
| MetisTuner      | tuners     | Yes       | MetisTuner           | nni.metis_tuner.metis_tuner              |
| GPTuner         | tuners     | Yes       | GPTuner              | nni.gp_tuner.gp_tuner                    |
| PBTTuner        | tuners     | Yes       | PBTTuner             | nni.pbt_tuner.pbt_tuner                  |
| SMAC            | tuners     | No        | SMACTuner            | nni.smac_tuner.smac_tuner                |
| PPOTuner        | tuners     | No        | PPOTuner             | nni.ppo_tuner.ppo_tuner                  |
| Medianstop      | assessors  | Yes       | MedianstopAssessor   | nni.medianstop_assessor.medianstop_...   |
| Curvefitting    | assessors  | Yes       | CurvefittingAssessor | nni.curvefitting_assessor.curvefitt...   |
| Hyperband       | advisors   | Yes       | Hyperband            | nni.hyperband_advisor.hyperband_adv...   |
| BOHB            | advisors   | Yes       | BOHB                 | nni.bohb_advisor.bohb_advisor            |
+-----------------+------------+-----------+----------------------+------------------------------------------+
```

### Uninstall package

Run following command to uninstall an installed package:

`nnictl package uninstall <builtin name>`

For example:

`nnictl package uninstall demotuner`
