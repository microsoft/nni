**如何将自定义的算法安装为内置的 Tuner，Assessor 和 Advisor**
===

## 概述

NNI 提供了大量可用于超参优化的[内置 Tuner](../Tuner/BuiltinTuner.md), [Advisor](../Tuner/BuiltinTuner.md#Hyperband) 以及 [Assessor](../Assessor/BuiltinAssessor.md)，其它算法可在 NNI 安装后，通过 `nnictl package install --name <name>` 安装。 可通过 `nnictl package list` 命令查看其它算法。

NNI 中，还可以创建自定义的 Tuner，Advisor 和 Assessor。 并根据 Experiment 配置文件的说明来使用这些自定义的算法，可参考 [自定义 Tuner](../Tuner/CustomizeTuner.md)/[Advisor](../Tuner/CustomizeAdvisor.md)/[Assessor](../Assessor/CustomizeAssessor.md)。

用户可将自定义的算法作为内置算法安装，以便像其它内置 Tuner、Advisor、Assessor 一样使用。 更重要的是，这样更容易向其他人分享或发布自己实现的算法。 自定义的 Tuner、Advisor、Assessor 可作为内置算法安装到 NNI 中，安装完成后，可在 Experiment 配置文件中像内置算法一样使用。 例如，将自定义的算法 `mytuner` 安装到 NNI 后，可在配置文件中直接使用：
```yaml
tuner:
  builtinTunerName: mytuner
```

## 将自定义的算法安装为内置的 Tuner，Assessor 或 Advisor
可参考下列步骤来构建自定义的 Tuner、Assessor、Advisor，并作为内置算法安装。

### 1. 创建自定义的 Tuner、Assessor、Advisor
参考下列说明来创建：
* [自定义 Tuner](../Tuner/CustomizeTuner.md)
* [自定义 Assessor](../Assessor/CustomizeAssessor.md)
* [自定义 Advisor](../Tuner/CustomizeAdvisor.md)

### 2. (可选) 创建 Validator 来验证 classArgs
NNI 提供了 `ClassArgsValidator` 接口，自定义的算法可用它来验证 Experiment 配置文件中传给构造函数的 classArgs 参数。 `ClassArgsValidator` 接口如下：
```python
class ClassArgsValidator(object):
    def validate_class_args(self, **kwargs):
        """
        Experiment 配置中的 classArgs 字段会作为 dict
        传入到 kwargs。
        """
        pass
```
例如，可将 Validator 如下实现：
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
在 Experiment 启动时，会调用 Validator，检查 classArgs 字段是否正确。

### 3. 准备安装源
In order to be installed as builtin tuners, assessors and advisors, the customized algorithms need to be packaged as installable source which can be recognized by `pip` command, under the hood nni calls `pip` command to install the package. Besides being a common pip source, the package needs to provide meta information in the `classifiers` field. Format of classifiers field is a following:
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

Reference [customized tuner example](https://github.com/microsoft/nni/blob/master/examples/tuners/customized_tuner/README.md) for a full example.

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
