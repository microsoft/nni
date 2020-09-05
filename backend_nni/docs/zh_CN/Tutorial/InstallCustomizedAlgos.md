**如何将自定义的算法安装为内置的 Tuner，Assessor 和 Advisor**
===

## 概述

NNI 提供了大量可用于超参优化的[内置 Tuner](../Tuner/BuiltinTuner.md), [Advisor](../Tuner/HyperbandAdvisor.md) 以及 [Assessor](../Assessor/BuiltinAssessor.md)，其它算法可在 NNI 安装后，通过 `nnictl package install --name <name>` 安装。 可通过 `nnictl package list` 命令查看其它算法。

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
要作为内置 Tuner，Assessor，Advisor 安装，自定义算法需要用 `pip` 命令能识别的方法来打包，NNI 会调用 `pip` 命令来安装包。 除了能作为公共的 pip 源，包需要在 `classifiers` 字段中提供元信息。 classifiers 字段格式如下：
```
NNI Package :: <type> :: <builtin name> :: <full class name of tuner> :: <full class name of class args validator>
```
* `type`: 算法类型，可为 `tuner`, `assessor`, `advisor`
* `builtin name`: 在 Experiment 配置文件中使用的内置名称
* `full class name of tuner`: Tuner 类名，包括模块名，如：`demo_tuner.DemoTuner`
* `full class name of class args validator`: 类的参数验证类 validator 的类名，包括模块名，如：`demo_tuner.MyClassArgsValidator`

安装包 `setup.py` 中的 classfiers 示例如下：

```python
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: ',
        'NNI Package :: tuner :: demotuner :: demo_tuner.DemoTuner :: demo_tuner.MyClassArgsValidator'
    ],
```

在 `setup.py` 中加入元信息后，可构建 pip 安装源：
* 在包目录中运行 `python setup.py develop` 命令，此命令会将目录作为 pip 安装源。
* 在包目录中运行 `python setup.py bdist_wheel` 命令，会构建 whl 文件。

在通过 `nnictl package install <source>` 命令安装时，NNI 会查找 `NNI Package` 开头的 classifier，获取包的元信息。

参考[自定义 Tuner 的完整示例](https://github.com/microsoft/nni/blob/master/examples/tuners/customized_tuner/README.md)。

### 4. 将自定义算法包安装到 NNI 中

如果安装源是通过 `python setup.py develop` 准备的源代码目录，可通过下列命令安装：

`nnictl package install <安装源目录>`

例如：

`nnictl package install nni/examples/tuners/customized_tuner/`

如果安装源是通过 `python setup.py bdist_wheel` 准备的 whl 文件，可通过下列命令安装：

`nnictl package install <whl 文件路径>`

例如：

`nnictl package install nni/examples/tuners/customized_tuner/dist/demo_tuner-0.1-py3-none-any.whl`

## 5. 在 Experiment 中使用安装的算法
在自定义算法安装后，可用其它内置 Tuner、Assessor、Advisor 的方法在 Experiment 配置文件中使用，例如：

```yaml
tuner:
  builtinTunerName: demotuner
  classArgs:
    #可选项: maximize, minimize
    optimize_mode: maximize
```


## 用 `nnictl package` 命令管理包

### 列出已安装的包

运行以下命令列出已安装的包：

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

运行以下命令列出包括不能卸载的所有包。

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

### 卸载包

运行以下命令卸载已安装的包：

`nnictl package uninstall <包名称>`

例如：

`nnictl package uninstall demotuner`
