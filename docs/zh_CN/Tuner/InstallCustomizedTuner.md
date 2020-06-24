# 如何将自定义的 Tuner 安装为内置 Tuner

参考下列步骤将自定义 Tuner： `nni/examples/tuners/customized_tuner` 安装为内置 Tuner。

## 准备安装源和安装包

有两种方法安装自定义的 Tuner：

### 方法 1: 从目录安装

Step 1: From `nni/examples/tuners/customized_tuner` directory, run:

`python setup.py develop`

This command will build the `nni/examples/tuners/customized_tuner` directory as a pip installation source.

Step 2: Run command:

`nnictl package install ./`

### Option 2: install from whl file

Step 1: From `nni/examples/tuners/customized_tuner` directory, run:

`python setup.py bdist_wheel`

This command build a whl file which is a pip installation source.

Step 2: Run command:

`nnictl package install dist/demo_tuner-0.1-py3-none-any.whl`

## Check the installed package

Then run command `nnictl package list`, you should be able to see that demotuner is installed:
```
+-----------------+------------+-----------+--------=-------------+------------------------------------------+
|      Name       |    Type    | Installed |      Class Name      |               Module Name                |
+-----------------+------------+-----------+----------------------+------------------------------------------+
| demotuner       | tuners     | Yes       | DemoTuner            | demo_tuner                               |
+-----------------+------------+-----------+----------------------+------------------------------------------+
```

## Use the installed tuner in experiment

Now you can use the demotuner in experiment configuration file the same way as other builtin tuners:

```yaml
tuner:
  builtinTunerName: demotuner
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
```
