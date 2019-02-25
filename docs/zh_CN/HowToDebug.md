# **在 NNI 中调试代码**

## 概述

NNI 中的日志分为三部分。 包括 NNI Manager， Dispatcher 以及 Trial。 这里会简单介绍这些组件。 更多信息可参考[概述](Overview.md)。

- **NNI Controller**：NNI Controller (nnictl) 是命令行工具，用来管理 Experiments（如：启动 Experiment）。
- **NNI Manager**：这是 NNI 的核心。当 Experiment 出现严重错误时，从它的日志中才能找到原因。（例如，Web 界面无法打开，或者训练服务失败）
- **Dispatcher**: Dispatcher 调用 **Tuner** 和 **Assessor** 的方法。 它的日志与 Tuner 和 Assessor 代码有关。 
    - **Tuner**: Tuner 是一个自动机器学习算法，会为下一个 Trial 生成新的配置。 新的 Trial 会使用这组配置来运行。
    - **Assessor**：Assessor 分析 Trial 的中间结果（例如，测试数据集上定期的精度），来确定 Trial 是否应该被提前终止。
- **Trial**：Trial 的代码是用户编写的代码，每次 Trial 运行时会尝试一组新的配置（例如，一组新的超参值，或者某个神经网络结构）。

## 日志的位置

NNI 中有三种日志。 在创建 Experiment 时，可增加命令行参数 `--debug`，将日志级别设置为 debug 级别。 此外，还可以在配置文件中使用 `logLevel` 来设置日志级别。 可设置的日志级别包括：`trace`, `debug`, `info`, `warning`, `error`, `fatal`。

### NNI Controller

在启动 NNI Experiment 时发生的错误，都可以在这里找到。

通过 `nnictl log stderr` 命令来查看错误信息。 参考 [NNICTL](NNICTLDOC.md) 了解更多命令选项。

### Experiment 根目录

每个 Experiment 都有一个根目录，会显示在 Web 界面的右上角。 如果无法打开 Web 界面，可将 `~/nni/experiment/experiment_id/` 中的 `experiment_id` 替换为实际的 Experiment ID，来组合出根目录。 `experiment_id` 可以在运行 `nnictl create ...` 来创建新 Experiment 的输出中找到。

> 如有需要，可以在配置文件中修改 `logDir`，来指定存储 Experiment 的目录。（默认为 `~/nni/experiment`）。 参考[配置](ExperimentConfig.md)文档，了解更多信息。

在此目录下，还会有另一个叫做 `log` 的目录，`nnimanager.log` 和 `dispatcher.log` 都在此目录中。

### Trial 根目录

在 Web 界面中，可通过点击每个 Trial 左边的 `+` 来展开详情并看到它的日志路径。

在 Experiment 的根目录中，会有一个 `trials` 目录，这里存放了所有 Trial 的信息。 每个 Trial 都有一个用其 ID 命名的目录。 目录中会有一个 `stderr` 文件，是 Trial 的错误信息。另一个 `trial.log` 文件是 Trial 的日志。

## 不同类型的错误

NNI 中有不同的错误类型。 根据严重程度，可分为三类。 当 NNI 中发生错误时，需要按顺序检查以下三种错误。

一般情况下，打开 Web 界面，可以在 `Overview` 标签的 `Status` 上看到错误信息。 如果 Web 界面无法打开，可以通过命令行来检查。

### **NNI** 失败

这是最严重的错误。 发生这种错误时，整个 Experiment 都会失败，Trial 也不会运行。 这通常是由安装问题导致的。

先检查 `nnictl` 的错误输出文件 `stderr` (运行 nnictl log stderr)，然后检查 `nnimanager` 的日志来看看是否由任何错误。

### **Dispatcher** Fails

Dispatcher fails. Usually, for some new users of NNI, it means that tuner fails. You could check dispatcher's log to see what happens to your dispatcher. For built-in tuner, some common errors might be invalid search space (unsupported type of search space or inconsistence between initializing args in configuration file and actual tuner's *\_init*\_ function args).

Take the later situation as an example. If you write a customized tuner who's *\_init*\_ function has an argument called `optimize_mode`, which you do not provide in your configuration file, NNI will fail to run your tuner so the experiment fails. You can see errors in the webUI like:

![](../img/dispatcher_error.jpg)

Here we can see it is a dispatcher error. So we can check dispatcher's log, which might look like:

    [2019-02-19 19:36:45] DEBUG (nni.main/MainThread) START
    [2019-02-19 19:36:47] ERROR (nni.main/MainThread) __init__() missing 1 required positional arguments: 'optimize_mode'
    Traceback (most recent call last):
      File "/usr/lib/python3.7/site-packages/nni/__main__.py", line 202, in <module>
        main()
      File "/usr/lib/python3.7/site-packages/nni/__main__.py", line 164, in main
        args.tuner_args)
      File "/usr/lib/python3.7/site-packages/nni/__main__.py", line 81, in create_customized_class_instance
        instance = class_constructor(**class_args)
    TypeError: __init__() missing 1 required positional arguments: 'optimize_mode'.
    

### **Trial** Fails

In this situation, NNI can still run and create new trials.

It means your trial code (which is run by NNI) fails. This kind of error is strongly related to your trial code. Please check trial's log to fix any possible errors shown there.

A common example of this would be run the mnist example without installing tensorflow. Surely there is an Import Error (that is, not installing tensorflow but trying to import it in your trial code) and thus every trial fails.

![](../img/trial_error.jpg)

As it shows, every trial has a log path, where you can find trial'log and stderr.