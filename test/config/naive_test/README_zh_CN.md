## 用法

* 安装前测试： `python3 run.py --preinstall`
* 安装的集成测试： `python3 run.py`
* 如果没有问题，最终会打印绿色的 `PASS`。

## 详细说明
* 这是测试 Trial 和 Tuner、Assessor 之间通信的测试用例。
* Trial 会收到整数 `x` 作为参数，并返回 `x`, `x²`, `x³`, ... , `x¹⁰` 作为指标。
* The naive tuner simply generates the sequence of natural numbers, and print received metrics to `tuner_result.txt`.
* The naive assessor kills trials when `sum(metrics) % 11 == 1`, and print killed trials to `assessor_result.txt`.
* When tuner and assessor exit with exception, they will append `ERROR` to corresponding result file.
* When the experiment is done, meaning it is successfully done in this case, `Experiment done` can be detected in the nni_manager.log file.

## Issues
* Private APIs are used to detect whether tuner and assessor have terminated successfully.
* The output of REST server is not tested.
* Remote machine training service is not tested.