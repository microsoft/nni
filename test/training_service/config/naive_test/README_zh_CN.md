## 用法

* 安装前测试： `python3 run.py --preinstall`
* 安装的集成测试： `python3 run.py`
* 如果没有问题，最终会打印绿色的 `PASS`。

## 详细说明
* 这是测试 Trial 和 Tuner、Assessor 之间通信的测试用例。
* Trial 会收到整数 `x` 作为参数，并返回 `x`, `x²`, `x³`, ... , `x¹⁰` 作为指标。
* Tuner 会简单的生成自然数序列，并将收到的指标输出到 `tuner_result.txt`。
* 当 `sum(metrics) % 11 == 1` 时，Assessor 会终止 Trial，并将终止的 Trial 输出到 `assessor_result.txt`。
* 当 Tuner 和 Assessor 发生异常时，会在相应的文件中输出 `ERROR`。
* 当 Experiment 结束时，也表示用例成功执行，可以在 nni_manager.log 文件中找到 `Experiment done`。

## 问题
* 使用了私有 API 来检测是否 Tuner 和 Assessor 成功结束。
* RESTful 服务的输出未测试。
* 远程计算机训练平台没有被测试。