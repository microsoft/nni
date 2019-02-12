# 内置 Assessor

NNI 提供了先进的调优算法，使用上也很简单。 下面是内置 Assessor 的介绍：

Note: Click the **Assessor's name** to get a detailed description of the algorithm, click the corresponding **Usage** to get the Assessor's installation requirements, suggested scenario and using example.

| Assessor                                                                                      | 算法简介                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Medianstop](../src/sdk/pynni/nni/medianstop_assessor/README.md) [(用法)](#MedianStop)          | Medianstop is a simple early stopping rule. 如果 Trial X 的在步骤 S 的最好目标值比所有已完成 Trial 的步骤 S 的中位数值明显要低，就会停止运行 Trial X。 [参考论文](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)                                                                               |
| [Curvefitting](../src/sdk/pynni/nni/curvefitting_assessor/README.md) [(Usage)](#Curvefitting) | Curve Fitting Assessor 是一个 LPA (learning, predicting, assessing，即学习、预测、评估) 的算法。 如果预测的 Trial X 在 step S 比性能最好的 Trial要差，就会提前终止它。 In this algorithm, we use 12 curves to fit the accuracy curve. [参考论文](http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) |

## 用法

Use builtin assessors provided by NNI SDK requires to declare the **builtinAssessorName** and **classArgs** in `config.yml` file. In this part, we will introduce the detailed usage about the suggested scenarios, classArg requirements, and example for each assessor.

Note: Please follow the format when you write your `config.yml` file.

<a name="MedianStop"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Median Stop Assessor`

> 名称：**Medianstop**

**Suggested scenario**

It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress.

**Requirement of classArg**

* **optimize_mode** (*maximize 或 minimize, 可选, 默认值为 maximize*) - 如果为 'maximize', Assessor 会在结果小于期望值时**终止** Trial。 如果为 'minimize'，Assessor 会在结果大于期望值时**终止** Trial。
* **start_step** (*int, 可选, 默认值为 0*) - 只有收到 start_step 个中间结果后，才开始判断是否一个 Trial 应该被终止。

**Usage example:**

```yaml
# config.yml
assessor:
    builtinAssessorName: Medianstop
    classArgs:
      optimize_mode: maximize
      start_step: 5
```

<br />

<a name="Curvefitting"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Curve Fitting Assessor`

> 名称：**Curvefitting**

**Suggested scenario**

It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress. Even better, it's able to handle and assess curves with similar performance.

**Requirement of classArg**

* **epoch_num** (*int, **必需***) - epoch 的总数。 需要此数据来决定需要预测点的总数。
* **optimize_mode** (*maximize 或 minimize, 可选, 默认值为 maximize*) - 如果为 'maximize', Assessor 会在结果小于期望值时**终止** Trial。 如果为 'minimize'，Assessor 会在结果大于期望值时**终止** Trial。
* **start_step** (*int, 可选, 默认值为 6*) - 只有收到 start_step 个中间结果后，才开始判断是否一个 Trial 应该被终止。
* **threshold** (*float, 可选, 默认值为 0.95*) - 用来确定提前终止较差结果的阈值。 例如，如果 threshold = 0.95, optimize_mode = maximize，最好的历史结果是 0.9，那么会在 Trial 的预测值低于 0.95 * 0.9 = 0.855 时停止。

**Usage example:**

```yaml
# config.yml
assessor:
    builtinAssessorName: Curvefitting
    classArgs:
      epoch_num: 20
      optimize_mode: maximize
      start_step: 6
      threshold: 0.95
```