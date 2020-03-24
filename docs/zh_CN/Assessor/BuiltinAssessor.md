# 内置 Assessor

NNI provides state-of-the-art tuning algorithms within our builtin-assessors and makes them easy to use. Below is a brief overview of NNI's current builtin Assessors.

Note: Click the **Assessor's name** to get each Assessor's installation requirements, suggested usage scenario, and a config example. A link to a detailed description of each algorithm is provided at the end of the suggested scenario for each Assessor.

Currently, we support the following Assessors:

| Assessor                          | 算法简介                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**Medianstop**](#MedianStop)     | Medianstop 是一个简单的提前终止算法。 如果 Trial X 在步骤 S 的最好目标值低于所有已完成 Trial 前 S 个步骤目标平均值的中位数，这个 Trial 就会被提前停止。 [参考论文](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)                                                                                                                                                          |
| [**Curvefitting**](#Curvefitting) | Curve Fitting Assessor is an LPA (learning, predicting, assessing) algorithm. It stops a pending trial X at step S if the prediction of the final epoch's performance worse than the best final performance in the trial history. 此算法中采用了 12 种曲线来拟合精度曲线。 [参考论文](http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) |

## 用法

Usage of builtin assessors provided by the NNI SDK requires one to declare the **builtinAssessorName** and **classArgs** in the `config.yml` file. In this part, we will introduce the details of usage and the suggested scenarios, classArg requirements, and an example for each assessor.

Note: Please follow the provided format when writing your `config.yml` file.

<a name="MedianStop"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Median Stop Assessor`

> 名称：**Medianstop**

**建议场景**

It's applicable in a wide range of performance curves, thus, it can be used in various scenarios to speed up the tuning progress. [详细说明](./MedianstopAssessor.md)

**classArgs requirements:**

* **optimize_mode** (*maximize 或 minimize, 可选, 默认值为 maximize*) - 如果为 'maximize', Assessor 会在结果小于期望值时**终止** Trial。 如果为 'minimize'，Assessor 会在结果大于期望值时**终止** Trial。
* **start_step** (*int, optional, default = 0*) - A trial is determined to be stopped or not only after receiving start_step number of reported intermediate results.

**使用示例：**

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

**建议场景**

It's applicable in a wide range of performance curves, thus, it can be used in various scenarios to speed up the tuning progress. 更好的地方是，它能处理并评估性能类似的曲线。 [详细说明](./CurvefittingAssessor.md)

**classArgs requirements:**

* **epoch_num** (*int, **required***) - The total number of epochs. We need to know the number of epochs to determine which points we need to predict.
* **optimize_mode** (*maximize 或 minimize, 可选, 默认值为 maximize*) - 如果为 'maximize', Assessor 会在结果小于期望值时**终止** Trial。 如果为 'minimize'，Assessor 会在结果大于期望值时**终止** Trial。
* **start_step** (*int, optional, default = 6*) - A trial is determined to be stopped or not only after receiving start_step number of reported intermediate results.
* **threshold** (*float, optional, default = 0.95*) - The threshold that we use to decide to early stop the worst performance curve. For example: if threshold = 0.95, optimize_mode = maximize, and the best performance in the history is 0.9, then we will stop the trial who's predicted value is lower than 0.95 * 0.9 = 0.855.
* **gap** (*int, 可选, 默认值为 1*) - Assessor 两次评估之间的间隔次数。 For example: if gap = 2, start_step = 6, then we will assess the result when we get 6, 8, 10, 12...intermediate results.

**使用示例：**

```yaml
# config.yml
assessor:
    builtinAssessorName: Curvefitting
    classArgs:
      epoch_num: 20
      optimize_mode: maximize
      start_step: 6
      threshold: 0.95
      gap: 1
```