# 内置 Assessor

NNI 提供了先进的调优算法，使用上也很简单。 下面是内置 Assessor 的介绍：

注意：点击 **Assessor 的名称**可跳转到算法的详细描述，点击**用法**可看到 Assessor 的安装要求、建议场景和使用样例等等。

Currently we support the following Assessors:

* [Medianstop](medianstopAssessor.md)
* [Curvefitting](curvefittingAssessor.md)

| Assessor                                     | 算法简介                                                                                                                                                                                                                                                      |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| __Medianstop__[(Usage)](#MedianStop)     | Medianstop 是一个简单的提前终止算法。 如果 Trial X 的在步骤 S 的最好目标值比所有已完成 Trial 的步骤 S 的中位数值明显要低，这个 Trial 就会被提前停止。 [参考论文](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)                                                         |
| __Curvefitting__[(Usage)](#Curvefitting) | Curve Fitting Assessor 是一个 LPA (learning, predicting, assessing，即学习、预测、评估) 的算法。 如果预测的 Trial X 在 step S 比性能最好的 Trial 要差，就会提前终止它。 此算法中采用了 12 种曲线来拟合精度曲线。 [参考论文](http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) |

## 用法

Use builtin assessors provided by NNI SDK requires to declare the **builtinAssessorName** and **classArgs** in `config.yml` file. In this part, we will introduce the detailed usage about the suggested scenarios, classArg requirements, and example for each assessor.

Note: Please follow the format when you write your `config.yml` file.

<a name="MedianStop"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Median Stop Assessor`

> 名称：**Medianstop**

**Suggested scenario**

It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress.

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', assessor will **stop** the trial with smaller expectation. If 'minimize', assessor will **stop** the trial with larger expectation.
* **start_step** (*int, optional, default = 0*) - A trial is determined to be stopped or not, only after receiving start_step number of reported intermediate results.

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

* **epoch_num** (*int, **required***) - The total number of epoch. We need to know the number of epoch to determine which point we need to predict.
* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', assessor will **stop** the trial with smaller expectation. If 'minimize', assessor will **stop** the trial with larger expectation.
* **start_step** (*int, optional, default = 6*) - A trial is determined to be stopped or not, we start to predict only after receiving start_step number of reported intermediate results.
* **threshold** (*float, optional, default = 0.95*) - The threshold that we decide to early stop the worse performance curve. For example: if threshold = 0.95, optimize_mode = maximize, best performance in the history is 0.9, then we will stop the trial which predict value is lower than 0.95 * 0.9 = 0.855.

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