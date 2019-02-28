# 内置 Assessor

NNI 提供了先进的调优算法，使用上也很简单。 下面是内置 Assessor 的介绍：

注意：点击 **Assessor 的名称**可跳转到算法的详细描述，点击**用法**可看到 Assessor 的安装要求、建议场景和使用样例等等。

当前支持的 Assessor：

| Assessor                          | 算法简介                                                                                                                                                                                                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**Medianstop**](#MedianStop)     | Medianstop 是一个简单的提前终止算法。 如果 Trial X 的在步骤 S 的最好目标值比所有已完成 Trial 的步骤 S 的中位数值明显要低，这个 Trial 就会被提前停止。 [参考论文](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)                                                         |
| [**Curvefitting**](#Curvefitting) | Curve Fitting Assessor 是一个 LPA (learning, predicting, assessing，即学习、预测、评估) 的算法。 如果预测的 Trial X 在 step S 比性能最好的 Trial 要差，就会提前终止它。 此算法中采用了 12 种曲线来拟合精度曲线。 [参考论文](http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) |

## 用法

要使用 NNI 内置的 Assessor，需要在 `config.yml` 文件中添加 **builtinAssessorName** 和 **classArgs**。 这一节会介绍推荐的场景、参数等详细用法以及样例。

注意：参考样例中的格式来创建新的 `config.yml` 文件。

<a name="MedianStop"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Median Stop Assessor`

> 名称：**Medianstop**

**建议场景**

适用于各种性能曲线，可用到各种场景中来加速优化过程。

**参数**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', assessor will **stop** the trial with smaller expectation. If 'minimize', assessor will **stop** the trial with larger expectation.
* **start_step** (*int, optional, default = 0*) - A trial is determined to be stopped or not, only after receiving start_step number of reported intermediate results.

**使用样例：**

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

适用于各种性能曲线，可用到各种场景中来加速优化过程。 更好的地方是，它能处理并评估性能类似的曲线。

**参数**

* **epoch_num** (*int, **required***) - The total number of epoch. We need to know the number of epoch to determine which point we need to predict.
* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', assessor will **stop** the trial with smaller expectation. If 'minimize', assessor will **stop** the trial with larger expectation.
* **start_step** (*int, optional, default = 6*) - A trial is determined to be stopped or not, we start to predict only after receiving start_step number of reported intermediate results.
* **threshold** (*float, optional, default = 0.95*) - The threshold that we decide to early stop the worse performance curve. For example: if threshold = 0.95, optimize_mode = maximize, best performance in the history is 0.9, then we will stop the trial which predict value is lower than 0.95 * 0.9 = 0.855.
* **gap** (*int, optional, default = 1*) - The gap interval between Assesor judgements. For example: if gap = 2, start_step = 6, then we will assess the result when we get 6, 8, 10, 12...intermedian result.

**使用样例：**

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