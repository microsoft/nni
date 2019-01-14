# Assessors

## Overview

In order to save our computing resources, NNI supports an early stop policy and creates **Assessor** to finish this job.

Assessor receives intermediate result from Trial and decides whether the Trial should be killed by specific algorithm. Once the Trial experiment meets the early stop conditions(which means assessor is pessimistic about the final results), the assessor will kill the trial and the status of experiement will be `"EARLY_STOPPED"`.

Here is an experimental result of MNIST after using 'Curvefitting' Assessor in 'maximize' mode, you can see that assessor successfully **early stopped** many trials with bad hyperparameters in advance. If you use assessor, we may get a better hyperparameters under the same computing resources.

*Implemented code directory: [config_assessor.yml][5]*

![](./img/Assessor.png)

NNI provides the-state-of-art tuning algorithm in our builtin-assessors, and makes them easy to use. Below is the brief overview of NNI current builtin Assessors:

|Assessor|Brief Introduction of Algorithm|
|---|---|
|**Medianstop**<br>[(Usage)](#MedianStop)|Medianstop is a simple early stopping rule mentioned in the [paper][1]. It stops a pending trial X at step S if the trial’s best objective value by step S is strictly worse than the median value of the running averages of all completed trials’ objectives reported up to step S.|It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress.|
|[Curvefitting][2]<br>[(Usage)](#Curvefitting)|Curve Fitting Assessor is a LPA(learning, predicting, assessing) algorithm. It stops a pending trial X at step S if the prediction of final epoch's performance worse than the best final performance in the trial history. In this algorithm, we use 12 curves to fit the accuracy curve|It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress. Even better, it's able to handle and assess curves with similar performance.|

<br>

## Usage of Builtin Assessors

Use builtin assessors provided by NNI sdk requires to declare the  **builtinAssessorName** and **classArgs** in `config.yml` file. In this part, we will introduce the detailed usage about the suggested scenarios, classArg requirments and example for each assessor.

Note: Please follow the format when you write your `config.yml` file.

<a name="MedianStop"></a>

![#1589F0](https://placehold.it/15/1589F0/000000?text=+) `Median Stop Assessor`

> Builtin Assessor Name: **Medianstop**

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

<br>

<a name="Curvefitting"></a>

![#1589F0](https://placehold.it/15/1589F0/000000?text=+) `Curve Fitting Assessor`

> Builtin Assessor Name: **Curvefitting**

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

[1]: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf
[2]: https://github.com/Microsoft/nni/blob/master/src/sdk/pynni/nni/curvefitting_assessor/README.md
[5]: https://github.com/Microsoft/nni/blob/master/examples/trials/mnist/config_assessor.yml