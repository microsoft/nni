# Built-in Assessors

NNI provides state-of-the-art tuning algorithm in our builtin-assessors and makes them easy to use. Below is the brief overview of NNI current builtin Assessors:

Note: Click the **Assessor's name** to get the Assessor's installation requirements, suggested scenario and using example. The link for a detailed description of the algorithm is at the end of the suggested scenario of each Assessor.

Currently we support the following Assessors:

|Assessor|Brief Introduction of Algorithm|
|---|---|
|[__Medianstop__](#MedianStop)|Medianstop is a simple early stopping rule. It stops a pending trial X at step S if the trial’s best objective value by step S is strictly worse than the median value of the running averages of all completed trials’ objectives reported up to step S. [Reference Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)|
|[__Curvefitting__](#Curvefitting)|Curve Fitting Assessor is a LPA(learning, predicting, assessing) algorithm. It stops a pending trial X at step S if the prediction of final epoch's performance worse than the best final performance in the trial history. In this algorithm, we use 12 curves to fit the accuracy curve. [Reference Paper](http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf)|

## Usage of Builtin Assessors

Use builtin assessors provided by NNI SDK requires to declare the  **builtinAssessorName** and **classArgs** in `config.yml` file. In this part, we will introduce the detailed usage about the suggested scenarios, classArg requirements, and example for each assessor.

Note: Please follow the format when you write your `config.yml` file.

<a name="MedianStop"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Median Stop Assessor`

> Builtin Assessor Name: **Medianstop**

**Suggested scenario**

It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress. [Detailed Description](./MedianstopAssessor.md)

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

![](https://placehold.it/15/1589F0/000000?text=+) `Curve Fitting Assessor`

> Builtin Assessor Name: **Curvefitting**

**Suggested scenario**

It is applicable in a wide range of performance curves, thus, can be used in various scenarios to speed up the tuning progress. Even better, it's able to handle and assess curves with similar performance. [Detailed Description](./CurvefittingAssessor.md)

**Requirement of classArg**

* **epoch_num** (*int, **required***) - The total number of epoch. We need to know the number of epoch to determine which point we need to predict.
* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', assessor will **stop** the trial with smaller expectation. If 'minimize', assessor will **stop** the trial with larger expectation.
* **start_step** (*int, optional, default = 6*) - A trial is determined to be stopped or not, we start to predict only after receiving start_step number of reported intermediate results.
* **threshold** (*float, optional, default = 0.95*) - The threshold that we decide to early stop the worse performance curve. For example: if threshold = 0.95, optimize_mode = maximize, best performance in the history is 0.9, then we will stop the trial which predict value is lower than 0.95 * 0.9 = 0.855.
* **gap** (*int, optional, default = 1*) - The gap interval between Assesor judgements. For example: if gap = 2, start_step = 6, then we will assess the result when we get 6, 8, 10, 12...intermedian result.

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
      gap: 1
```