# Built-in Assessors

NNI provides state-of-the-art tuning algorithms within our builtin-assessors and makes them easy to use. Below is a brief overview of NNI's current builtin Assessors.

Note: Click the **Assessor's name** to get each Assessor's installation requirements, suggested usage scenario, and a config example. A link to a detailed description of each algorithm is provided at the end of the suggested scenario for each Assessor.

Currently, we support the following Assessors:

|Assessor|Brief Introduction of Algorithm|
|---|---|
|[__Medianstop__](#MedianStop)|Medianstop is a simple early stopping rule. It stops a pending trial X at step S if the trial’s best objective value by step S is strictly worse than the median value of the running averages of all completed trials’ objectives reported up to step S. [Reference Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)|
|[__Curvefitting__](#Curvefitting)|Curve Fitting Assessor is an LPA (learning, predicting, assessing) algorithm. It stops a pending trial X at step S if the prediction of the final epoch's performance worse than the best final performance in the trial history. In this algorithm, we use 12 curves to fit the accuracy curve. [Reference Paper](http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf)|

## Usage of Builtin Assessors

Usage of builtin assessors provided by the NNI SDK requires one to declare the  **builtinAssessorName** and **classArgs** in the `config.yml` file. In this part, we will introduce the details of usage and the suggested scenarios, classArg requirements, and an example for each assessor.

Note: Please follow the provided format when writing your `config.yml` file.

<a name="MedianStop"></a>

### Median Stop Assessor

> Builtin Assessor Name: **Medianstop**

**Suggested scenario**

It's applicable in a wide range of performance curves, thus, it can be used in various scenarios to speed up the tuning progress. [Detailed Description](./MedianstopAssessor.md)

**classArgs requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', assessor will **stop** the trial with smaller expectation. If 'minimize', assessor will **stop** the trial with larger expectation.
* **start_step** (*int, optional, default = 0*) - A trial is determined to be stopped or not only after receiving start_step number of reported intermediate results.

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

### Curve Fitting Assessor

> Builtin Assessor Name: **Curvefitting**

**Suggested scenario**

It's applicable in a wide range of performance curves, thus, it can be used in various scenarios to speed up the tuning progress. Even better, it's able to handle and assess curves with similar performance. [Detailed Description](./CurvefittingAssessor.md)

**Note**, according to the original paper, only incremental functions are supported. Therefore this assessor can only be used to maximize optimization metrics. For example, it can be used for accuracy, but not for loss.


**classArgs requirements:**

* **epoch_num** (*int, **required***) - The total number of epochs. We need to know the number of epochs to determine which points we need to predict.
* **start_step** (*int, optional, default = 6*) - A trial is determined to be stopped or not only after receiving start_step number of reported intermediate results.
* **threshold** (*float, optional, default = 0.95*) - The threshold that we use to decide to early stop the worst performance curve. For example: if threshold = 0.95, and the best performance in the history is 0.9, then we will stop the trial who's predicted value is lower than 0.95 * 0.9 = 0.855.
* **gap** (*int, optional, default = 1*) - The gap interval between Assessor judgements. For example: if gap = 2, start_step = 6, then we will assess the result when we get 6, 8, 10, 12...intermediate results.

**Usage example:**

```yaml
# config.yml
assessor:
    builtinAssessorName: Curvefitting
    classArgs:
      epoch_num: 20
      start_step: 6
      threshold: 0.95
      gap: 1
```
