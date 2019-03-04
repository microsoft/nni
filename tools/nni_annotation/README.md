# NNI Annotation 


## Overview

To improve user experience and reduce user effort, we design an annotation grammar. Using NNI annotation, users can adapt their code to NNI just by adding some standalone annotating strings, which does not affect the execution of the original code. 

Below is an example:

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1
```
The meaning of this example is that NNI will choose one of several values (0.1, 0.01, 0.001) to assign to the learning_rate variable. Specifically, this first line is an NNI annotation, which is a single string. Following is an assignment statement. What nni does here is to replace the right value of this assignment statement according to the information provided by the annotation line.


In this way, users could either run the python code directly or launch NNI to tune hyper-parameter in this code, without changing any codes.

## Types of Annotation:

In NNI, there are mainly four types of annotation:


### 1. Annotate variables

   `'''@nni.variable(sampling_algo, name)'''`

`@nni.variable` is used in NNI to annotate a variable.

**Arguments**

- **sampling_algo**: Sampling algorithm that specifies a search space. User should replace it with a built-in NNI sampling function whose name consists of an `nni.` identification and a search space type specified in [SearchSpaceSpec](SearchSpaceSpec.md) such as `choice` or `uniform`. 
- **name**: The name of the variable that the selected value will be assigned to. Note that this argument should be the same as the left value of the following assignment statement.

An example here is:

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1
```

### 2. Annotate functions

   `'''@nni.function_choice(*functions, name)'''`

`@nni.function_choice` is used to choose one from several functions.

**Arguments**

- **functions**: Several functions that are waiting to be selected from. Note that it should be a complete function call with arguments. Such as `max_pool(hidden_layer, pool_size)`.
- **name**: The name of the function that will be replaced in the following assignment statement.

An example here is:

```python
"""@nni.function_choice(max_pool(hidden_layer, pool_size), avg_pool(hidden_layer, pool_size), name=max_pool)"""
h_pooling = max_pool(hidden_layer, pool_size)
```

### 3. Annotate intermediate result

   `'''@nni.report_intermediate_result(metrics)'''`

`@nni.report_intermediate_result` is used to report intermediate result, whose usage is the same as `nni.report_intermediate_result` in [Trials.md](Trials.md)

### 4. Annotate final result

   `'''@nni.report_final_result(metrics)'''`

`@nni.report_final_result` is used to report the final result of the current trial, whose usage is the same as `nni.report_final_result` in [Trials.md](Trials.md)

In this way, they can easily implement automatic tuning on NNI.

For `@nni.variable`, `nni.choice` is the type of search space and there are 10 types to express your search space as follows:

## Types

All types of sampling strategies and their parameter are listed here:

* {"_type":"choice","_value":options}
  * Which means the variable value is one of the options, which should be a list. The elements of options can themselves be [nested] stochastic expressions. In this case, the stochastic choices that only appear in some of the options become conditional parameters.

* {"_type":"randint","_value":[upper]}
  * Which means the variable value is a random integer in the range [0, upper). The semantics of this distribution is that there is no more correlation in the loss function between nearby integer values, as compared with more distant integer values. This is an appropriate distribution for describing random seeds for example. If the loss function is probably more correlated for nearby integer values, then you should probably use one of the "quantized" continuous distributions, such as either quniform, qloguniform, qnormal or qlognormal. Note that if you want to change lower bound, you can use `quniform` for now.

* {"_type":"uniform","_value":[low, high]}
  * Which means the variable value is a value uniformly between low and high.
  * When optimizing, this variable is constrained to a two-sided interval.

* {"_type":"quniform","_value":[low, high, q]}
  * Which means the variable value is a value like round(uniform(low, high) / q) * q
  * Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below. If you want to uniformly choose integer from a range [low, high], you can write `_value` like this: `[low, high, 1]`.

* {"_type":"loguniform","_value":[low, high]}
  * Which means the variable value is a value drawn from a range [low, high] according to a loguniform distribution like exp(uniform(log(low), log(high))), so that the logarithm of the return value is uniformly distributed.
  * When optimizing, this variable is constrained to be positive.

* {"_type":"qloguniform","_value":[low, high, q]}
  * Which means the variable value is a value like round(loguniform(low, high)) / q) * q
  * Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.

* {"_type":"normal","_value":[mu, sigma]}
  * Which means the variable value is a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.

* {"_type":"qnormal","_value":[mu, sigma, q]}
  * Which means the variable value is a value like round(normal(mu, sigma) / q) * q
  * Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.

* {"_type":"lognormal","_value":[mu, sigma]}
  * Which means the variable value is a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed. When optimizing, this variable is constrained to be positive.

* {"_type":"qlognormal","_value":[mu, sigma, q]}
  * Which means the variable value is a value like round(exp(normal(mu, sigma)) / q) * q
  * Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.

## Search Space Types Supported by Each Tuner

|                   | choice  | randint | uniform | quniform | loguniform | qloguniform | normal  | qnormal | lognormal | qlognormal |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| TPE Tuner         | &#10003; | &#10003; | &#10003; | &#10003;  | &#10003;    | &#10003;     | &#10003; | &#10003; | &#10003;   | &#10003;    |
| Random Search Tuner| &#10003; | &#10003; | &#10003; | &#10003;  | &#10003;    | &#10003;     | &#10003; | &#10003; | &#10003;   | &#10003;    |
| Anneal Tuner   | &#10003; | &#10003; | &#10003; | &#10003;  | &#10003;    | &#10003;     | &#10003; | &#10003; | &#10003;   | &#10003;    |
| Evolution Tuner   | &#10003; | &#10003; | &#10003; | &#10003;  | &#10003;    | &#10003;     | &#10003; | &#10003; | &#10003;   | &#10003;    |
| SMAC Tuner        | &#10003; | &#10003; | &#10003; | &#10003;  | &#10003;    |      |  |  |    |     |
| Batch Tuner       | &#10003; |  |  |   |     |      |  |  |    |     |
| Grid Search Tuner | &#10003; |  |  | &#10003;  |     | &#10003;     |  |  |    |     |
| Hyperband Advisor | &#10003; | &#10003; | &#10003; | &#10003;  | &#10003;    | &#10003;     | &#10003; | &#10003; | &#10003;   | &#10003;    |
| Metis Tuner   | &#10003; | &#10003; | &#10003; | &#10003;  |     |      |  |  |    |     |

Note that In Grid Search Tuner, for users' convenience, the definition of `quniform` and `qloguniform` change, where q here specifies the number of values that will be sampled. Details about them are listed as follows

* Type 'quniform' will receive three values [low, high, q], where [low, high] specifies a range and 'q' specifies the number of values that will be sampled evenly. Note that q should be at least 2. It will be sampled in a way that the first sampled value is 'low', and each of the following values is (high-low)/q larger that the value in front of it.
* Type 'qloguniform' behaves like 'quniform' except that it will first change the range to [log(low), log(high)] and sample and then change the sampled value back.

Note that Metis Tuner only support numerical `choice` now
