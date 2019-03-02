# Search Space

## Overview

In NNI, tuner will sample parameters/architecture according to the search space, which is defined as a json file.

To define a search space, users should define the name of variable, the type of sampling strategy and its parameters.

* A example of search space definition as follow:

```yaml
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "batch_size":{"_type":"choice","_value":[50, 250, 500]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}

```

Take the first line as an example. `dropout_rate` is defined as a variable whose priori distribution is a uniform distribution of a range from `0.1` and `0.5`.

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

* {"_type":"normal","_value":[label, mu, sigma]}
  * Which means the variable value is a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.

* {"_type":"qnormal","_value":[label, mu, sigma, q]}
  * Which means the variable value is a value like round(normal(mu, sigma) / q) * q
  * Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.

* {"_type":"lognormal","_value":[label, mu, sigma]}
  * Which means the variable value is a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed. When optimizing, this variable is constrained to be positive.

* {"_type":"qlognormal","_value":[label, mu, sigma, q]}
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
