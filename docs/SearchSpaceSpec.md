## How to define search space?

### Hyper-parameter Search Space

* A search space configure example as follow:

```python
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "batch_size":{"_type":"choice","_value":[50, 250, 500]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}

```

The example define `dropout_rate` as variable which priori distribution is uniform distribution, and its value from `0.1` and `0.5`.
The tuner will sample parameters/architecture by understanding the search space first.

User should define the name of variable, type and candidate value of variable.
The candidate type and value for variable is here:

* {"_type":"choice","_value":options}
   * Which means the variable value is one of the options, which should be a list The elements of options can themselves be [nested] stochastic expressions. In this case, the stochastic choices that only appear in some of the options become conditional parameters.
<br/>

* {"_type":"randint","_value":[upper]}
   * Which means the variable value is a random integer in the range [0, upper). The semantics of this distribution is that there is no more correlation in the loss function between nearby integer values, as compared with more distant integer values. This is an appropriate distribution for describing random seeds for example. If the loss function is probably more correlated for nearby integer values, then you should probably use one of the "quantized" continuous distributions, such as either quniform, qloguniform, qnormal or qlognormal. Note that if you want to change lower bound, you can use `quniform` for now.
<br/>

* {"_type":"uniform","_value":[low, high]}
   * Which means the variable value is a value uniformly between low and high.
   * When optimizing, this variable is constrained to a two-sided interval.
<br/>

* {"_type":"quniform","_value":[low, high, q]}
   * Which means the variable value is a value like round(uniform(low, high) / q) * q
   * Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below. If you want to uniformly choose integer from a range [low, high], you can write `_value` like this: `[low, high, 1]`.
<br/>

* {"_type":"loguniform","_value":[low, high]}
   * Which means the variable value is a value drawn from a range [low, high] according to a loguniform distribution like exp(uniform(log(low), log(high))), so that the logarithm of the return value is uniformly distributed.
   * When optimizing, this variable is constrained to be positive.
<br/>

* {"_type":"qloguniform","_value":[low, high, q]}
   * Which means the variable value is a value like round(loguniform(low, high)) / q) * q
   * Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.
<br/>

* {"_type":"normal","_value":[label, mu, sigma]}
   * Which means the variable value is a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.
<br/>

* {"_type":"qnormal","_value":[label, mu, sigma, q]}
   * Which means the variable value is a value like round(normal(mu, sigma) / q) * q
   * Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.
<br/>

* {"_type":"lognormal","_value":[label, mu, sigma]}
   * Which means the variable value is a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed. When optimizing, this variable is constrained to be positive.
<br/>

* {"_type":"qlognormal","_value":[label, mu, sigma, q]}
   * Which means the variable value is a value like round(exp(normal(mu, sigma)) / q) * q
   * Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.
<br/>

Note that SMAC only supports a subset of the types above, including `choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`. In the current version, SMAC does not support cascaded search space (i.e., conditional variable in SMAC).

Note that GridSearch Tuner only supports a subset of the types above, including `choice`, `quniform` and `qloguniform`, where q here specifies the number of values that will be sampled. Details about the last two type as follows
* Type 'quniform' will receive three values [low, high, q], where [low, high] specifies a range and 'q' specifies the number of values that will be sampled evenly. Note that q should be at least 2. It will be sampled in a way that the first sampled value is 'low', and each of the following values is (high-low)/q larger that the value in front of it.
* Type 'qloguniform' behaves like 'quniform' except that it will first change the range to [log(low), log(high)] and sample and then change the sampled value back.
