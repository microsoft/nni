# NNI Annotation 


## Overview

For improve user experience and reduce user effort, we design a good annotation grammar.

Below is an example:

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1
```

Here, this first line is a nni annotation, which is simply a single string. Following is an assignment statement. What nni does here is to replace the right value of this assignment statement according to the information provided by the annotation line.

In the annotation line, notice that:

- `nni.varialbe` is an outer function which means the hyper-parameter is an varible.
- The first argument in this function is another nni function -- `nni.choice`, which specifies how to choose the hyper-parameter.
- The second variable means the name of the variable that this hyper-parameter will be assigned to, which should be the same as the left value of the following assignment statement.

Note that every annotation should be followed by an assignment statement.

In this way, users could either run the python code directly or launch nni to tune hyper-parameter in this code, without changing any codes.


## Types

### Types of Annotation:

In NNI, there are mainly four types of annotation:


 1. Annotate variables in code as:

    `'''@nni.variable(nni.choice(2,3,5,7),name=self.conv_size)'''`

 2. Annotate functions in code as:

    `'''@nni.function_choice(max_pool(h_conv1, self.pool_size), avg_pool(h_conv1, self.pool_size), name=max_pool)'''`

 3. Annotate intermediate result in code as:

    `'''@nni.report_intermediate_result(test_acc)'''`

 4. Annotate final result in code as:

    `'''@nni.report_final_result(test_acc)'''`


### Types of Search Space

For `@nni.variable`, **`nni.choice`** is the type of search space and there are 10 types to express your search space as follows:

* nni.choice(option1,option2,...,optionN)
   * Which means the variable value is one of the options, which should be a list The elements of options can themselves be [nested] stochastic expressions. In this case, the stochastic choices that only appear in some of the options become conditional parameters.
<br/>

* nni.randint(upper)
   * Which means the variable value is a random integer in the range [0, upper). The semantics of this distribution is that there is no more correlation in the loss function between nearby integer values, as compared with more distant integer values. This is an appropriate distribution for describing random seeds for example. If the loss function is probably more correlated for nearby integer values, then you should probably use one of the "quantized" continuous distributions, such as either quniform, qloguniform, qnormal or qlognormal. Note that if you want to change lower bound, you can use `quniform` for now.
<br/>

* nni.uniform(low, high, q)
   * Which means the variable value is a value uniformly between low and high.
   * When optimizing, this variable is constrained to a two-sided interval.
<br/>

* nni.quniform(low, high, q)
   * Which means the variable value is a value like round(uniform(low, high) / q) * q
   * Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below. If you want to uniformly choose integer from a range [low, high], you can write `_value` like this: `[low, high, 1]`.
<br/>

* nni.loguniform(low, high)
   * Which means the variable value is a value drawn from a range [low, high] according to a loguniform distribution like exp(uniform(log(low), log(high))), so that the logarithm of the return value is uniformly distributed.
   * When optimizing, this variable is constrained to be positive.
<br/>

* nni.qloguniform(low, high, q)
   * Which means the variable value is a value like round(loguniform(low, high)) / q) * q
   * Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.
<br/>

* nni.normal(label, mu, sigma)
   * Which means the variable value is a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.
<br/>

* nni.qnormal(label, mu, sigma, q)
   * Which means the variable value is a value like round(normal(mu, sigma) / q) * q
   * Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.
<br/>

* nni.lognormal(label, mu, sigma)
   * Which means the variable value is a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed. When optimizing, this variable is constrained to be positive.
<br/>

* nni.qlognormal(label, mu, sigma, q)
   * Which means the variable value is a value like round(exp(normal(mu, sigma)) / q) * q
   * Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.
<br/>

Note that SMAC only supports a subset of the types above, including `choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`. In the current version, SMAC does not support cascaded search space (i.e., conditional variable in SMAC).

Note that GridSearch Tuner only supports a subset of the types above, including `choic`, `quniform` and `qloguniform`, where q here specifies the number of values that will be sampled. Details about the last two type as follows:

* Type 'quniform' will receive three values [low, high, q], where [low, high] specifies a range and 'q' specifies the number of values that will be sampled evenly. Note that q should be at least 2. It will be sampled in a way that the first sampled value is 'low', and each of the following values is (high-low)/q larger that the value in front of it.
* Type 'qloguniform' behaves like 'quniform' except that it will first change the range to [log(low), log(high)] and sample and then change the sampled value back.

