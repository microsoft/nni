BOHB Advisor on NNI
===

## 1. Introduction
BOHB is a robust and efficient hyperparameter tuning algorithm mentioned in [reference paper](https://arxiv.org/abs/1807.01774). BO is the abbreviation of Bayesian optimization and HB is the abbreviation of Hyperband.

BOHB relies on HB(Hyperband) to determine how many configurations to evaluate with which budget, but it **replaces the random selection of configurations at the beginning of each HB iteration by a model-based search(Byesian Optimization)**. Once the desired number of configurations for the iteration is reached, the standard successive halving procedure is carried out using these configurations. We keep track of the performance of all function evaluations g(x, b) of configurations x on all budgets b to use as a basis for our models in later iterations.

Below we divide introduction of the BOHB process into two parts:

### HB (Hyperband)

We follow Hyperband’s way of choosing the budgets and continue to use SuccessiveHalving, for more details, you can refer to the [Hyperband in NNI](hyperbandAdvisor.md) and [reference paper of Hyperband](https://arxiv.org/abs/1603.06560). This procedure is summarized by the pseudocode below.

![](../img/bohb_1.png)

### BO (Bayesian Optimization)

The BO part of BOHB closely resembles TPE, with one major difference: we opted for a single multidimensional KDE compared to the hierarchy of one-dimensional KDEs used in TPE in order to better handle interaction effects in the input space.

Tree Parzen Estimator(TPE): uses a KDE(kernel density estimator) to model the densities.

![](../img/bohb_2.png)

To fit useful KDEs, we require a minimum number of data points Nmin; this is set to d + 1 for our experiments, where d is the number of hyperparameters. To build a model as early as possible, we do not wait until Nb = |Db|, the number of observations for budget b, is large enough to satisfy q · Nb ≥ Nmin. Instead, after initializing with Nmin + 2 random configurations, we choose the

![](../img/bohb_3.png)

best and worst configurations, respectively, to model the two densities.

Note that we alse sample a constant fraction named **random fraction** of the configurations uniformly at random.

This procedure is summarized by the pseudocode below.

![](../img/bohb_4.png)

## 2. Usage

BOHB advisor requires [ConfigSpace](https://github.com/automl/ConfigSpace), user could use `pip3 install ConfigSpace==0.4.7` to install it.

To use BOHB, you should add the following spec in your experiment's YAML config file:

```yml
advisor:
  builtinAdvisorName: BOHB
  classArgs:
    optimize_mode: maximize
    min_budget: 1
    max_budget: 27
    eta: 3
    min_points_in_model: 7
    top_n_percent: 15
    num_samples: 64
    random_fraction: 1/3
    bandwidth_factor: 3
    min_bandwidth: 1e-3
```

**Requirement of classArg**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will return the hyperparameter set with larger expectation. If 'minimize', tuner will return the hyperparameter set with smaller expectation.
* **min_budget** (*int, optional, default = 1*) - The smallest budget to consider. Needs to be positive.
* **max_budget** (*int, optional, default = 3*) - The largest budget to consider. Needs to be larger than min_budget.
* **eta** (*int, optional, default = 3*) - In each iteration, a complete run of sequential halving is executed. In it, after evaluating each configuration on the same subset size, only a fraction of 1/eta of them 'advances' to the next round. Must be greater or equal to 2.
* **min_points_in_model**(*int, optional, default = None*): number of observations to start building a KDE. Default 'None' means dim+1, the bare minimum, reset value must be positive.
* **top_n_percent**(*int, optional, default = 15*): percentage (between 1 and 99, default 15) of the observations that are considered good.
* **num_samples**(*int, optional, default = 64*): number of samples to optimize EI (default 64)
* **random_fraction**(*float, optional, default = 1/3*): fraction of purely random configurations that are sampled from the prior without the model.
* **bandwidth_factor**(*float, optional, default = 3*): to encourage diversity, the points proposed to optimize EI, are sampled from a 'widened' KDE where the bandwidth is multiplied by this factor
* **min_bandwidth**(*float, optional, default = 1e-3*): to keep diversity, even when all (good) samples have the same value for one of the parameters, a minimum bandwidth (default: 1e-3) is used instead of zero.

## 3. File Structure
The advisor has a lot of different files, functions and classes. Here we will only give most of those files a brief introduction:

* `bohb_advisor.py` Defination of BOHB, handle the interaction with the dispatcher, including generating new trial and processing results. Also includes the implementation of HB(Hyperband) part.
* `config_generator.py` includes the implementation of BO(Bayesian Optimization) part. The function *get_config* can generate new configuration base on BO, the function *new_result* will update model with the new result.

## 4. Experiment

### MNIST with BOHB

code implementation: [examples/trials/mnist-advisor](https://github.com/Microsoft/nni/tree/master/examples/trials/)

We chose BOHB to build CNN on the MNIST dataset. The following is our experimental final results:

![](../img/bohb_5.png)

More experimental result can be found in the [reference paper](https://arxiv.org/abs/1807.01774), we can see that BOHB makes good use of previous results, and has a balance trade-off in exploration and exploitation.