# NNI 中的 BOHB Advisor

## 1. 介绍

BOHB is a robust and efficient hyperparameter tuning algorithm mentioned in [this reference paper](https://arxiv.org/abs/1807.01774). BO is an abbreviation for "Bayesian Optimization" and HB is an abbreviation for "Hyperband".

BOHB relies on HB (Hyperband) to determine how many configurations to evaluate with which budget, but it **replaces the random selection of configurations at the beginning of each HB iteration by a model-based search (Bayesian Optimization)**. 一旦贝叶斯优化生成的参数达到迭代所需的配置数, 就会使用这些配置开始执行标准的连续减半过程（successive halving）。 观察这些参数在不同资源配置（budget）下的表现 g(x, b)，用于在以后的迭代中用作我们贝叶斯优化模型选择参数的基准数据。

Below we divide the introduction of the BOHB process into two parts:

### HB（Hyperband）

We follow Hyperband’s way of choosing the budgets and continue to use SuccessiveHalving. For more details, you can refer to the [Hyperband in NNI](HyperbandAdvisor.md) and the [reference paper for Hyperband](https://arxiv.org/abs/1603.06560). This procedure is summarized by the pseudocode below.

![](../../img/bohb_1.png)

### BO（贝叶斯优化）

The BO part of BOHB closely resembles TPE with one major difference: we opted for a single multidimensional KDE compared to the hierarchy of one-dimensional KDEs used in TPE in order to better handle interaction effects in the input space.

Tree Parzen Estimator(TPE): uses a KDE (kernel density estimator) to model the densities.

![](../../img/bohb_2.png)

为了建模有效的核密度估计（KDE），设置了一个建立模型所需的最小观察点数（Nmin），在 Experiment 中它的默认值为 d+1（d是搜索空间的维度），其中 d 也是一个可以设置的超参数。 To build a model as early as possible, we do not wait until Nb = |Db|, where the number of observations for budget b is large enough to satisfy q · Nb ≥ Nmin. Instead, after initializing with Nmin + 2 random configurations, we choose the

![](../../img/bohb_3.png)

按照公式将观察到的点分成好与坏两类点，来分别拟合两个不同的密度分布。

Note that we also sample a constant fraction named **random fraction** of the configurations uniformly at random.

## 2. 流程

![](../../img/bohb_6.jpg)

以上这张图展示了 BOHB 的工作流程。 将每次训练的最大资源配置（max_budget）设为 9，最小资源配置设为（min_budget）1，逐次减半比例（eta）设为 3，其他的超参数为默认值。 In this case, s_max = 2, so we will continuously run the {s=2, s=1, s=0, s=2, s=1, s=0, ...} cycle. In each stage of SuccessiveHalving (the orange box), we will pick the top 1/eta configurations and run them again with more budget, repeating the SuccessiveHalving stage until the end of this iteration. At the same time, we collect the configurations, budgets and final metrics of each trial and use these to build a multidimensional KDEmodel with the key "budget". 这个多维的核密度估计（KDE）模型将用于指导下一个循环的参数选择。

The sampling procedure (using Multidimensional KDE to guide selection) is summarized by the pseudocode below.

![](../../img/bohb_4.png)

## 3. 用法

BOHB advisor requires the [ConfigSpace](https://github.com/automl/ConfigSpace) package. ConfigSpace can be installed using the following command.

```bash
nnictl package install --name=SMAC
```

要使用 BOHB，需要在 Experiment 的 YAML 配置文件进行如下改动：

```yaml
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
    random_fraction: 0.33
    bandwidth_factor: 3.0
    min_bandwidth: 0.001
```

**classArgs Requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will try to maximize metrics. If 'minimize', tuner will try to minimize metrics.
* **min_budget** (*int, optional, default = 1*) - The smallest budget to assign to a trial job, (budget can be the number of mini-batches or epochs). 该参数必须为正数。
* **max_budget** (*int, optional, default = 3*) - The largest budget to assign to a trial job, (budget can be the number of mini-batches or epochs). 该参数必须大于“min_budget”。
* **eta** (*整数, 可选项, 默认值为3*) - 在每次迭代中，执行完整的“连续减半”算法。 在这里，当一个使用相同计算资源的子集结束后，选择表现前 1/eta 好的参数，给予更高的优先级，进入下一轮比较（会获得更多计算资源）。 该参数必须大于等于 2。
* **min_points_in_model**(*整数, 可选项, 默认值为None*): 建立核密度估计（KDE）要求的最小观察到的点。 Default 'None' means dim+1; when the number of completed trials in this budget is equal to or larger than `max{dim+1, min_points_in_model}`, BOHB will start to build a KDE model of this budget then use said KDE model to guide configuration selection. Needs to be positive. (dim means the number of hyperparameters in search space)
* **top_n_percent**(*int, optional, default = 15*): percentage (between 1 and 99) of the observations which are considered good. 区分表现好的点与坏的点是为了建立树形核密度估计模型。 For example, if you have 100 observed trials and top_n_percent is 15, then the top 15% of points will be used for building the good points models "l(x)". The remaining 85% of points will be used for building the bad point models "g(x)".
* **num_samples** (*整数, 可选项, 默认值为64*): 用于优化 EI 值的采样个数（默认值为64）。 In this case, we will sample "num_samples" points and compare the result of l(x)/g(x). Then we will return the one with the maximum l(x)/g(x) value as the next configuration if the optimize_mode is `maximize`. Otherwise, we return the smallest one.
* **random_fraction**(*浮点数, 可选项, 默认值为0.33*): 使用模型的先验（通常是均匀）来随机采样的比例。
* **bandwidth_factor**(*float, optional, default = 3.0*): to encourage diversity, the points proposed to optimize EI are sampled from a 'widened' KDE where the bandwidth is multiplied by this factor. We suggest using the default value if you are not familiar with KDE.
* **min_bandwidth**(< 1>float, 可选, 默认值 = 0.001 </em>): 为了保持多样性, 即使所有好的样本对其中一个参数具有相同的值，使用最小带宽 (默认值: 1e-3) 而不是零。 We suggest using the default value if you are not familiar with KDE.

*Please note that the float type currently only supports decimal representations. You have to use 0.333 instead of 1/3 and 0.001 instead of 1e-3.*

## 4. 文件结构

The advisor has a lot of different files, functions, and classes. Here, we will only give most of those files a brief introduction:

* `bohb_advisor.py` Definition of BOHB, handles interaction with the dispatcher, including generating new trials and processing results. Also includes the implementation of the HB (Hyperband) part.
* `config_generator.py` Includes the implementation of the BO (Bayesian Optimization) part. The function *get_config* can generate new configurations based on BO; the function *new_result* will update the model with the new result.

## 5. 实验

### BOHB 在 MNIST 数据集上的表现

源码地址： [examples/trials/mnist-advisor](https://github.com/Microsoft/nni/tree/master/examples/trials/)

We chose BOHB to build a CNN on the MNIST dataset. 下面是实验结果：

![](../../img/bohb_5.png)

More experimental results can be found in the [reference paper](https://arxiv.org/abs/1807.01774). We can see that BOHB makes good use of previous results and has a balanced trade-off in exploration and exploitation.