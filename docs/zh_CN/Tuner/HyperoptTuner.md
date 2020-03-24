# TPE, Random Search, Anneal Tuners

## TPE

Tree-structured Parzen Estimator (TPE) 是一种 sequential model-based optimization（SMBO，即基于序列模型优化）的方法。 SMBO 方法根据历史指标数据来按顺序构造模型，来估算超参的性能，随后基于此模型来选择新的超参。 The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evaluation matric. P(x|y) 通过变换超参的生成过程来建模，用非参数密度（non-parametric densities）代替配置的先验分布。 细节可参考 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)。 ​

### TPE 的并行优化

为了利用多个计算节点，TPE 方法是异步运行的，这样能避免浪费时间等待 Trial 评估的完成。 The original algorithm design was optimized for sequential computation. If we were to use TPE with much concurrency, its performance will be bad. We have optimized this case using the Constant Liar algorithm. For these principles of optimization, please refer to our [research blog](../CommunitySharings/ParallelizingTpeSearch.md).

### 用法

要使用 TPE，需要在 Experiment 的 YAML 配置文件进行如下改动：

```yaml
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
    parallel_optimize: True
    constant_liar_type: min
```

**classArgs requirements:**

* **optimize_mode** (*maximize or minimize, optional, default = maximize*) - If 'maximize', tuners will try to maximize metrics. If 'minimize', tuner will try to minimize metrics.
* **parallel_optimize** (*bool, optional, default = False*) - If True, TPE will use the Constant Liar algorithm to optimize parallel hyperparameter tuning. 否则，TPE 不会区分序列或并发的情况。
* **constant_liar_type** (*min or max or mean, optional, default = min*) - The type of constant liar to use, will logically be determined on the basis of the values taken by y at X. There are three possible values, min{Y}, max{Y}, and mean{Y}.

## Random Search（随机搜索）

In [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) we show that Random Search might be surprisingly effective despite its simplicity. We suggest using Random Search as a baseline when no knowledge about the prior distribution of hyper-parameters is available.

## Anneal（退火算法）

This simple annealing algorithm begins by sampling from the prior but tends over time to sample from points closer and closer to the best ones observed. 此算法是随机搜索的简单变体，利用了反应曲面的平滑性。 退火率不是自适应的。