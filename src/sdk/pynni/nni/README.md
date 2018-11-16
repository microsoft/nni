# How to use Tuner that NNI supports?

For now, NNI has supported the following tuner algorithms. Note that NNI installation only installs a subset of those algorithms, other algorithms should be installed through `nnictl package install` before you use them. For example, for SMAC the installation command is `nnictl package install --name=SMAC`.

 - TPE
 - Random Search
 - Anneal
 - Naive Evolution
 - Grid Search
 - SMAC (to install through `nnictl`)
 - ENAS (ongoing)
 - Batch (ongoing)

 ## 1. Tuner algorithm introduction


We will introduce some basic knowledge about tuner algorithm here. If you are an expert, you could skip this part and jump to how to use.

**TPE**

The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. 
    
The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evaluate matric. P(x|y) is modeled by transforming the generative process of hyperparameters, replacing the distributions of the configuration prior with non-parametric densities. This optimization approach is described in detail in [Algorithms for Hyper-Parameter Optimization][1].
    
Comparing with other algorithm, TPE could be achieve better result when the number of trial experiment is small. Also TPE support continuous or discrete hyper-parameters. From a large amount of experiments, we could found that TPE is far better than Random Search.

**Random Search**

In [Random Search for Hyper-Parameter Optimization][2] show that Random Search might be surprisingly simple and effective. We suggests that we could use Random Search as baseline when we have no knowledge about the prior distribution of hyper-parameters.
    
**Anneal**
    
**Naive Evolution**

Naive Evolution comes from [Large-Scale Evolution of Image Classifiers][3]. Naive Evolution require more experiments to works, but it's very simple and easily to expand new features. There are some tips for user: 

1) large initial population could avoid to fall into local optimum
2) use some strategies to keep the diversity of population could be better.

**SMAC**

[SMAC][4] is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by nni is a wrapper on [the SMAC3 github repo][5]. 

Note that SMAC only supports a subset of the types in [search space spec](../../../../docs/SearchSpaceSpec.md), including `choice`, `randint`, `uniform`, `loguniform`, `quniform(q=1)`.

**Batch**

Batch allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done.

**Gridsearch**

Gridsearch performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file

Note that the only acceptable types of search space are 'quniform', 'qloguniform' and 'choice':

* Type 'choice' will select one of the options. Note that it can also be nested.
* Type 'quniform' will receive three values [low, high, q], where [low, high] specifies a range and 'q' specifies the number of values that will be sampled evenly. It will be sampled in a way that the first sampled value is 'low', and each of the following values is (high-low)/q larger that the value in front of it.
* Type 'qloguniform' behaves like 'quniform' except that it will first change the range to [log10(low), log10(high)] and sample and then change the sampled value back.

 ## 2. How to use the tuner algorithm in NNI?

User only need to do one thing: choose a Tuner```config.yaml```.
Here is an example:


    ```
    # config.yaml
    tuner:
      # choice: TPE, Random, Anneal, Evolution, ...
      builtinTunerName: TPE
      classArgs:
        # choice: maximize, minimize
        optimize_mode: maximize
    ```

There are two filed you need to set: 

```builtinTunerName``` and ```optimize_mode```.

    builtinTunerName: TPE / Random / Anneal / Evolution
    optimize_mode:  maximize / minimize


  [1]: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
  [2]: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
  [3]: https://arxiv.org/pdf/1703.01041.pdf
  [4]: https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
  [5]: https://github.com/automl/SMAC3
