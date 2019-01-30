# How to use Tuner that NNI support?

For now, NNI could support tuner algorithm as following:

 - TPE
 - Random Search
 - Anneal
 - Naive Evolution
 - ENAS (on going)


 **1. Tuner algorithm introduction**


We will introduce some basic knowledge about tuner algorithm here. If you are an expert, you could skip this part and jump to how to use.

*1.1 TPE*

The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. 
    
The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evalate matric. P(x|y) is modeled by transforming the generative process of hyperparameters, replacing the distributions of the configuration prior with non-parametric densities. This optimization approach is described in detail in [Algorithms for Hyper-Parameter Optimization][1].
    
Comparing with other algorithm, TPE could be achieve better result when the number of trial experiment is small. Also TPE support continuous or discrete hyper-parameters. From a large amount of experiments, we could found that TPE is far better than Random Search.

*1.2 Random Search*

In [Random Search for Hyper-Parameter Optimization][2] show that Random Search might be surprsingly simple and effective. We suggests that we could use Random Search as basline when we have no knowledge about the prior distribution of hyper-parameters.
    
*1.3 Anneal*
    
*1.4 Naive Evolution*
Naive Evolution comes from [Large-Scale Evolution of Image Classifiers][3]. Naive Evolution requir more experiments to works, but it's very simple and easily to expand new features. There are some tips for user: 

1) large initial population could avoid to fall into local optimum
2) use some strategies to keep the deversity of population could be better.


 **2. How to use the tuner algorithm in NNI?**

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