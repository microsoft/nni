# Tuners

## Overview

NNI provides an easy to adopt approach to set up parameter tuning algorithms, we call them **Tuners**. Tuner specifies the algorithm you use to generate hyperparameter sets for each trial.

In NNI, we support two approaches to set the tuner.

1. Directly use tuner provided by nni sdk

        required fields: builtinTunerName and classArgs.

2. Customize your own tuner file

        required fields: codeDirectory, classFileName, className and classArgs.

For now, NNI has supported the following tuner algorithms:

|Tuner|Brief introduction to the algorithm|Suggested scenario|Usage|Reference Paper|
|---|---|---|---|---|
|**TPE**|The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model.|TPE, as a black-box optimization, can be used in various scenarios, and shows good performance in general. Especially when you have limited computation resource and can only try a small number of trials. From a large amount of experiments, we could found that TPE is far better than Random Search.|[TPE Usage][1]|[Algorithms for Hyper-Parameter Optimization][2]|
|**Random Search**|In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggests that we could use Random Search as baseline when we have no knowledge about the prior distribution of hyper-parameters.|Random search is suggested when each trial does not take too long (e.g., each trial can be completed very soon, or early stopped by assessor quickly), and you have enough computation resource. Or you want to uniformly explore the search space. Random Search could be considered as baseline of search algorithm.|[Random Search Usage][3]||
|**Anneal**|This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on random search that leverages smoothness in the response surface. The annealing rate is not adaptive.|Anneal is suggested when each trial does not take too long, and you have enough computation resource(almost same with Random Search). Or the variables in search space could be sample from some prior distribution.|[Anneal Usage][4]||
|**Naive Evolution**|Naive Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population based on search space. For each generation, it chooses better ones and do some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easily to expand new features.|Its requirement of computation resource is relatively high. Specifically, it requires large inital population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress.|[Naive Evolution][5]|[Large-Scale Evolution of Image Classifiers][6]

[1]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[2]: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
[3]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[4]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[5]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[6]: https://arxiv.org/pdf/1703.01041.pdf