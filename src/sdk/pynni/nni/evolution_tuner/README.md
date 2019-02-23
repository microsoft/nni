Naive Evolution Tuners on NNI
===

## Naive Evolution

Naive Evolution comes from [Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf). It randomly initializes a population based on search space. For each generation, it chooses better ones and do some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easily to expand new features.