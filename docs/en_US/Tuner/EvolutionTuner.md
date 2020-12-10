Naive Evolution Tuners on NNI
===

## Naive Evolution

Naive Evolution comes from [Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf). It randomly initializes a population based on the search space. For each generation, it chooses better ones and does some mutation (e.g., changes a hyperparameter, adds/removes one layer, etc.) on them to get the next generation. Naive Evolution requires many trials to works but it's very simple and it's easily expanded with new features.
