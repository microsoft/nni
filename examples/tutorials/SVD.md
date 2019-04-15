# Overview

[Recommenders](https://github.com/Microsoft/Recommenders) is a repository that provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail its learnings on five key tasks:

- [Prepare Data](https://github.com/Microsoft/Recommenders/blob/master/notebooks/01_prepare_data/README.md): Preparing and loading data for each recommender algorithm
- [Model](https://github.com/Microsoft/Recommenders/blob/master/notebooks/02_model/README.md): Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) or eXtreme Deep Factorization Machines ([xDeepFM](https://arxiv.org/abs/1803.05170)).
- [Evaluate](https://github.com/Microsoft/Recommenders/blob/master/notebooks/03_evaluate/README.md): Evaluating algorithms with offline metrics
- [Model Select and Optimize](https://github.com/Microsoft/Recommenders/blob/master/notebooks/04_model_select_and_optimize/README.md): Tuning and optimizing hyperparameters for recommender models
- [Operationalize](https://github.com/Microsoft/Recommenders/blob/master/notebooks/05_operationalize/README.md): Operationalizing models in a production environment on Azure

# NNI in Recommenders

In the fourth tasks mentioned above, NNI is utilized for hyperparameter tuning of the matrix factorization method SVD from [Surprise library](https://surprise.readthedocs.io/en/stable/) on Movielens100k. In this notebook, different NNI tuning algorithms are used, including `Annealing`, `SMAC`, `Random Search`, `TPE`, `Hyperband`, `Metis` and `Evolution`. Finally, the results of different tuning algorithms are compared.

This is demonstrated by a [jupyter notebook](https://github.com/Microsoft/Recommenders), which menas that it also shows how to use NNI and communicate with NNI restful server in a jupyter notebook
