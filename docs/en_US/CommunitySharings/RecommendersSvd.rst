
Automatically tuning SVD (NNI in Recommenders)
==============================================

In this tutorial, we first introduce a github repo `Recommenders <https://github.com/Microsoft/Recommenders>`_. It is a repository that provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. It has various models that are popular and widely deployed in recommendation systems. To provide a complete end-to-end experience, they present each example in five key tasks, as shown below:


* `Prepare Data <https://github.com/Microsoft/Recommenders/blob/master/notebooks/01_prepare_data/README.md>`_\ : Preparing and loading data for each recommender algorithm.
* `Model <https://github.com/Microsoft/Recommenders/blob/master/notebooks/02_model/README.md>`_\ : Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares (\ `ALS <https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS>`_\ ) or eXtreme Deep Factorization Machines (\ `xDeepFM <https://arxiv.org/abs/1803.05170>`_\ ).
* `Evaluate <https://github.com/Microsoft/Recommenders/blob/master/notebooks/03_evaluate/README.md>`_\ : Evaluating algorithms with offline metrics.
* `Model Select and Optimize <https://github.com/Microsoft/Recommenders/blob/master/notebooks/04_model_select_and_optimize/README.md>`_\ : Tuning and optimizing hyperparameters for recommender models.
* `Operationalize <https://github.com/Microsoft/Recommenders/blob/master/notebooks/05_operationalize/README.md>`_\ : Operationalizing models in a production environment on Azure.

The fourth task is tuning and optimizing the model's hyperparameters, this is where NNI could help. To give a concrete example that NNI tunes the models in Recommenders, let's demonstrate with the model `SVD <https://github.com/Microsoft/Recommenders/blob/master/notebooks/02_model/surprise_svd_deep_dive.ipynb>`_\ , and data Movielens100k. There are more than 10 hyperparameters to be tuned in this model.

`This Jupyter notebook <https://github.com/Microsoft/Recommenders/blob/master/notebooks/04_model_select_and_optimize/nni_surprise_svd.ipynb>`_ provided by Recommenders is a very detailed step-by-step tutorial for this example. It uses different built-in tuning algorithms in NNI, including ``Annealing``\ , ``SMAC``\ , ``Random Search``\ , ``TPE``\ , ``Hyperband``\ , ``Metis`` and ``Evolution``. Finally, the results of different tuning algorithms are compared. Please go through this notebook to learn how to use NNI to tune SVD model, then you could further use NNI to tune other models in Recommenders.
