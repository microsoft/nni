Automatically tuning SVD (NNI in Recommenders)
==============================================

In this tutorial, we first introduce a github repo `Recommenders <https://github.com/Microsoft/Recommenders>`__. It is a repository that provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. It has various models that are popular and widely deployed in recommendation systems. To provide a complete end-to-end experience, they present each example in five key tasks, as shown below:


* `Prepare Data <https://github.com/microsoft/recommenders/tree/master/examples/01_prepare_data>`__\ : Preparing and loading data for each recommender algorithm.
* Model(`collaborative filtering algorithms <https://github.com/microsoft/recommenders/tree/master/examples/02_model_collaborative_filtering>`__\ , `content-based filtering algorithms <https://github.com/microsoft/recommenders/tree/master/examples/02_model_content_based_filtering>`__\ , `hybrid algorithms <https://github.com/microsoft/recommenders/tree/master/examples/02_model_hybrid>`__\ ): Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares (\ `ALS <https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS>`__\ ) or eXtreme Deep Factorization Machines (\ `xDeepFM <https://arxiv.org/abs/1803.05170>`__\ ).
* `Evaluate <https://github.com/microsoft/recommenders/tree/master/examples/03_evaluate>`__\ : Evaluating algorithms with offline metrics.
* `Model Select and Optimize <https://github.com/microsoft/recommenders/tree/master/examples/04_model_select_and_optimize>`__\ : Tuning and optimizing hyperparameters for recommender models.
* `Operationalize <https://github.com/microsoft/recommenders/tree/master/examples/05_operationalize>`__\ : Operationalizing models in a production environment on Azure.

The fourth task is tuning and optimizing the model's hyperparameters, this is where NNI could help. To give a concrete example that NNI tunes the models in Recommenders, let's demonstrate with the model `SVD <https://github.com/microsoft/recommenders/blob/master/examples/02_model_collaborative_filtering/surprise_svd_deep_dive.ipynb>`__\ , and data Movielens100k. There are more than 10 hyperparameters to be tuned in this model.

This `Jupyter notebook <https://github.com/microsoft/recommenders/blob/master/examples/04_model_select_and_optimize/nni_surprise_svd.ipynb>`__ provided by Recommenders is a very detailed step-by-step tutorial for this example. It uses different built-in tuning algorithms in NNI, including ``Annealing``\ , ``SMAC``\ , ``Random Search``\ , ``TPE``\ , ``Hyperband``\ , ``Metis`` and ``Evolution``. Finally, the results of different tuning algorithms are compared. Please go through this notebook to learn how to use NNI to tune SVD model, then you could further use NNI to tune other models in Recommenders.
