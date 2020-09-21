# 自动调优 SVD（在推荐系统中使用 NNI）

本教程中，会首先介绍 GitHub 存储库：[Recommenders](https://github.com/Microsoft/Recommenders)。 它使用 Jupyter Notebook 提供了构建推荐系统的一些示例和实践技巧。 其中大量的模型被广泛的应用于推荐系统中。 为了提供完整的体验，每个示例都通过以下五个关键任务中展示：

- [准备数据](https://github.com/Microsoft/Recommenders/blob/master/notebooks/01_prepare_data/README.md)：为每个 Recommender 算法准备并读取数据。
- [模型](https://github.com/Microsoft/Recommenders/blob/master/notebooks/02_model/README.md)：使用各种经典的以及深度学习推荐算法，如交替最小二乘法（[ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)）或极限深度分解机（[xDeepFM](https://arxiv.org/abs/1803.05170)）。
- [评估](https://github.com/Microsoft/Recommenders/blob/master/notebooks/03_evaluate/README.md)：使用离线指标来评估算法。
- [模型选择和优化](https://github.com/Microsoft/Recommenders/blob/master/notebooks/04_model_select_and_optimize/README.md)：为推荐算法模型调优超参。
- [运营](https://github.com/Microsoft/Recommenders/blob/master/notebooks/05_operationalize/README.md)：在 Azure 的生产环境上运行模型。

在第四项调优模型超参的任务上，NNI 可以发挥作用。 在 NNI 上调优推荐模型的具体示例，采用了 [SVD](https://github.com/Microsoft/Recommenders/blob/master/notebooks/02_model/surprise_svd_deep_dive.ipynb) 算法，以及数据集 Movielens100k。 此模型有超过 10 个超参需要调优。

由 Recommenders 提供的[ Jupyter notebook](https://github.com/Microsoft/Recommenders/blob/master/notebooks/04_model_select_and_optimize/nni_surprise_svd.ipynb) 中有非常详细的一步步的教程。 其中使用了不同的调优函数，包括 `Annealing`，`SMAC`，`Random Search`，`TPE`，`Hyperband`，`Metis` 以及 `Evolution`。 最后比较了不同调优算法的结果。 请参考此 Notebook，来学习如何使用 NNI 调优 SVD 模型，并可以继续使用 NNI 来调优 Recommenders 中的其它模型。