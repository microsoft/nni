##########################
神经网络架构搜索
##########################

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。
最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动调整的模型。
代表工作有 NASNet, ENAS, DARTS, Network Morphism, 以及 Evolution 等。 此外，新的创新不断涌现。

但是，要实现 NAS 算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。
为了促进 NAS 创新 (如, 设计实现新的 NAS 模型，比较不同的 NAS 模型)，
易于使用且灵活的编程接口非常重要。

Thus, we design `Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__. It is a deep learning framework that supports the exploratory training on a neural network model space, rather than on a single neural network model.
Exploratory training with Retiarii allows user to express various search spaces for *Neural Architecture Search* and *Hyper-Parameter Tuning* with high flexibility.

Some frequently used terminologies in this document:

* *Model search space*: it means a set of models from which the best model is explored/searched. Sometimes we use *search space* or *model space* in short.
* *Exploration strategy*: the algorithm that is used to explore a model search space.
* *Model evaluator*: it is used to train a model and evaluate the model's performance.

Follow the instructions below to start your journey with Retiarii.

..  toctree::
    :maxdepth: 2

    概述 <NAS/Overview>
    自定义 NAS 算法 <NAS/Advanced>
    编写搜索空间 <NAS/WriteSearchSpace>
    NAS 可视化 <NAS/Visualization>
    One-Shot NAS <NAS/one_shot_nas>
    NAS 基准测试 <NAS/Benchmarks>
    API 参考 <NAS/NasReference>
