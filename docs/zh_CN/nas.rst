##########################
神经网络架构搜索
##########################

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。
最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动调整的模型。
代表工作有 NASNet, ENAS, DARTS, Network Morphism, 以及 Evolution 等。 此外，新的创新不断涌现。

但是，要实现 NAS 算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。
为了促进 NAS 创新 (如, 设计实现新的 NAS 模型，比较不同的 NAS 模型)，
易于使用且灵活的编程接口非常重要。

因此，我们为 NAS 提供了统一的接口，
来加速 NAS 创新，并更快的将最先进的算法用于现实世界的问题上。
详细信息，参考以下教程：

..  toctree::
    :maxdepth: 2

    概述 <NAS/Overview>
    编写搜索空间 <NAS/WriteSearchSpace>
    经典 NAS <NAS/ClassicNas>
    One-Shot NAS <NAS/one_shot_nas>
    自定义 NAS 算法 <NAS/Advanced>
    NAS 可视化 <NAS/Visualization>
    NAS 基准测试 <NAS/Benchmarks>
    API 参考 <NAS/NasReference>
