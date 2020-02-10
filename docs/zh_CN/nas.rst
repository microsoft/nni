##########################
Neural Architecture Search
##########################

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。
Recent research works have proved the feasibility of automatic NAS, and also found some models that could beat manually tuned models.
代表工作有 NASNet, ENAS, DARTS, Network Morphism, 以及 Evolution 等。 Moreover, new innovations keep emerging.

However, it takes great efforts to implement NAS algorithms, and it is hard to reuse code base of existing algorithms in a new one.
为了促进 NAS 创新 (如, 设计实现新的 NAS 模型，比较不同的 NAS 模型)，
易于使用且灵活的编程接口非常重要。

Therefore, we provide a unified interface for NAS,
来加速 NAS 创新，并更快的将最先进的算法用于现实世界的问题上。
详细信息，参考以下教程：

..  toctree::
    :maxdepth: 2

    概述 <NAS/Overview>
    Tutorial <NAS/NasGuide>
    ENAS <NAS/ENAS>
    DARTS <NAS/DARTS>
    P-DARTS <NAS/PDARTS>
    SPOS <NAS/SPOS>
    CDARTS <NAS/CDARTS>
    API Reference <NAS/NasReference>
