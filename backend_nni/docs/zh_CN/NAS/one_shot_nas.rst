One-Shot NAS 算法
=======================

One-Shot NAS 算法利用了搜索空间中模型间的权重共享来训练超网络，并使用超网络来指导选择出更好的模型。 与从头训练每个模型（我们称之为 "经典 NAS"）算法相比，此类算法大大减少了使用的计算资源。 NNI 支持下列流行的 One-Shot NAS 算法。


..  toctree::
    :maxdepth: 1

    快速入门 <NasGuide>
    ENAS <ENAS>
    DARTS <DARTS>
    P-DARTS <PDARTS>
    SPOS <SPOS>
    CDARTS <CDARTS>
    ProxylessNAS <Proxylessnas>
    TextNAS <TextNAS>