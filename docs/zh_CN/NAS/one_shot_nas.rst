One-shot NAS
=======================

One-Shot NAS 算法利用了搜索空间中模型间的权重共享来训练超网络，并使用超网络来指导选择出更好的模型。 This type of algorihtms greatly reduces computational resource compared to independently training each model from scratch (which we call "Multi-trial NAS"). NNI 支持下列流行的 One-Shot NAS 算法。


..  toctree::
    :maxdepth: 1

    Run One-shot NAS <OneshotTrainer>
    ENAS <ENAS>
    DARTS <DARTS>
    SPOS <SPOS>
    ProxylessNAS <Proxylessnas>
    FBNet <FBNet>
    Customize one-shot NAS <WriteOneshot>
