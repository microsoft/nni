One-shot NAS
============

One-shot NAS algorithms leverage weight sharing among models in neural architecture search space to train a supernet, and use this supernet to guide the selection of better models. This type of algorihtms greatly reduces computational resource compared to independently training each model from scratch (which we call "Multi-trial NAS"). NNI has supported many popular One-shot NAS algorithms as following.


..  toctree::
    :maxdepth: 1

    Run One-shot NAS <OneshotTrainer>
    ENAS <ENAS>
    DARTS <DARTS>
    SPOS <SPOS>
    ProxylessNAS <Proxylessnas>
    FBNet <FBNet>
    Customize One-shot NAS <WriteOneshot>
