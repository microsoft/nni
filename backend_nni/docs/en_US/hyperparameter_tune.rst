#############################
Auto (Hyper-parameter) Tuning
#############################

Auto tuning is one of the key features provided by NNI; a main application scenario being
hyper-parameter tuning. Tuning specifically applies to trial code. We provide a lot of popular
auto tuning algorithms (called Tuner), and some early stop algorithms (called Assessor).
NNI supports running trials on various training platforms, for example, on a local machine,
on several servers in a distributed manner, or on platforms such as OpenPAI, Kubernetes, etc.

Other key features of NNI, such as model compression, feature engineering, can also be further
enhanced by auto tuning, which we'll described when introducing those features.

NNI has high extensibility, advanced users can customize their own Tuner, Assessor, and Training Service
according to their needs.

..  toctree::
    :maxdepth: 2

    Write Trial <TrialExample/Trials>
    Tuners <builtin_tuner>
    Assessors <builtin_assessor>
    Training Platform <training_services>
    Examples <examples>
    WebUI <Tutorial/WebUI>
    How to Debug <Tutorial/HowToDebug>
    Advanced <hpo_advanced>