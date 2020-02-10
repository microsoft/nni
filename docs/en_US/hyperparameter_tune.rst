#############################
Auto (Hyper-parameter) Tuning
#############################

Auto tuning is one of the key features provided by NNI, a main application scenario is
hyper-parameter tuning. Trial code is the one to be tuned, we provide a lot of popular
auto tuning algorithms (called Tuner), and some early stop algorithms (called Assessor).
NNI supports running trial on various training platforms, for example, on a local machine,
on several servers in a distributed manner, or on platforms such as OpenPAI, Kubernetes.

Other key features of NNI, such as model compression, feature engineering, can also be further
enhanced by auto tuning, which is described when introduing those features.

NNI has high extensibility, advanced users could customized their own Tuner, Assessor, and Training Service
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