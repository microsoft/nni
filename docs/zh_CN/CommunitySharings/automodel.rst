######################
自动模型调优
######################

NNI 可以应用于各种模型调优任务。 一些最先进的模型搜索算法，如EfficientNet，可以很容易地在NNI上构建。 流行的模型，例如，推荐模型，可以使用 NNI 进行调优。 下面是一些用例，展示了如何在您的模型调优任务中使用 NNI，以及如何使用 NNI 构建您自己的流水线。

..  toctree::
    :maxdepth: 1

    SVD 自动调优 <RecommendersSvd>
    NNI 中的 EfficientNet <./TrialExample/EfficientNet>
    用于阅读理解的自动模型架构搜索<../TrialExample/SquadEvolutionExamples>
    TPE 的并行优化<ParallelizingTpeSearch>