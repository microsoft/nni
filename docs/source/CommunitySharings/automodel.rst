######################
Automatic Model Tuning
######################

NNI can be applied on various model tuning tasks. Some state-of-the-art model search algorithms, such as EfficientNet, can be easily built on NNI. Popular models, e.g., recommendation models, can be tuned with NNI. The following are some use cases to illustrate how to leverage NNI in your model tuning tasks and how to build your own pipeline with NNI.

..  toctree::
    :maxdepth: 1

    Tuning SVD automatically <RecommendersSvd>
    EfficientNet on NNI <../TrialExample/EfficientNet>
    Automatic Model Architecture Search for Reading Comprehension <../TrialExample/SquadEvolutionExamples>
    Parallelizing Optimization for TPE <ParallelizingTpeSearch>