Builtin-Tuners
==============

NNI provides an easy way to adopt an approach to set up parameter tuning algorithms, we call them **Tuner**.

Tuner receives metrics from `Trial` to evaluate the performance of a specific parameters/architecture configuration. Tuner sends the next hyper-parameter or architecture configuration to Trial.

The following table briefly describes the built-in tuners provided by NNI. Click the **Tuner's name** to get the Tuner's installation requirements, suggested scenario, and an example configuration. A link for a detailed description of each algorithm is located at the end of the suggested scenario for each tuner. Here is an `article <../CommunitySharings/HpoComparison.rst>`__ comparing different Tuners on several problems.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Tuner
     - Brief Introduction of Algorithm

   * - `TPE <./TpeTuner.rst>`__
     - The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. `Reference Paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__

   * - `Random Search <./RandomTuner.rst>`__
     - In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggest that we could use Random Search as the baseline when we have no knowledge about the prior distribution of hyper-parameters. `Reference Paper <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`__

   * - `Anneal <./AnnealTuner.rst>`__
     - This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on the random search that leverages smoothness in the response surface. The annealing rate is not adaptive.

   * - `Naïve Evolution <./EvolutionTuner.rst>`__
     - Naïve Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population-based on search space. For each generation, it chooses better ones and does some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naïve Evolution requires many trials to work, but it's very simple and easy to expand new features. `Reference paper <https://arxiv.org/pdf/1703.01041.pdf>`__

   * - `SMAC <./SmacTuner.rst>`__
     - SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by NNI is a wrapper on the SMAC3 GitHub repo.

       Notice, SMAC needs to be installed by ``pip install nni[SMAC]`` command. `Reference Paper, <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ `GitHub Repo <https://github.com/automl/SMAC3>`__

   * - `Batch tuner <./BatchTuner.rst>`__
     - Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.

   * - `Grid Search <./GridsearchTuner.rst>`__
     - Grid Search performs an exhaustive searching through the search space.

   * - `Hyperband <./HyperbandAdvisor.rst>`__
     - Hyperband tries to use limited resources to explore as many configurations as possible and returns the most promising ones as a final result. The basic idea is to generate many configurations and run them for a small number of trials. The half least-promising configurations are thrown out, the remaining are further trained along with a selection of new configurations. The size of these populations is sensitive to resource constraints (e.g. allotted search time). `Reference Paper <https://arxiv.org/pdf/1603.06560.pdf>`__

   * - `Metis Tuner <./MetisTuner.rst>`__
     - Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter. `Reference Paper <https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/>`__

   * - `BOHB <./BohbAdvisor.rst>`__
     - BOHB is a follow-up work to Hyperband. It targets the weakness of Hyperband that new configurations are generated randomly without leveraging finished trials. For the name BOHB, HB means Hyperband, BO means Bayesian Optimization. BOHB leverages finished trials by building multiple TPE models, a proportion of new configurations are generated through these models. `Reference Paper <https://arxiv.org/abs/1807.01774>`__

   * - `GP Tuner <./GPTuner.rst>`__
     - Gaussian Process Tuner is a sequential model-based optimization (SMBO) approach with Gaussian Process as the surrogate. `Reference Paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__, `Github Repo <https://github.com/fmfn/BayesianOptimization>`__

   * - `PBT Tuner <./PBTTuner.rst>`__
     - PBT Tuner is a simple asynchronous optimization algorithm which effectively utilizes a fixed computational budget to jointly optimize a population of models and their hyperparameters to maximize performance. `Reference Paper <https://arxiv.org/abs/1711.09846v1>`__

   * - `DNGO Tuner <./DngoTuner.rst>`__
     - Use of neural networks as an alternative to GPs to model distributions over functions in bayesian optimization.

..  toctree::
    :maxdepth: 1

    TPE <Tuner/TpeTuner>
    Random Search <Tuner/RandomTuner>
    Anneal <Tuner/AnnealTuner>
    Naive Evolution <Tuner/EvolutionTuner>
    SMAC <Tuner/SmacTuner>
    Metis Tuner <Tuner/MetisTuner>
    Batch Tuner <Tuner/BatchTuner>
    Grid Search <Tuner/GridsearchTuner>
    GP Tuner <Tuner/GPTuner>
    Network Morphism <Tuner/NetworkmorphismTuner>
    Hyperband <Tuner/HyperbandAdvisor>
    BOHB <Tuner/BohbAdvisor>
    PBT Tuner <Tuner/PBTTuner>
    DNGO Tuner <Tuner/DngoTuner>
