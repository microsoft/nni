.. role:: raw-html(raw)
   :format: html


HyperParameter Tuning with NNI Built-in Tuners
==============================================

To fit a machine/deep learning model into different tasks/problems, hyperparameters always need to be tuned. Automating the process of hyperparaeter tuning always requires a good tuning algorithm. NNI has provided state-of-the-art tuning algorithms as part of our built-in tuners and makes them easy to use. Below is the brief summary of NNI's current built-in tuners:

Note: Click the **Tuner's name** to get the Tuner's installation requirements, suggested scenario, and an example configuration. A link for a detailed description of each algorithm is located at the end of the suggested scenario for each tuner. Here is an `article <../CommunitySharings/HpoComparison.rst>`__ comparing different Tuners on several problems.

Currently, we support the following algorithms:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Tuner
     - Brief Introduction of Algorithm
   * - `TPE <#TPE>`__
     - The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. `Reference Paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
   * - `Random Search <#Random>`__
     - In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggest that we could use Random Search as the baseline when we have no knowledge about the prior distribution of hyper-parameters. `Reference Paper <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`__
   * - `Anneal <#Anneal>`__
     - This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on the random search that leverages smoothness in the response surface. The annealing rate is not adaptive.
   * - `Naïve Evolution <#Evolution>`__
     - Naïve Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population-based on search space. For each generation, it chooses better ones and does some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naïve Evolution requires many trials to work, but it's very simple and easy to expand new features. `Reference paper <https://arxiv.org/pdf/1703.01041.pdf>`__
   * - `SMAC <#SMAC>`__
     - SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by NNI is a wrapper on the SMAC3 GitHub repo. Notice, SMAC needs to be installed by ``pip install nni[SMAC]`` command. `Reference Paper, <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ `GitHub Repo <https://github.com/automl/SMAC3>`__
   * - `Batch tuner <#Batch>`__
     - Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.
   * - `Grid Search <#GridSearch>`__
     - Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file. Note that the only acceptable types of search space are choice, quniform, randint.
   * - `Hyperband <#Hyperband>`__
     - Hyperband tries to use limited resources to explore as many configurations as possible and returns the most promising ones as a final result. The basic idea is to generate many configurations and run them for a small number of trials. The half least-promising configurations are thrown out, the remaining are further trained along with a selection of new configurations. The size of these populations is sensitive to resource constraints (e.g. allotted search time). `Reference Paper <https://arxiv.org/pdf/1603.06560.pdf>`__
   * - `Network Morphism <#NetworkMorphism>`__
     - Network Morphism provides functions to automatically search for deep learning architectures. It generates child networks that inherit the knowledge from their parent network which it is a morph from. This includes changes in depth, width, and skip-connections. Next, it estimates the value of a child network using historic architecture and metric pairs. Then it selects the most promising one to train. `Reference Paper <https://arxiv.org/abs/1806.10282>`__
   * - `Metis Tuner <#MetisTuner>`__
     - Metis offers the following benefits when it comes to tuning parameters: While most tools only predict the optimal configuration, Metis gives you two outputs: (a) current prediction of optimal configuration, and (b) suggestion for the next trial. No more guesswork. While most tools assume training datasets do not have noisy data, Metis actually tells you if you need to re-sample a particular hyper-parameter. `Reference Paper <https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/>`__
   * - `BOHB <#BOHB>`__
     - BOHB is a follow-up work to Hyperband. It targets the weakness of Hyperband that new configurations are generated randomly without leveraging finished trials. For the name BOHB, HB means Hyperband, BO means Bayesian Optimization. BOHB leverages finished trials by building multiple TPE models, a proportion of new configurations are generated through these models. `Reference Paper <https://arxiv.org/abs/1807.01774>`__
   * - `GP Tuner <#GPTuner>`__
     - Gaussian Process Tuner is a sequential model-based optimization (SMBO) approach with Gaussian Process as the surrogate. `Reference Paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__\ , `Github Repo <https://github.com/fmfn/BayesianOptimization>`__
   * - `PBT Tuner <#PBTTuner>`__
     - PBT Tuner is a simple asynchronous optimization algorithm which effectively utilizes a fixed computational budget to jointly optimize a population of models and their hyperparameters to maximize performance. `Reference Paper <https://arxiv.org/abs/1711.09846v1>`__
   * - `DNGO Tuner <#DNGOTuner>`__
     - Use of neural networks as an alternative to GPs to model distributions over functions in bayesian optimization.

Usage of Built-in Tuners
------------------------

Using a built-in tuner provided by the NNI SDK requires one to declare the  **builtinTunerName** and **classArgs** in the ``config.yml`` file. In this part, we will introduce each tuner along with information about usage and suggested scenarios, classArg requirements, and an example configuration.

Note: Please follow the format when you write your ``config.yml`` file. Some built-in tuners have dependencies that need to be installed using ``pip install nni[<tuner>]``, like SMAC's dependencies can be installed using ``pip install nni[SMAC]``.

:raw-html:`<a name="TPE"></a>`

TPE
^^^

..

   Built-in Tuner Name: **TPE**

TPE, as a black-box optimization, can be used in various scenarios and shows good performance in general. Especially when you have limited computation resources and can only try a small number of trials. From a large amount of experiments, we found that TPE is far better than Random Search. `Detailed Description <./HyperoptTuner.rst>`__

:raw-html:`<br>`

:raw-html:`<a name="Random"></a>`

Random Search
^^^^^^^^^^^^^

..

   Built-in Tuner Name: **Random**

Random search is suggested when each trial does not take very long (e.g., each trial can be completed very quickly, or early stopped by the assessor), and you have enough computational resources. It's also useful if you want to uniformly explore the search space. Random Search can be considered a baseline search algorithm. `Detailed Description <./HyperoptTuner.rst>`__

:raw-html:`<br>`

:raw-html:`<a name="Anneal"></a>`

Anneal
^^^^^^

..

   Built-in Tuner Name: **Anneal**

Anneal is suggested when each trial does not take very long and you have enough computation resources (very similar to Random Search). It's also useful when the variables in the search space can be sample from some prior distribution. `Detailed Description <./HyperoptTuner.rst>`__

:raw-html:`<br>`

:raw-html:`<a name="Evolution"></a>`

Naïve Evolution
^^^^^^^^^^^^^^^

..

   Built-in Tuner Name: **Evolution**

Its computational resource requirements are relatively high. Specifically, it requires a large initial population to avoid falling into a local optimum. If your trial is short or leverages assessor, this tuner is a good choice. It is also suggested when your trial code supports weight transfer; that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training process. `Detailed Description <./EvolutionTuner.rst>`__

:raw-html:`<br>`

:raw-html:`<a name="SMAC"></a>`

SMAC
^^^^

..

   Built-in Tuner Name: **SMAC**


**Please note that SMAC doesn't support running on Windows currently**. For the specific reason, please refer to this `GitHub issue <https://github.com/automl/SMAC3/issues/483>`__.

Similar to TPE, SMAC is also a black-box tuner that can be tried in various scenarios and is suggested when computational resources are limited. It is optimized for discrete hyperparameters, thus, it's suggested when most of your hyperparameters are discrete. `Detailed Description <./SmacTuner.rst>`__

:raw-html:`<br>`

:raw-html:`<a name="Batch"></a>`

Batch Tuner
^^^^^^^^^^^

..

   Built-in Tuner Name: BatchTuner


**Suggested scenario**

If the configurations you want to try have been decided beforehand, you can list them in search space file (using ``choice``\ ) and run them using batch tuner. `Detailed Description <./BatchTuner.rst>`__

:raw-html:`<a name="GridSearch"></a>`

Grid Search
^^^^^^^^^^^

..

   Built-in Tuner Name: **Grid Search**

Note that the only acceptable types within the search space are ``choice``\ , ``quniform``\ , and ``randint``.

This is suggested when the search space is small. It's suggested when it is feasible to exhaustively sweep the whole search space. `Detailed Description <./GridsearchTuner.rst>`__

:raw-html:`<br>`

:raw-html:`<a name="Hyperband"></a>`

Hyperband
^^^^^^^^^

..

   Built-in Advisor Name: **Hyperband**

This is suggested when you have limited computational resources but have a relatively large search space. It performs well in scenarios where intermediate results can indicate good or bad final results to some extent. For example, when models that are more accurate early on in training are also more accurate later on. `Detailed Description <./HyperbandAdvisor.rst>`__

:raw-html:`<br>`

:raw-html:`<a name="NetworkMorphism"></a>`

Network Morphism
^^^^^^^^^^^^^^^^

..

   Built-in Tuner Name: **NetworkMorphism**

This is suggested when you want to apply deep learning methods to your task but you have no idea how to choose or design a network. You may modify this :githublink:`example <examples/trials/network_morphism/cifar10/cifar10_keras.py>` to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate, or optimizer. Currently, this tuner only supports the computer vision domain. `Detailed Description <./NetworkmorphismTuner.rst>`__

:raw-html:`<br>`

:raw-html:`<a name="MetisTuner"></a>`

Metis Tuner
^^^^^^^^^^^

..

   Built-in Tuner Name: **MetisTuner**

Similar to TPE and SMAC, Metis is a black-box tuner. If your system takes a long time to finish each trial, Metis is more favorable than other approaches such as random search. Furthermore, Metis provides guidance on subsequent trials. Here is an :githublink:`example <examples/trials/auto-gbdt/search_space_metis.json>` on the use of Metis. Users only need to send the final result, such as ``accuracy``\ , to the tuner by calling the NNI SDK. `Detailed Description <./MetisTuner.rst>`__

Note that the only acceptable types of search space types are ``quniform``\ , ``uniform``\ , ``randint``\ , and numerical ``choice``. Only numerical values are supported since the values will be used to evaluate the 'distance' between different points.

:raw-html:`<br>`

:raw-html:`<a name="BOHB"></a>`

BOHB Advisor
^^^^^^^^^^^^

..

   Built-in Tuner Name: **BOHB**

Similar to Hyperband, BOHB is suggested when you have limited computational resources but have a relatively large search space. It performs well in scenarios where intermediate results can indicate good or bad final results to some extent. In this case, it may converge to a better configuration than Hyperband due to its usage of Bayesian optimization. `Detailed Description <./BohbAdvisor.rst>`__

:raw-html:`<a name="GPTuner"></a>`

GP Tuner
^^^^^^^^

..

   Built-in Tuner Name: **GPTuner**


Note that the only acceptable types within the search space are ``randint``\ , ``uniform``\ , ``quniform``\ ,  ``loguniform``\ , ``qloguniform``\ , and numerical ``choice``. Only numerical values are supported since the values will be used to evaluate the 'distance' between different points.

As a strategy in a Sequential Model-based Global Optimization (SMBO) algorithm, GP Tuner uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) to solve and common tools can be employed to solve it. Therefore, GP Tuner is most adequate for situations where the function to be optimized is very expensive to evaluate. GP can be used when computational resources are limited. However, GP Tuner has a computational cost that grows at *O(N^3)* due to the requirement of inverting the Gram matrix, so it's not suitable when lots of trials are needed. `Detailed Description <./GPTuner.rst>`__

:raw-html:`<a name="PBTTuner"></a>`

PBT Tuner
^^^^^^^^^

..

   Built-in Tuner Name: **PBTTuner**


**Suggested scenario**

Population Based Training (PBT) bridges and extends parallel search methods and sequential optimization methods. It requires relatively small computation resource, by inheriting weights from currently good-performing ones to explore better ones periodically. With PBTTuner, users finally get a trained model, rather than a configuration that could reproduce the trained model by training the model from scratch. This is because model weights are inherited periodically through the whole search process. PBT can also be seen as a training approach. If you don't need to get a specific configuration, but just expect a good model, PBTTuner is a good choice. `See details <./PBTTuner.rst>`__

:raw-html:`<a name="DNGOTuner"></a>`

DNGO Tuner
^^^^^^^^^^

..

   Built-in Tuner Name: **DNGOTuner**

Applicable to large scale hyperparameter optimization. Bayesian optimization that rapidly finds competitive models on benchmark object recognition tasks using convolutional networks, and image caption generation using neural language models. `See details <./DngoTuner.rst>`__


**Reference and Feedback**
------------------------------


* To `report a bug <https://github.com/microsoft/nni/issues/new?template=bug-report.rst>`__ for this feature in GitHub;
* To `file a feature or improvement request <https://github.com/microsoft/nni/issues/new?template=enhancement.rst>`__ for this feature in GitHub;
* To know more about :githublink:`Feature Engineering with NNI <docs/en_US/FeatureEngineering/Overview.rst>`\ ;
* To know more about :githublink:`NAS with NNI <docs/en_US/NAS/Overview.rst>`\ ;
* To know more about :githublink:`Model Compression with NNI <docs/en_US/Compression/Overview.rst>`\ ;
