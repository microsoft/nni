GP Tuner
========

Bayesian optimization works by constructing a posterior distribution of functions (a Gaussian Process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not.

GP Tuner is designed to minimize/maximize the number of steps required to find a combination of parameters that are close to the optimal combination. To do so, this method uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) to solve, and it's amenable to common tools. Therefore, Bayesian Optimization is suggested for situations where sampling the function to be optimized is very expensive.

Note that the only acceptable types within the search space are ``randint``, ``uniform``, ``quniform``, ``loguniform``, ``qloguniform``, and numerical ``choice``.

This optimization approach is described in Section 3 of `Algorithms for Hyper-Parameter Optimization <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__.

Usage
-----

classArgs requirements
^^^^^^^^^^^^^^^^^^^^^^

* **optimize_mode** (*'maximize' or 'minimize', optional, default = 'maximize'*) - If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
* **utility** (*'ei', 'ucb' or 'poi', optional, default = 'ei'*) - The utility function (acquisition function). 'ei', 'ucb', and 'poi' correspond to 'Expected Improvement', 'Upper Confidence Bound', and 'Probability of Improvement', respectively.
* **kappa** (*float, optional, default = 5*) - Used by the 'ucb' utility function. The bigger ``kappa`` is, the more exploratory the tuner will be.
* **xi** (*float, optional, default = 0*) - Used by the 'ei' and 'poi' utility functions. The bigger ``xi`` is, the more exploratory the tuner will be.
* **nu** (*float, optional, default = 2.5*) - Used to specify the Matern kernel. The smaller nu, the less smooth the approximated function is.
* **alpha** (*float, optional, default = 1e-6*) - Used to specify the Gaussian Process Regressor. Larger values correspond to an increased noise level in the observations.
* **cold_start_num** (*int, optional, default = 10*) - Number of random explorations to perform before the Gaussian Process. Random exploration can help by diversifying the exploration space.
* **selection_num_warm_up** (*int, optional, default = 1e5*) - Number of random points to evaluate when getting the point which maximizes the acquisition function.
* **selection_num_starting_points** (*int, optional, default = 250*) - Number of times to run L-BFGS-B from a random starting point after the warmup.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # config.yml
   tuner:
     name: GPTuner
     classArgs:
       optimize_mode: maximize
       utility: 'ei'
       kappa: 5.0
       xi: 0.0
       nu: 2.5
       alpha: 1e-6
       cold_start_num: 10
       selection_num_warm_up: 100000
       selection_num_starting_points: 250
