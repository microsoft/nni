Exploration Strategy
====================

There are two types of model space exploration approach: **Multi-trial strategy** and **One-shot strategy**. When the model space has been constructed, users can use either exploration approach to explore the model space. 

* :ref:`Mutli-trial strategy <multi-trial-nas>` trains each sampled model in the model space independently.
* :ref:`One-shot strategy <one-shot-nas>` samples the model from a super model.

Here is the list of exploration strategies that NNI has supported.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Category
     - Brief Description
   * - :class:`Random <nni.retiarii.strategy.Random>`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Randomly sample an architecture each time
   * - :class:`GridSearch <nni.retiarii.strategy.GridSearch>`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Traverse the search space and try all possibilities
   * - :class:`RegularizedEvolution <nni.retiarii.strategy.RegularizedEvolution>`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Evolution algorithm for NAS. `Reference <https://arxiv.org/abs/1802.01548>`__
   * - :class:`TPE <nni.retiarii.strategy.TPE>`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Tree-structured Parzen Estimator (TPE). `Reference <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
   * - :class:`PolicyBasedRL <nni.retiarii.strategy.PolicyBasedRL>`
     - :ref:`Multi-trial <multi-trial-nas>`
     - Policy-based reinforcement learning, based on implementation of tianshou. `Reference <https://arxiv.org/abs/1611.01578>`__
   * - :ref:`darts-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - Continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent. `Reference <https://arxiv.org/abs/1806.09055>`__
   * - :ref:`enas-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - RL controller learns to generate the best network on a super-net. `Reference <https://arxiv.org/abs/1802.03268>`__
   * - :ref:`fbnet-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - Choose the best block by using Gumbel Softmax random sampling and differentiable training. `Reference <https://arxiv.org/abs/1812.03443>`__
   * - :ref:`spos-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - Train a super-net with uniform path sampling. `Reference <https://arxiv.org/abs/1904.00420>`__
   * - :ref:`proxylessnas-strategy`
     - :ref:`One-shot <one-shot-nas>`
     - A low-memory-consuming optimized version of differentiable architecture search. `Reference <https://arxiv.org/abs/1812.00332>`__

.. _multi-trial-nas:

Multi-trial strategy
--------------------

Multi-trial NAS means each sampled model from model space is trained independently. A typical multi-trial NAS is `NASNet <https://arxiv.org/abs/1707.07012>`__. In multi-trial NAS, users need model evaluator to evaluate the performance of each sampled model, and need an exploration strategy to sample models from a defined model space. Here, users could use NNI provided model evaluators or write their own model evalutor. They can simply choose a exploration strategy. Advanced users can also customize new exploration strategy.

To use an exploration strategy, users simply instantiate an exploration strategy and pass the instantiated object to :class:`RetiariiExperiment <nni.retiarii.experiment.pytorch.RetiariiExperiment>`. Below is a simple example.

.. code-block:: python

   import nni.retiarii.strategy as strategy
   exploration_strategy = strategy.Random(dedup=True)

Rather than using :class:`strategy.Random <nni.retiarii.strategy.Random>`, users can choose one of the strategies from the table above.

.. _one-shot-nas:

One-shot strategy
-----------------

One-shot NAS algorithms leverage weight sharing among models in neural architecture search space to train a supernet, and use this supernet to guide the selection of better models. This type of algorihtms greatly reduces computational resource compared to independently training each model from scratch (which we call "Multi-trial NAS").

Currently, the usage of one-shot NAS strategy is a little different from multi-trial strategy. One-shot strategy is implemented with a special type of objects named *Trainer*. Following the common practice of one-shot NAS, *Trainer* trains the super-net and searches for the optimal architecture in a single run. For example,

.. code-block:: python

   from nni.retiarii.oneshot.pytorch import DartsTrainer

   trainer = DartsTrainer(
      model=model,
      loss=criterion,
      metrics=lambda output, target: accuracy(output, target, topk=(1,)),
      optimizer=optim,
      dataset=dataset_train,
      batch_size=32,
      log_frequency=50
   )
   trainer.fit()

One-shot strategy can be used without :class:`RetiariiExperiment <nni.retiarii.experiment.pytorch.RetiariiExperiment>`. Thus, the ``trainer.fit()`` here runs the experiment locally.

After ``trainer.fit()`` completes, we can use ``trainer.export()`` to export the searched architecture (a dict of choices) to a file.

.. code-block:: python

   final_architecture = trainer.export()
   print('Final architecture:', trainer.export())
   json.dump(trainer.export(), open('checkpoint.json', 'w'))

.. tip:: The trained super-net (neither the weights or exported JSON) can't be used directly. It's only an intermediate result used for deriving the final architecture. The exported architecture (can be retrieved with :meth:`nni.retiarii.fixed_arch`) needs to be *retrained* with a standard training recipe to get the final model.
