Execution Engine and Model Format
=================================

After a model space and strategy has been prepared, NNI will be responsible for spawning trials and exploring the model space. Specifically, NNI will first convert the model space into a special **model format** so that it becomes easier to mutate / explore. Then the mutated models will be sent to **execution engines** for running.

We list the model formats currently supported.

* **Raw format**: It's default for one-shot strategy. It directly operates on the raw model space. This should be used when the model space is attached to a set of weights and the weights are intended to be kept / inherited.
* **Simplified format**: It's default for multi-trial strategy. It converts the model space to a dict of mutables, so that strategy will not need to mutate the dict when creating new models. This will be extremely memory efficient, but all weights and states on the model will be lost when instanting the new model.
* **Graph format**: Converting the model to a graph representation (called graph IR) using `TorchScript <https://pytorch.org/docs/stable/jit.html>`__. Each mutation on the model space will be reflected as an node/edge operation on the graph. See :doc:`mutation primitives <construct_space>` and :doc:`mutators <mutator>` for details.

NNI supports two execution engines, along with one execution engine middleware:

* **Training service execution engine** is the default engine for multi-trial strategy. It will spawn the trials concurrently and the trials will run by :doc:`NNI training service </experiment/training_service/overview>`.
* **Sequential execution engine** is the default engine for one-shot strategy. It will run the trials in the current process. The trials will run sequentially without parallelism. It's also good for debugging multi-trial strategies.
* **Cross-graph optimization middleware** experimentally supports cross-model optimizations, which makes model space exploration faster. It is only compatible with graph format above.

Execution engine and model format are configurable via :class:`~nni.nas.experiment.NasExperimentConfig`. If not configured, it will be automatically inferred based on the settings of your model space and choice of exploration strategy. In most cases, the default setting should work well.

.. tip::

   The legacy graph-based execution engine is identical to training service engine + graph format now.

Advanced Usages
---------------

Graph-based Model Format
""""""""""""""""""""""""

Graph model format converts user-defined model to a graph representation (called graph IR) using `TorchScript <https://pytorch.org/docs/stable/jit.html>`__, each instantiated module in the model is converted to a subgraph. Then mutations are applied to the graph to generate new graphs. Each new graph is then converted back to PyTorch code and executed on the user specified training service.

Users may find ``_nni_basic_unit`` of :class:`~nni.nas.nn.pytorch.ParameterizedModule` helpful in some cases. ``_nni_basic_unit`` here means the module will not be converted to a subgraph, instead, it is converted to a single graph node as a basic unit. When a module cannot be successfully parsed to a subgraph, please inherit :class:`~nni.nas.nn.pytorch.ParameterizedModule`, which will automatically enable ``_nni_basic_unit``. The parse failure could be due to complex control flow. Currently Retiarii does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as serializable module. For example, the following ``MyModule`` should be a basic unit.

  .. code-block:: python

    class MyModule(ParameterizedModule):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* Some inline mutation APIs require their handled module to be a basic unit. For example, user-defined module that is provided to :class:`~nni.nas.nn.pytorch.LayerChoice` as a candidate op should be basic units.

For exporting top models, graph-based execution engine supports exporting source code for top models by running ``exp.export_top_models(formatter='code')``.

.. _cgo-execution-engine:

Cross Graph Optimization (experimental)
"""""""""""""""""""""""""""""""""""""""

CGO (Cross-Graph Optimization) middleware does cross-model optimizations based on the graph-based model format. With CGO, multiple models could be merged and trained together in one trial. Currently, it only supports ``DedupInputOptimizer`` that can merge graphs sharing the same dataset to only loading and pre-processing each batch of data once, which can avoid bottleneck on data loading. 

.. note :: To use CGO engine, PyTorch Lightning >= 1.6.1 is required.

To enable CGO execution engine, you need to follow these steps:

1. Use training service engine.
2. Set training service to remote training service. CGO middleware currently only supports remote training service.

.. code-block:: python
  
    exp = NasExperiment(base_model, evaluator, strategy, config=NasExperimentConfig('cgo', 'graph', 'remote'))
    # ...
    # other configurations of NasExperimentConfig

    config.max_concurrency_cgo = 3 # the maximum number of concurrent models to merge
    config.batch_waiting_time = 10  # how many seconds CGO should wait before optimizing a new batch of models

    rm_conf = RemoteMachineConfig()

    # ...
    # server configuration in rm_conf
    rm_conf.gpu_indices = [0, 1, 2, 3] # gpu_indices must be set in RemoteMachineConfig for CGO

    config.training_service.machine_list = [rm_conf]
    exp.run(config, 8099)

CGO middleware only supports pytorch-lightning trainer that inherits :class:`~nni.nas.execution.cgo.MultiModelLightningModule`.
For a trial running multiple models, the trainers inheriting :class:`~nni.nas.execution.cgo.MultiModelTrainer` can handle the multiple outputs from the merged model for training, test and validation.

Sometimes, a mutated model cannot be executed (e.g., due to shape mismatch). When a trial running multiple models contains 
a bad model, CGO will re-run each model independently in separate trials without cross-model optimizations.
