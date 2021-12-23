Pruning Scheduler
=================

Pruning scheduler is new feature supported in pruning v2. It can bring more flexibility for pruning the model iteratively.
All the built-in iterative pruners (e.g., AGPPruner, SimulatedAnnealingPruner) are based on three abstracted components: pruning scheduler, pruners and task generators.
In addition to using the NNI built-in iterative pruners,
users can directly use the pruning schedulers to customize their own iterative pruning logic.

Workflow of Pruning Scheduler
-----------------------------

In iterative pruning, the final goal will be broken down into different small goals, and complete a small goal in each iteration.
For example, each iteration increases a little sparsity ratio, and after several pruning iterations, the continuous pruned model reaches the final overall sparsity;
fix the overall sparsity, try different ways to allocate sparsity between layers in each iteration, and find the best allocation way.

We define a small goal as ``Task``, it usually includes states inherited from previous iterations (eg. pruned model and masks) and description of the current goal (eg. a config list that describes how to allocate sparsity).
Details about ``Task`` can be found in this :githublink:`file <nni/algorithms/compression/v2/pytorch/base/scheduler.py>`.

Pruning scheduler handles two main components, a basic pruner, and a task generator. The logic of generating ``Task`` is encapsulated in the task generator.
In an iteration (one pruning step), pruning scheduler parses the ``Task`` getting from the task generator,
and reset the pruner by ``model``, ``masks``, ``config_list`` parsing from the ``Task``.
Then pruning scheduler generates the new masks by the pruner. During an iteration, the new masked model may also experience speed-up, finetuning, and evaluating.
After one iteration is done, the pruning scheduler collects the compact model, new masks and evaluation score, packages them into ``TaskResult``, and passes it to task generator.
The iteration process will end until the task generator has no more ``Task``.

How to Customized Iterative Pruning
-----------------------------------

Using AGP Pruning as an example to explain how to implement an iterative pruning by scheduler in NNI.

.. code-block:: python

    from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner, PruningScheduler
    from nni.algorithms.compression.v2.pytorch.pruning.tools import AGPTaskGenerator

    pruner = L1NormPruner(model=None, config_list=None, mode='dependency_aware', dummy_input=torch.rand(10, 3, 224, 224).to(device))
    task_generator = AGPTaskGenerator(total_iteration=10, origin_model=model, origin_config_list=config_list, log_dir='.', keep_intermediate_result=True)
    scheduler = PruningScheduler(pruner, task_generator, finetuner=finetuner, speed_up=True, dummy_input=dummy_input, evaluator=None, reset_weight=False)

    scheduler.compress()
    _, model, masks, _, _ = scheduler.get_best_result()

The full script can be found :githublink:`here <examples/model_compress/pruning/v2/scheduler_torch.py>`.

In this example, we use ``dependency_aware`` mode L1 Norm Pruner as a basic pruner during each iteration.
Note we do not need to pass ``model`` and ``config_list`` to the pruner, because in each iteration the ``model`` and ``config_list`` used by the pruner are received from the task generator.
Then we can use ``scheduler`` as an iterative pruner directly. In fact, this is the implementation of ``AGPPruner`` in NNI.

More about Task Generator
-------------------------

The task generator is used to give the model that needs to be pruned in each iteration and the corresponding config_list.
For example, ``AGPTaskGenerator`` will give the model pruned in the previous iteration and compute the sparsity using in the current iteration.
``TaskGenerator`` put all these pruning information into ``Task`` and pruning scheduler will get the ``Task``, then run it.
The pruning result will return to the ``TaskGenerator`` at the end of each iteration and ``TaskGenerator`` will judge whether and how to generate the next ``Task``.

The information included in the ``Task`` and ``TaskResult`` can be found :githublink:`here <nni/algorithms/compression/v2/pytorch/base/scheduler.py>`.

A clearer iterative pruning flow chart can be found `here <v2_pruning.rst>`__.

If you want to implement your own task generator, please following the ``TaskGenerator`` :githublink:`interface <nni/algorithms/compression/v2/pytorch/pruning/tools/base.py>`.
Two main functions should be implemented, ``init_pending_tasks(self) -> List[Task]`` and ``generate_tasks(self, task_result: TaskResult) -> List[Task]``.

Why Use Pruning Scheduler
-------------------------

One of the benefits of using a scheduler to do iterative pruning is users can use more functions of NNI pruning components,
because of simplicity of the interface and the restoration of the paper, NNI not fully exposing all the low-level interfaces to the upper layer.
For example, resetting weight value to the original model in each iteration is a key point in lottery ticket pruning algorithm, and this is implemented in ``LotteryTicketPruner``.
To reduce the complexity of the interface, we only support this function in ``LotteryTicketPruner``, not other pruners.
If users want to reset weight during each iteration in AGP pruning, ``AGPPruner`` can not do this, but users can easily set ``reset_weight=True`` in ``PruningScheduler`` to implement this.

What's more, for a customized pruner or task generator, using scheduler can easily enhance the algorithm.
In addition, users can also customize the scheduling process to implement their own scheduler.
