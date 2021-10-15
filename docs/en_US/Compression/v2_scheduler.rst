Pruning Scheduler
=================

Pruning scheduler is new feature supported in pruning v2. It can bring more flexibility for pruning the model iteratively.
All the iterative pruners are based on pruning scheduler, pruners and task generators. In addition to using the NNI official iterative pruner implementations,
users can directly using pruning scheduler to implement their own iterative pruning logic.

What is Task Generator
----------------------

The task generator is used to give the model that needs to be pruned in each iteration and the corresponding config_list.
For example, ``AGPTaskGenerator`` will give the model pruned in the previous iteration and compute the sparsity using in the current iteration.
``TaskGenerator`` put all these pruning information into ``Task`` and pruning scheduler will call ``TaskGenerator.next()`` to get the next task, then run the task.
The pruning result during each iteration will return to the ``TaskGenerator`` by ``TaskGenerator.receive_task_result(TaskResult)``.

The information included in the ``Task`` and ``TaskResult`` can be found :githublink:`here <nni/algorithms/compression/v2/pytorch/base/scheduler.py>`.

A clearer iterative pruning flow chart can be found `here <v2_pruning.rst>`__.

If you want to implement your own task generator, please following the ``TaskGenerator`` :githublink:`interface <nni/algorithms/compression/v2/pytorch/pruning/tools/base.py>`.
Two main functions should be implemented, ``init_pending_tasks(self) -> List[Task]`` and ``generate_tasks(self, task_result: TaskResult) -> List[Task]``.

How to Customized Iterative Pruning
-----------------------------------

Using AGP Pruning as an example to explain how to implement an iterative pruning by scheduler in NNI.

.. code-block:: python

    from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner, PruningScheduler
    from nni.algorithms.compression.v2.pytorch.pruning.tools import AGPTaskGenerator

    pruner = L1NormPruner(model=None, config_list=None, mode='dependency_aware', dummy_input=torch.rand(10, 3, 224, 224).to(device))
    task_generator = AGPTaskGenerator(total_iteration=10, origin_model=model, origin_config_list=config_list, log_dir='.', keep_intermediate_result=True)
    scheduler = PruningScheduler(pruner, task_generator, finetuner=finetuner, speed_up=True, dummy_input=dummy_input, evaluator=None, reset_weight=True)

    scheduler.compress()
    _, model, masks, _, _ = scheduler.get_best_result()

The full script can be found :githublink:`here <examples/model_compress/pruning/v2/scheduler_torch.py>`.

In this example, we use ``dependency_aware`` mode L1 Norm Pruner as a basic pruner during each iteration.
Note we do not need to pass ``model`` and ``config_list`` to the pruner, because in each iteration the ``model`` and ``config_list`` used by the pruner are received from the task generator.
Then we can use ``scheduler`` as an iterative pruner directly. In fact, this is the implementation of ``AGPPruner`` in NNI.

The benefit of using scheduler to do iterative pruning is users can use more functions of NNI pruning components.
For example, resetting weight value to the original model in each iteration is a key point in lottery ticket pruning algorithm, and this is implemented in ``LotteryTicketPruner``.
To reduce the complexity of the interface, we only support this function in ``LotteryTicketPruner``, not other pruners.
If users want to reset weight during each iteration in AGP pruning, ``AGPPruner`` can not do this, but users can easily set ``reset_weight=True`` in ``PruningScheduler`` to implement this.
What's more, for a customized pruner or task generator, using scheduler can easily enhance the algorithm.
In addition, users can also customize the scheduling process to implement their own scheduler.
