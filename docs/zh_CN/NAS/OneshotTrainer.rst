One-shot NAS
============

在阅读本教程之前，我们强烈建议您先阅读有关如何 `定义一个模型空间 <./QuickStart.rst#id1>`__ 的教程。

使用 One-shot Trainer 进行模型搜素
--------------------------------------------------------------------

对于定义的模型空间，用户可以通过两种方式来探索。 一个是使用探索策略和单架构评估器，正如 `在这里 <./QuickStart.rst#id4>`__ 所演示的那样。 另一种是使用 one-shot trainer，与第一种相比，它消耗的计算资源要少得多。 在本教程中，我们专注于 One-Shot 方法。 One-Shot 方法的原理是将模型空间中的所有模型合并成一个大模型（通常称为超级模型或超级图）。 通过训练和评估整个超级模型，来实现搜索、训练和测试的任务。

我们在此列出已支持的 One-Shot Trainer：

* DARTS trainer
* ENAS trainer
* ProxylessNAS trainer
* Single-path (random) trainer

参见 `API 参考 <./ApiReference.rst>`__ 获得详细用法。 在这里，我们展示了一个例子来手动使用 DARTS Trainer。

.. code-block:: python

  from nni.retiarii.oneshot.pytorch import DartsTrainer
  trainer = DartsTrainer(
      model=model,
      loss=criterion,
      metrics=lambda output, target: accuracy(output, target, topk=(1,)),
      optimizer=optim,
      num_epochs=args.epochs,
      dataset=dataset_train,
      batch_size=args.batch_size,
      log_frequency=args.log_frequency,
      unrolled=args.unrolled
  )
  trainer.fit()
  final_architecture = trainer.export()

**导出架构的格式**：未来将会支持。
