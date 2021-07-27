One-shot NAS
============

Before reading this tutorial, we highly recommend you to first go through the tutorial of how to `define a model space <./QuickStart.rst#define-your-model-space>`__.

Model Search with One-shot Trainer
----------------------------------

With a defined model space, users can explore the space in two ways. One is using strategy and single-arch evaluator as demonstrated `here <./QuickStart.rst#explore-the-defined-model-space>`__. The other is using one-shot trainer, which consumes much less computational resource compared to the first one. In this tutorial we focus on this one-shot approach. The principle of one-shot approach is combining all the models in a model space into one big model (usually called super-model or super-graph). It takes charge of both search, training and testing, by training and evaluating this big model.

We list the supported one-shot trainers here:

* DARTS trainer
* ENAS trainer
* ProxylessNAS trainer
* Single-path (random) trainer

See `API reference <./ApiReference.rst>`__ for detailed usages. Here, we show an example to use DARTS trainer manually.

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

After the searching is done, we can use the exported architecture to instantiate the full network for retraining. Here is an example:

.. code-block:: python

    from nni.retiarii import fixed_arch
    with fixed_arch('/path/to/checkpoint.json'):
        model = Model()
