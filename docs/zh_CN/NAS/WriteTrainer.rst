自定义模型 Evaluator
===============================

模型评估器（Evaluator）对于评估新探索的模型的性能是必要的。 一个模型评估器通常包括训练、验证和测试一个单一的模型。 我们为用户提供了两种方法来编写新的模型评估器，下面将分别演示。

使用 FunctionalEvaluator
------------------------

定制一个新的评估器的最简单的方法是使用功能性的 API，当训练代码已经可用时，这就非常容易。 用户只需要编写一个 fit 函数来包装所有内容。 此函数接收一个位置参数（``model_cls``）和可能的关键字参数。 关键字参数（除 ``model_cls`` 外）作为 FunctionEvaluator 的初始化参数被输入。 通过这种方式，用户可以控制一切，但向框架公开的信息较少，因此进行优化的机会也较少。 示例如下。

.. code-block:: python

    from nni.retiarii.evaluator import FunctionalEvaluator
    from nni.retiarii.experiment.pytorch import RetiariiExperiment

    def fit(model_cls, dataloader):
        model = model_cls()
        train(model, dataloader)
        acc = test(model, dataloader)
        nni.report_final_result(acc)

    evaluator = FunctionalEvaluator(fit, dataloader=DataLoader(foo, bar))
    experiment = RetiariiExperiment(base_model, evaluator, mutators, strategy)

.. note:: 由于我们目前的实施限制，``fit`` 函数应该放在另一个 python 文件中，而不是放在主文件中。 这个限制将在未来的版本中得到修复。

使用 PyTorch-Lightning
----------------------

NNI 建议以 PyTorch-Lightning 风格编写训练代码，即编写一个 LightningModule，定义训练所需的所有元素（例如 loss function、optimizer），并定义一个 Trainer，使用 dataloader 来执行训练（可选）。 在此之前，请阅读 `PyTorch-lightning 文档 <https://pytorch-lightning.readthedocs.io/>` 了解 PyTorch-lightning 的基本概念和组件。 在此之前，请阅读 `PyTorch-lightning 文档 <https://pytorch-lightning.readthedocs.io/>`__ 了解 PyTorch-lightning 的基本概念和组件。

在实践中，在 NNI 中编写一个新的训练模块应继承 ``nni.retiarii.trainer.pytorch.lightning.LightningModule``，它将在 ``__init__`` 之后调用 ``set_model`` 函数，以将候选模型（由策略生成的）保存为 ``self.model``。 编写其余过程（如 ``training_step``）应与其他 lightning 模块相同。 Evaluators 还应该通过两个 API 调用与策略进行通讯（对于中间指标而言为 ``nni.report_intermediate_result``，对于最终指标而言为 ``nni.report_final_result``），分别被添加在 ``on_validation_epoch_end`` 和 ``teardown`` 中。 

示例如下。

.. code-block:: python

    from nni.retiarii.evaluator.pytorch.lightning import LightningModule  # please import this one

    @basic_unit
    class AutoEncoder(LightningModule):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 28*28)
            )

        def forward(self, x):
            embedding = self.model(x)  # let's search for encoder
            return embedding

        def training_step(self, batch, batch_idx):
            # training_step 定义了训练循环
            # 它独立于 forward 函数
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.model(x)  # 模型是一个被搜索的模型
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            # 默认日志记录到 TensorBoard
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.model(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log('val_loss', loss)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

        def on_validation_epoch_end(self):
            nni.report_intermediate_result(self.trainer.callback_metrics['val_loss'].item())

        def teardown(self, stage):
            if stage == 'fit':
                nni.report_final_result(self.trainer.callback_metrics['val_loss'].item())

然后，用户需要将所有东西（包括 LightningModule、trainer 和 dataloaders）包装成一个 ``Lightning`` 对象，并将这个对象传递给 Retiarii Experiment。

.. code-block:: python

    import nni.retiarii.evaluator.pytorch.lightning as pl
    from nni.retiarii.experiment.pytorch import RetiariiExperiment

    lightning = pl.Lightning(AutoEncoder(),
                             pl.Trainer(max_epochs=10),
                             train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                             val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))
    experiment = RetiariiExperiment(base_model, lightning, mutators, strategy)
