自定义 Trainer
=======================

Trainer 对评估新模型的性能是必要的。 在 NAS 场景中，Trainer 进一步分为两类：

1. **Single-arch trainers**：用于训练和评估单个模型的 Trainer。
2. **One-shot trainers**：端到端同时处理训练和搜索的 Trainer。

Single-arch trainers
--------------------

使用 PyTorch-Lightning
^^^^^^^^^^^^^^^^^^^^^^

NNI 建议以 PyTorch-Lightning 风格编写训练代码，即编写一个 LightningModule，定义训练所需的所有元素（例如 loss function、optimizer），并定义一个 Trainer，使用 dataloader 来执行训练（可选）。 在此之前，请阅读 `PyTorch-lightning 文档 <https://pytorch-lightning.readthedocs.io/>` 了解 PyTorch-lightning 的基本概念和组件。

在实践中，在 NNI 中编写一个新的训练模块应继承 ``nni.retiarii.trainer.pytorch.lightning.LightningModule``，它将在 ``__init__`` 之后调用 ``set_model`` 函数，以将候选模型（由策略生成的）保存为 ``self.model``。 编写其余过程（如 ``training_step``）应与其他 lightning 模块相同。 Trainer 还应该通过两个 API 调用与策略进行通讯（对于中间指标而言为 ``nni.report_intermediate_result``，对于最终指标而言为 ``nni.report_final_result``），分别被添加在 ``on_validation_epoch_end`` 和 ``teardown`` 中。 

示例如下。

.. code-block::python

    from nni.retiarii.trainer.pytorch.lightning import LightningModule  # please import this one

    @blackbox_module
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
            z = self.model(x)  # model is the one that is searched for
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

.. code-block::python

    import nni.retiarii.trainer.pytorch.lightning as pl
    from nni.retiarii.experiment.pytorch import RetiariiExperiment

    lightning = pl.Lightning(AutoEncoder(),
                             pl.Trainer(max_epochs=10),
                             train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                             val_dataloaders=pl.DataLoader(test_dataset, batch_size=100))
    experiment = RetiariiExperiment(base_model, lightning, mutators, strategy)

使用 FunctionalTrainer
^^^^^^^^^^^^^^^^^^^^^^

还有另一种使用功能性 API 自定义新 Trainer 的方法，该方法提供了更大的灵活性。 用户只需要编写一个 fit 函数来包装所有内容。 此函数接收一个位置参数（model）和可能的关键字参数。 通过这种方式，用户可以控制一切，但向框架公开的信息较少，因此可能进行优化的机会也较少。 示例如下。

.. code-block::python

    from nni.retiarii.trainer import FunctionalTrainer
    from nni.retiarii.experiment.pytorch import RetiariiExperiment

    def fit(model, dataloader):
        train(model, dataloader)
        acc = test(model, dataloader)
        nni.report_final_result(acc)

    trainer = FunctionalTrainer(fit, dataloader=DataLoader(foo, bar))
    experiment = RetiariiExperiment(base_model, trainer, mutators, strategy)


One-shot trainers
-----------------

One-shot Trainer 应继承 ``nni.retiarii.trainer.BaseOneShotTrainer``，并需要实现``fit()`` 函数（用于进行拟合和搜索过程）和 ``export()`` 方法（用于返回搜索到的最佳架构）。

编写一个 One-Shot Trainer 与经典 Trainer 有很大不同。 首先，init 方法参数没有限制，可以接收任何 Python 参数。 其次，输入到 One-Shot Trainer 中的模型可能带有 Retiarii 特定的模块（例如 LayerChoice 和 InputChoice）的模型。 这种模型不能直接向前传播，Trainer 需要决定如何处理这些模块。

一个典型的示例是 DartsTrainer，其中可学习参数用于在 LayerChoice 中组合多个 Choice。 Retiarii为模块替换提供了易于使用的函数，即 ``replace_layer_choice``, ``replace_input_choice``。 示例如下。 

.. code-block::python

    from nni.retiarii.trainer.pytorch import BaseOneShotTrainer
    from nni.retiarii.trainer.pytorch.utils import replace_layer_choice, replace_input_choice


    class DartsLayerChoice(nn.Module):
        def __init__(self, layer_choice):
            super(DartsLayerChoice, self).__init__()
            self.name = layer_choice.key
            self.op_choices = nn.ModuleDict(layer_choice.named_children())
            self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

        def forward(self, *args, **kwargs):
            op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
            alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
            return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)


    class DartsTrainer(BaseOneShotTrainer):

        def __init__(self, model, loss, metrics, optimizer):
            self.model = model
            self.loss = loss
            self.metrics = metrics
            self.num_epochs = 10

            self.nas_modules = []
            replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)

            ... # 初始化 dataloaders 和 optimizers

        def fit(self):
            for i in range(self.num_epochs):
                for (trn_X, trn_y), (val_X, val_y) in zip(self.train_loader, self.valid_loader):
                    self.train_architecture(val_X, val_y)
                    self.train_model_weight(trn_X, trn_y)

        @torch.no_grad()
        def export(self):
            result = dict()
            for name, module in self.nas_modules:
                if name not in result:
                    result[name] = select_best_of_module(module)
            return result

Retsarii 源代码提供了 DartsTrainer 的完整代码。 请查看 :githublink:`nni/retiarii/trainer/pytorch/darts.py`.
