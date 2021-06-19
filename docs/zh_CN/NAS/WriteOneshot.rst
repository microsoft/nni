自定义 One-shot Trainer
=========================================

One-shot Trainer 应继承 ``nni.retiarii.oneshot.BaseOneShotTrainer``，并需要实现``fit()`` 函数（用于进行拟合和搜索过程）和 ``export()`` 方法（用于返回搜索到的最佳架构）。

编写一个 One-Shot Trainer 与单个结构的 evaluator 有很大不同。 首先，init 方法参数没有限制，可以接收任何 Python 参数。 其次，输入到 One-Shot Trainer 中的模型可能带有 Retiarii 特定的模块（例如 LayerChoice 和 InputChoice）的模型。 这种模型不能直接向前传播，Trainer 需要决定如何处理这些模块。

一个典型的示例是 DartsTrainer，其中可学习参数用于在 LayerChoice 中组合多个 Choice。 Retiarii为模块替换提供了易于使用的函数，即 ``replace_layer_choice``, ``replace_input_choice``。 示例如下。 

.. code-block:: python

    from nni.retiarii.oneshot import BaseOneShotTrainer
    from nni.retiarii.oneshot.pytorch import replace_layer_choice, replace_input_choice


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

Retsarii 源代码提供了 DartsTrainer 的完整代码。 请参考 :githublink:`DartsTrainer <nni/retiarii/oneshot/pytorch/darts.py>`。
