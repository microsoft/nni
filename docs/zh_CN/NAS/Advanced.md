# 自定义 NAS 算法

## 扩展 One-Shot Trainer

如果要在真实任务上使用 Trainer，还需要更多操作，如分布式训练，低精度训练，周期日志，写入 TensorBoard，保存检查点等等。 如前所述，一些 Trainer 支持了上述某些功能。 有两种方法可往已有的 Trainer 中加入功能：继承已有 Trainer 并重载，或复制已有 Trainer 并修改。

无论哪种方法，都需要实现新的 Trainer。 基本上，除了新的 Mutator 的概念，实现 One-Shot Trainer 与普通的深度学习 Trainer 相同。 因此，有两处会有所不同：

* 初始化

```python
model = Model()
mutator = MyMutator(model)
```

* 训练

```python
for _ in range(epochs):
    for x, y in data_loader:
        mutator.reset()  # 在模型中重置所有 Choice
        out = model(x)  # 与普通模型相同
        loss = criterion(out, y)
        loss.backward()
        # 以下代码没有不同
```

要展示 Mutator 的用途，需要先了解 One-Shot NAS 的工作原理。 通常 One-Shot NAS 会同时优化模型权重和架构权重。 它会反复的：对架构采样，或由超网络中的几种架构组成，然后像普通深度学习模型一样训练，将训练后的参数更新到超网络中，然后用指标或损失作为信号来指导架构的采样。 Mutator，在这里用作架构采样，通常会是另一个深度学习模型。 因此，可将其看作一个通过定义参数，并使用优化器进行优化的任何模型。 Mutator 是由一个模型来初始化的。 一旦 Mutator 绑定到了某个模型，就不能重新绑定到另一个模型上。

`mutator.reset()` 是关键步骤。 这一步确定了模型最终的所有 Choice。 重置的结果会一直有效，直到下一次重置来刷新数据。 重置后，模型可看作是普通的模型来进行前向和反向传播。

最后，Mutator 会提供叫做 `mutator.export()` 的方法来将模型的架构参数作为 dict 导出。 注意，当前 dict 是从 Mutable 键值到选择张量的映射。 为了存储到 JSON，用户需要将张量显式的转换为 Python 的 list。

同时，NNI 提供了工具，能更容易地实现 Trainer。 参考 [Trainer](./NasReference.md) 了解详情。

## 实现新的 Mutator

这是为了演示 `mutator.reset()` 和 `mutator.export()` 的伪代码。

```python
def reset(self):
    self.apply_on_model(self.sample_search())
```

```python
def export(self):
    return self.sample_final()
```

重置时，新架构会通过 `sample_search()` 采样，并应用到模型上。 然后，对模型进行一步或多步的搜索。 导出时，新架构通过 `sample_final()` 来采样，**不对模型做操作**。 可用于检查点或导出最终架构。

`sample_search()` 和 `sample_final()` 返回值的要求一致：从 Mutable 键值到张量的映射。 张量可以是 BoolTensor （true 表示选择，false 表示没有），或 FloatTensor 将权重应用于每个候选对象。 选定的分支会被计算出来（对于 `LayerChoice`，模型会被调用；对于 `InputChoice`，只有权重），并通过 Choice 的剪枝操作来剪枝模型。 这是 Mutator 实现的示例，大多数算法只需要关心前面部分。

```python
class RandomMutator(Mutator):
    def __init__(self, model):
        super().__init__(model)  # 记得调用 super
        # 别的操作

    def sample_search(self):
        result = dict()
        for mutable in self.mutables:  # 这是用户模型中所有 Mutable 模块
            # 共享同样键值的 Mutable 会去重
            if isinstance(mutable, LayerChoice):
                # 决定此模型会选择 `gen_index`
                gen_index = np.random.randint(mutable.length)
                result[mutable.key] = torch.tensor([i == gen_index for i in range(mutable.length)], 
                                                   dtype=torch.bool)
            elif isinstance(mutable, InputChoice):
                if mutable.n_chosen is None:  # n_chosen 是 None，表示选择所有数字
                    result[mutable.key] = torch.randint(high=2, size=(mutable.n_candidates,)).view(-1).bool()
                # 其它
        return result

    def sample_final(self):
        return self.sample_search()  # 使用同样的逻辑 其它操作
```

随机 Mutator 的完整示例在[这里](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/nas/pytorch/random/mutator.py)。

对于高级用法，例如，需要在 `LayerChoice` 执行的时候操作模型，可继承 `BaseMutator`，并重载 `on_forward_layer_choice` 和`on_forward_input_choice`。这些是 `LayerChoice` 和 `InputChoice` 对应的回调实现。 还可使用属性 `mutables` 来获得模型中所有的 `LayerChoice` 和 `InputChoice`。 详细信息，[参考这里](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch)。

```eval_rst
.. tip::
    用于调试的随机 Mutator。 使用

    .. code-block:: python

        mutator = RandomMutator(model)
        mutator.reset()

    会立刻从搜索空间中选择一个来激活。
```

## 实现分布式 NAS Tuner

在学习编写分布式 NAS Tuner前，应先了解如何写出通用的 Tuner。 阅读[自定义 Tuner](../Tuner/CustomizeTuner.md) 的教程。

当调用 "[nnictl ss_gen](../Tutorial/Nnictl.md)" 时，会生成下面这样的搜索空间文件：

```json
{
    "key_name": {
        "_type": "layer_choice",
        "_value": ["op1_repr", "op2_repr", "op3_repr"]
    },
    "key_name": {
        "_type": "input_choice",
        "_value": {
            "candidates": ["in1_key", "in2_key", "in3_key"],
            "n_chosen": 1
        }
    }
}
```

这是 Tuner 在 `update_search_space` 中会收到的搜索空间。 Tuner 需要解析搜索空间，并在 `generate_parameters` 中生成新的候选。 有效的 "参数" 格式如下：

```json
{
    "key_name": {
        "_value": "op1_repr",
        "_idx": 0
    },
    "key_name": {
        "_value": ["in2_key"],
        "_idex": [1]
    }
}
```

和普通超参优化 Tuner 类似，通过 `generate_parameters` 来发送。 参考 [SPOS](./SPOS.md) 示例代码。
