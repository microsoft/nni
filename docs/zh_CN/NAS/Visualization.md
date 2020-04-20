# NAS 可视化（测试版）

## 内置 Trainer 支持

当前，仅 ENAS 和 DARTS 支持可视化。 [ENAS](./ENAS.md) 和 [DARTS](./DARTS.md) 的示例演示了如何在代码中启用可视化，其需要在 `trainer.train()` 前添加代码。

```python
trainer.enable_visualization()
```

此代码会在当前目录中创建新目录 `logs/<current_time_stamp>`，并创建两个新文件 `graph.json` 和 `log`。

不必等到程序运行完后，再启动 NAS 界面，但需要确保这两个文件产生后，再启动。 启动 NAS 界面：

```bash
nnictl webui nas --logdir logs/<current_time_stamp> --port <port>
```

## 可视化定制的 Trainer

如果要定制 Trainer，参考[文档](./Advanced.md#extend-the-ability-of-one-shot-trainers)。

需要对已有 Trainer 代码做两处改动来支持可视化：

1. 在训练前导出图：

```python
vis_graph = self.mutator.graph(inputs)
# `inputs` 是模型的虚拟输入。 例如，torch.randn((1, 3, 32, 32)).cuda()
# 如果模型有多个输入，则要使用 tuple。
with open("/path/to/your/logdir/graph.json", "w") as f:
    json.dump(vis_graph, f)
```

2. 记录选择的 Choice。 可以每个 Epoch，批处理或任何频率下做次记录。

```python
def __init__(self):
    # ...
    self.status_writer = open("/path/to/your/logdir/log", "w")  # create a writer

def train(self):
    # ...
    print(json.dumps(self.mutator.status()), file=self.status_writer, flush=True)  # 保存状态
```

如果继承 `Trainer`，实现定制的 Trainer。 NNI 提供了 `enable_visualization()` 和 `_write_graph_status()` 来简化可视化。 All you need to do is calling `trainer.enable_visualization()` before start, and `trainer._write_graph_status()` each time you want to do the logging. But remember both of these APIs are experimental and subject to change in future.

Last but not least, invode NAS UI with

```bash
nnictl webui nas --logdir /path/to/your/logdir
```

## NAS UI Preview

![](../../img/nasui-1.png)

![](../../img/nasui-2.png)

## Limitations

* NAS visualization only works with PyTorch >=1.4. We've tested it on PyTorch 1.3.1 and it doesn't work.
* We rely on PyTorch support for tensorboard for graph export, which relies on `torch.jit`. It will not work if your model doesn't support `jit`.
* There are known performance issues when loading a moderate-size graph with many op choices (like DARTS search space).

## Feedback

NAS UI is currently experimental. We welcome your feedback. [Here](https://github.com/microsoft/nni/pull/2085) we have listed all the to-do items of NAS UI in the future. Feel free to comment (or [submit a new issue](https://github.com/microsoft/nni/issues/new?template=enhancement.md)) if you have other suggestions.
