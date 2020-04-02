# NAS Visualization (Experimental)

## Guide

Currently NAS visualization needs to customize trainer. If you don't know how, please read this [doc](./Advanced.md#extend-the-ability-of-one-shot-trainers).

The workflow of NAS visualization is:

* The trainer writes two files, `graph.json` and `log` to any directory (we will refer to it as `logdir`).
* Meanwhile (experiment can be still running, but make sure `graph.json` exists), launch NAS UI with `nnictl webui nas --logdir /path/to/your/logdir --port <port>`.

We provide an example of customizing trainers here:

```python
class MyTrainer(DartsTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_writer = open("/path/to/your/logdir/log", "w")

    def _logits_and_loss(self, X, y):
        self.mutator.reset()
        logits = self.model(X)
        loss = self.loss(logits, y)
        print(json.dumps(self.mutator.status()), file=self.status_writer)
        self.status_writer.flush()
        return logits, loss


model = some_model()
model.cuda()
mutator = DartsMutator(model)
vis_graph = mutator.graph(inputs)
# `inputs` is a dummy input to your model. For example, torch.randn((1, 3, 32, 32)).cuda()
# If your model has multiple inputs, it should be a tuple.
with open("/path/to/your/logdir/graph.json", "w") as f:
    json.dump(vis_graph, f)

# load dataset and train
```

## NAS UI Preview

![](../../img/nasui-1.png)

![](../../img/nasui-2.png)

## Limitations

* We rely on PyTorch support for tensorboard for graph export, which relies on `torch.jit`. It will not work if your model doesn't support `jit`.
* There are known performance issues when loading a moderate-size graph with many op choices (like DARTS search space).

## Feedback

NAS UI is currently experimental. We welcome your feedback. [Here](https://github.com/microsoft/nni/pull/2085) we have listed all the to-do items of NAS UI in the future. Feel free to comment (or [submit a new issue](https://github.com/microsoft/nni/issues/new?template=enhancement.md)) if you have other suggestions.
