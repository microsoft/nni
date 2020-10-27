# Customize a NAS Algorithm

## Extend the Ability of One-Shot Trainers

Users might want to do multiple things if they are using the trainers on real tasks, for example, distributed training, half-precision training, logging periodically, writing tensorboard, dumping checkpoints and so on. As mentioned previously, some trainers do have support for some of the items listed above; others might not. Generally, there are two recommended ways to add anything you want to an existing trainer: inherit an existing trainer and override, or copy an existing trainer and modify.

Either way, you are walking into the scope of implementing a new trainer. Basically, implementing a one-shot trainer is no different from any traditional deep learning trainer, except that a new concept called mutator will reveal itself. So that the implementation will be different in at least two places:

* Initialization

```python
model = Model()
mutator = MyMutator(model)
```

* Training

```python
for _ in range(epochs):
    for x, y in data_loader:
        mutator.reset()  # reset all the choices in model
        out = model(x)  # like traditional model
        loss = criterion(out, y)
        loss.backward()
        # no difference below
```

To demonstrate what mutators are for, we need to know how one-shot NAS normally works. Usually, one-shot NAS "co-optimize model weights and architecture weights". It repeatedly: sample an architecture or combination of several architectures from the supernet, train the chosen architectures like traditional deep learning model, update the trained parameters to the supernet, and use the metrics or loss as some signal to guide the architecture sampler. The mutator, is the architecture sampler here, often defined to be another deep-learning model. Therefore, you can treat it as any model, by defining parameters in it and optimizing it with optimizers. One mutator is initialized with exactly one model. Once a mutator is binded to a model, it cannot be rebinded to another model.

`mutator.reset()` is the core step. That's where all the choices in the model are finalized. The reset result will be always effective, until the next reset flushes the data. After the reset, the model can be seen as a traditional model to do forward-pass and backward-pass.

Finally, mutators provide a method called `mutator.export()` that export a dict with architectures to the model. Note that currently this dict this a mapping from keys of mutables to tensors of selection. So in order to dump to json, users need to convert the tensors explicitly into python list.

Meanwhile, NNI provides some useful tools so that users can implement trainers more easily. See [Trainers](./NasReference.md) for details.

## Implement New Mutators

To start with, here is the pseudo-code that demonstrates what happens on `mutator.reset()` and `mutator.export()`.

```python
def reset(self):
    self.apply_on_model(self.sample_search())
```

```python
def export(self):
    return self.sample_final()
```

On reset, a new architecture is sampled with `sample_search()` and applied on the model. Then the model is trained for one or more steps in search phase. On export, a new architecture is sampled with `sample_final()` and **do nothing to the model**. This is either for checkpoint or exporting the final architecture.

The requirements of return values of `sample_search()` and `sample_final()` are the same: a mapping from mutable keys to tensors. The tensor can be either a BoolTensor (true for selected, false for negative), or a FloatTensor which applies weight on each candidate. The selected branches will then be computed (in `LayerChoice`, modules will be called; in `InputChoice`, it's just tensors themselves), and reduce with the reduction operation specified in the choices. For most algorithms only worrying about the former part, here is an example of your mutator implementation.

```python
class RandomMutator(Mutator):
    def __init__(self, model):
        super().__init__(model)  # don't forget to call super
        # do something else

    def sample_search(self):
        result = dict()
        for mutable in self.mutables:  # this is all the mutable modules in user model
            # mutables share the same key will be de-duplicated
            if isinstance(mutable, LayerChoice):
                # decided that this mutable should choose `gen_index`
                gen_index = np.random.randint(mutable.length)
                result[mutable.key] = torch.tensor([i == gen_index for i in range(mutable.length)], 
                                                   dtype=torch.bool)
            elif isinstance(mutable, InputChoice):
                if mutable.n_chosen is None:  # n_chosen is None, then choose any number
                    result[mutable.key] = torch.randint(high=2, size=(mutable.n_candidates,)).view(-1).bool()
                # else do something else
        return result

    def sample_final(self):
        return self.sample_search()  # use the same logic here. you can do something different
```

The complete example of random mutator can be found [here](https://github.com/microsoft/nni/blob/v1.9/src/sdk/pynni/nni/nas/pytorch/random/mutator.py).

For advanced usages, e.g., users want to manipulate the way modules in `LayerChoice` are executed, they can inherit `BaseMutator`, and overwrite `on_forward_layer_choice` and `on_forward_input_choice`, which are the callback implementation of `LayerChoice` and `InputChoice` respectively. Users can still use property `mutables` to get all `LayerChoice` and `InputChoice` in the model code. For details, please refer to [reference](https://github.com/microsoft/nni/tree/v1.9/src/sdk/pynni/nni/nas/pytorch) here to learn more.

```eval_rst
.. tip::
    A useful application of random mutator is for debugging. Use
   
    .. code-block:: python

        mutator = RandomMutator(model)
        mutator.reset()

    will immediately set one possible candidate in the search space as the active one.
```

## Implemented a Distributed NAS Tuner

Before learning how to write a distributed NAS tuner, users should first learn how to write a general tuner. read [Customize Tuner](../Tuner/CustomizeTuner.md) for tutorials.

When users call "[nnictl ss_gen](../Tutorial/Nnictl.md)" to generate search space file, a search space file like this will be generated:

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

This is the exact search space tuners will receive in `update_search_space`. It's then tuners' responsibility to interpret the search space and generate new candidates in `generate_parameters`. A valid "parameters" will be in the following format:

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

Send it through `generate_parameters`, and the tuner would look like any HPO tuner. Refer to [SPOS](./SPOS.md) example code for an example.
