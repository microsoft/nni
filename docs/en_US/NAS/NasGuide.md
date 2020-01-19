# Guide: Using NAS on NNI

```eval_rst
.. contents::

.. Note:: The APIs are in an experimental stage. The current programing interface is subject to change.
```

![](../../img/nas_abstract_illustration.png)

Modern Neural Architecture Search (NAS) methods usually incorporate [three dimensions][1]: search space, search strategy, and performance estimation strategy. Search space often contains a limited neural network architectures to explore, while search strategy samples architectures from search space, gets estimations of their performance, and evolves itself. Ideally, search strategy should find the best architecture in the search space and report it to users. After users obtain such "best architecture", many methods use a "retrain step", which trains the network with the same pipeline as any traditional model.

## Implement a Search Space

Assuming now we've got a baseline model, what should we do to be empowered with NAS? Take [MNIST on PyTorch](https://github.com/pytorch/examples/blob/master/mnist/main.py) as an example, the story looks like this:

```python
from nni.nas.pytorch import mutables

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = mutables.LayerChoice([
            nn.Conv2d(1, 32, 3, 1),
            nn.Conv2d(1, 32, 5, 3)
        ])  # try 3x3 kernel and 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # ... same as original ...
        return output
```

The example above adds an option of choosing conv5x5 at conv1. The modification is as simple as declaring a `LayerChoice` with original conv3x3 and a new conv5x5 as its parameter. That's it! You don't have to modify the forward function in anyway. You can imagine conv1 as any another module without NAS.

So how about the possibilities of connections? This can be done by `InputChoice`. To allow for a skipconnection on an MNIST example, we add another layer called conv3. In the following example, a possible connection from conv2 is added to the output of conv3.

```python
from nni.nas.pytorch import mutables

class Net(nn.Module):
    def __init__(self):
        # ... same ...
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 1, 1)
        # declaring that there is exactly one candidate to choose from
        # search strategy will choose one or None
        self.skipcon = mutables.InputChoice(n_candidates=1)
        # ... same ...

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x0 = self.skipcon([x])  # choose one or none from [x]
        x = self.conv3(x)
        if x0 is not None:  # skipconnection is open
            x += x0
        x = F.max_pool2d(x, 2)
        # ... same ...
        return output
```

The general purpose of InputChoice is a callable module that receives a list of tensors and output the concatenation/sum/mean of some of them, or `None` if none is selected. As a module, we **need to initialize `InputChoice` in `__init__`**, and use it in `forward`. We will see later that this is to allow search algorithms to identify these choices, and do necessary preparation.

Users can specify a **key** for each mutable. By default NNI will assign one for you that is globally unique, but in case users want to share choices (for example, there are two `LayerChoice` with the same candidate operations, and you want them to have the same choice, i.e., if first one chooses the i-th op, the second one also chooses the i-th op), they can give them the same key. The key marks the identity for this choice, and will be used in dumped checkpoint. So if you want to increase the readability of your exported architecture, manually assigning keys to each mutable would be a good idea.

`LayerChoice` and `InputChoice` are both **mutable**s. For advanced usage on mutables, see [Mutables](./NasReference.md#mutables). 

## Use a Search Algorithm

Different in how the search space is explored and trials are spawned, there are at least two different ways users can do search. One runs NAS distributedly, which can be as naive as enumerating all the architectures and training each one from scratch, or leveraging more advanced technique. Since training many different architectures are known to be expensive, another family of methods, called one-shot NAS, builds a supernet containing every candidate in the search space as its subnetwork, and in each step a subnetwork or combination of several subnetworks is trained.

Currently, several one-shot NAS methods have been supported on NNI. For example, `DartsTrainer` which uses SGD to train architecture weights and model weights iteratively, `ENASTrainer` which [uses a controller to train the model][2]. New and more efficient NAS trainers keep emerging in research community.

### One-Shot NAS

Each one-shot NAS implements a trainer, which users can find detailed usages in the description of each algorithm. Here is a simple example, demonstrating how users can use `EnasTrainer`.

```python
# this is exactly same as traditional model training
model = Net()
dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)

# use NAS here
def top1_accuracy(output, target):
    # this is the function that computes the reward, as required by ENAS algorithm
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size

def metrics_fn(output, target):
    # metrics function receives output and target and computes a dict of metrics
    return {"acc1": reward_accuracy(output, target)}

from nni.nas.pytorch import enas
trainer = enas.EnasTrainer(model,
                           loss=criterion,
                           metrics=metrics_fn,
                           reward_function=top1_accuracy,
                           optimizer=optimizer,
                           batch_size=128
                           num_epochs=10,  # 10 epochs
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           log_frequency=10)  # print log every 10 steps
trainer.train()  # training
trainer.export(file="model_dir/final_architecture.json")  # export the final architecture to file
```

Users can directly run their training file by `python(3) train.py`, without `nnictl`. After training, users could export the best one of the found models through `trainer.export()`.

Normally, the trainer exposes a few things that you can customize, for example, loss function, metrics function, optimizer, and datasets. These should satisfy the needs from most usages, and we do our best to make sure our built-in trainers work on as many models, tasks and datasets as possible. But there is no guarantee. For example, some trainers have assumption that the task has to be a classification task; some trainers might have a different definition of "epoch" (e.g., an ENAS epoch = some child steps + some controller steps). So after a few tryouts, if you want to actually use the trainers on your very customized applications, you might very soon need to [customize your trainer](#extend-the-ability-of-one-shot-trainers).

### Distributed NAS

Neural architecture search is originally executed by running each child model independently as a trial job. We also support this searching approach, and it naturally fits in NNI hyper-parameter tuning framework, where tuner generates child model for next trial and trials run in training service.

To use this mode, there is no need to change the search space expressed with NNI NAS API (i.e., `LayerChoice`, `InputChoice`, `MutableScope`). After the model is initialized, apply the function `get_and_apply_next_architecture` on the model. One-shot NAS trainers are not used in this mode. Here is a simple example:

```python
model = Net()

# get the chosen architecture from tuner and apply it on model
get_and_apply_next_architecture(model)
train(model)  # your code for training the model
acc = test(model)  # test the trained model
nni.report_final_result(acc)  # report the performance of the chosen architecture
```

The search space should be automatically generated and sent to tuner. As with NNI NAS API the search space is embedded in user code, users could use "[nnictl ss_gen](../Tutorial/Nnictl.md)" to generate search space file. Then, put the path of the generated search space in the field `searchSpacePath` of `config.yml`. The other fields in `config.yml` can be filled by referring [this tutorial](../Tutorial/QuickStart.md).

You could use [NNI tuners](../Tuner/BuiltinTuner.md) to do the search. Currently, only PPO Tuner supports NAS search space.

We support standalone mode for easy debugging, where you could directly run the trial command without launching an NNI experiment. This is for checking whether your trial code can correctly run. The first candidate(s) are chosen for `LayerChoice` and `InputChoice` in this standalone mode.

A complete example can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/classic_nas/config_nas.yml).

### Retrain with Exported Architecture

After the searching phase, it's time to train the architecture found. Unlike many open-source NAS algorithms who write a whole new model specifically for retraining. We found that searching model and retraining model are usual very similar, and therefore you can construct your final model with the exact model code. For example

```python
model = Net()
apply_fixed_architecture(model, "model_dir/final_architecture.json")
```

After applying, the model is then fixed and ready for a final training. The model works as a single model, although it might contain more parameters than expected. For deeper reasons and possible workaround, see [Trainers](./NasReference.md#retrain).

## Customize a Search Algorithm

### Extend the Ability of One-Shot Trainers

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

Mutators are designed to be one-to-one to models. One model needs one mutator and one mutator is initialized with one and only one model. Once a mutator is binded to a model, it cannot be rebinded to another model. Mutator is a `nn.Module` so it can have its own parameters -- in fact that's exactly what most algorithms did: they co-optimize parameters in model and parameters in mutator alternatively.

`mutator.reset()` is the core step. That's where all the choices in the model are finalized until next reset. After the reset, the model can be seen as a traditional model to do forward-pass and backward-pass.

Finally, mutators provide a method called `mutator.export()` that export a dict with architectures to the model. Note that currently this dict this a mapping from keys of mutables to tensors of selection. So in order to dump to json, users need to convert the tensors explicitly into python list.

Meanwhile, NNI provides some useful tools so that users can implement trainers more easily. See [Trainers](./NasReference.md#trainers) for details.

### Invent New Mutators

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

The complete example of random mutator can be found [here](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/nas/pytorch/random/mutator.py).

For advanced usages, e.g., users want to manipulate the way modules in `LayerChoice` are executed, they can inherit `BaseMutator`, and overwrite `on_forward_layer_choice` and `on_forward_input_choice`, which are the callback implementation of `LayerChoice` and `InputChoice` respectively. Users can still use property `mutables` to get all `LayerChoice` and `InputChoice` in the model code. For details, please refer to [reference](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch) here to learn more.

```eval_rst
.. tip::
    A useful application of random mutator is for debugging. Use
   
    .. code-block:: python

        mutator = RandomMutator(model)
        mutator.reset()

    will immediately set one possible candidate in the search space as the active one.
```

### Search with Distribution

Behind the implementation of distributed NAS, is that NNI automatically generate a search space in JSON format from users' code, which looks like the example below:

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

For details on how to write a general tuner, read [Customize Tuner](../Tuner/CustomizeTuner.md).

[1]: https://arxiv.org/abs/1808.05377
[2]: https://arxiv.org/abs/1802.03268
