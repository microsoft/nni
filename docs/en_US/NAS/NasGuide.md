# Guide: Using NAS on NNI

```eval_rst
.. Note:: We are trying to support various NAS algorithms with unified programming interface, and it's in an experimental stage, which means the current programing interface may be updated in future.
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

The example above adds an option of choosing conv5x5 at conv1. The modification is as simple as declaring a `LayerChoice` with original conv3x3 and a new conv5x5 as its parameter. That's it! **You don't have to modify the forward function in anyway. You can imagine conv1 as any another module without NAS.

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

For advanced usage on `LayerChoice` and `InputChoice` and more details, see [Mutables](./NasMutables.md). 

## Use a Search Algorithm

### One-Shot NAS

### Distribution NAS

## Customize a Search Algorithm

### Extend the Ability of One-Shot Trainers

### Invent New Mutators and Trainers

### Search with Distribution


[1]: https://arxiv.org/abs/1808.05377
